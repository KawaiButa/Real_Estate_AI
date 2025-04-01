import json
import uuid
from datetime import datetime, timezone
from typing import List
import openai
from sqlalchemy import select
from seed.factories.base import BaseFactory
from database.models.article import Article
from database.models.tag import Tag
from domains.news.service import ArticleRepository
from configs.sqlalchemy import sqlalchemy_config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

client = openai.OpenAI()


class ArticleFactory(BaseFactory):
    repository = ArticleRepository

    def fetch_articles_from_openai(self, count: int) -> List[dict]:
        """
        Uses OpenAI's ChatCompletion to retrieve a list of real estate news articles.
        The LLM is prompted to act as an expert news aggregator for a tech-savvy audience
        of investors and industry professionals interested in Vietnamese real estate.

        The prompt instructs the model to:
         - Return exactly 'count' articles.
         - Output a JSON list where each article has: title, publish_date, content, short_description, author, and tags.
         - Use chain-of-thought reasoning internally, but only return the final answer without extra commentary.
         - Avoid including irrelevant information.
         - Provide output in clear, concise Vietnamese with a professional tone.
         - Limit the short_description to a maximum of 255 characters.

        An example output is provided as a guide.
        """
        prompt = (
            f"You are an expert real estate news aggregator for a tech-savvy audience of investors "
            f"and professionals interested in Vietnamese real estate. Your task is to retrieve the latest {count} high-quality, well-researched news articles from major Vietnamese news platforms. "
            f"Each article must be represented as an object in a JSON list with EXACTLY the following keys:\n"
            f"- 'title': a compelling and concise headline in Vietnamese.\n"
            f"- 'publish_date': an ISO 8601 formatted date string.\n"
            f"- 'content': the full HTML content of the article (including at least one <img> tag),\n"
            f"- 'short_description': a brief summary not exceeding 255 characters,\n"
            f"- 'author': the name of the journalist or news source,\n"
            f"- 'tags': an array of relevant keyword strings.\n\n"
            f"Do not include any extra fields or commentary. Return ONLY a valid JSON list. "
            f"Use clear, succinct language and maintain a professional tone. For example, a valid output is:\n\n"
            f"""[{{"title": "Bất động sản cao cấp tại Hà Nội: Cơ hội đầu tư hấp dẫn", "publish_date": "2023-04-15T09:30:00+07:00", "content": "<div><h2>Bất động sản cao cấp tại Hà Nội</h2><p>Nội dung bài báo...</p><img src='https://example.com/image.jpg'/></div>", "short_description": "Tóm tắt cơ hội đầu tư bất động sản cao cấp tại Hà Nội.", "author": "Nguyễn Văn A", "tags": ["bất động sản", "Hà Nội", "đầu tư"]}}]"""
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=10000,
            )
            text = response.choices[0].message.content.strip()
            print(text)
            articles = json.loads(text)
            if not isinstance(articles, list):
                raise ValueError("Expected a JSON list of articles.")
            return articles
        except Exception as e:
            print(f"Error fetching articles from OpenAI: {e}")
            return []

    async def get_or_create_tags(
        self, session: AsyncSession, tag_names: List[str]
    ) -> List[Tag]:
        """
        Retrieves existing Tag objects or creates new ones for each tag name.
        """
        tags = []
        for name in tag_names:
            result = await session.execute(select(Tag).filter_by(name=name))
            tag = result.scalars().first()
            if not tag:
                tag = Tag(id=uuid.uuid4(), name=name)
                session.add(tag)
            tags.append(tag)
        return tags

    async def seed(self, count: int = 10) -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                articles_data = self.fetch_articles_from_openai(count)
                for article_data in articles_data:
                    # Avoid duplicates by checking if an article with the same title exists.
                    result = await session.execute(
                        select(Article).filter_by(title=article_data.get("title"))
                    )
                    if result.scalars().first():
                        continue

                    publish_date_str = article_data.get("publish_date")
                    try:
                        publish_date = datetime.fromisoformat(publish_date_str)
                    except Exception:
                        publish_date = datetime.now(timezone.utc)

                    tag_names = article_data.get("tags", [])
                    tags = await self.get_or_create_tags(session, tag_names)

                    article = Article(
                        id=uuid.uuid4(),
                        title=article_data.get("title"),
                        publish_date=publish_date,
                        content=article_data.get("content"),
                        short_description=article_data.get("short_description"),
                        author=article_data.get("author"),
                        tags=tags,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    await self.repository(session=session).add(article)
            except Exception as e:
                await session.rollback()
                print(f"Error during ArticleFactory seeding: {e}")
                raise e
            finally:
                await session.commit()

    async def drop_all(self) -> None:
        async with sqlalchemy_config.get_session() as session:
            await self.repository(session=session).delete_where(Article.id.is_not(None))
            await session.commit()
