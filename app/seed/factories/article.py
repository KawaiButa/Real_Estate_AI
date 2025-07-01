import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from sqlalchemy import select
from seed.factories.base import BaseFactory
from database.models.article import Article
from database.models.tag import Tag
from domains.news.service import ArticleRepository
from configs.sqlalchemy import sqlalchemy_config
from sqlalchemy.ext.asyncio import AsyncSession
from configs.gemai import client
from google.genai.types import GenerateContentConfig
from advanced_alchemy.utils.text import slugify
from transformers import pipeline
import re

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]
_SUMMARY_PIPELINE = pipeline(
    "summarization",
    model="google/long-t5-tglobal-base",
    tokenizer="google/long-t5-tglobal-base",
    device=-1,
)

_KEYPHRASE_PIPELINE = pipeline(
    "text2text-generation",
    model="google/long-t5-tglobal-base",
    tokenizer="google/long-t5-tglobal-base",
    framework="pt",
    device=-1,
)


async def generate_tags_and_summary(article_html_content: str) -> dict:
    """
    Summarize and extract tags using small transformer models.
    """
    text = re.sub(r"<[^>]+>", " ", article_html_content)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 50:
        return {"tags": [], "short_description": text}
    try:
        summary_out = _SUMMARY_PIPELINE(
            text,
            max_length=200,
            min_length=30,
            do_sample=False,
        )
        short_description = summary_out[0]["summary_text"].strip()
    except Exception as e:
        print(f"Summarization error: {e}")
        short_description = text[:300] + ("…" if len(text) > 300 else "")
    try:
        prompt = "extract keyphrases: " + text[:1000]  # limit length
        kpop = _KEYPHRASE_PIPELINE(prompt, max_length=64, do_sample=False)
        raw = kpop[0]["generated_text"]
        tags = re.split(r"[;,]\s*", raw)
        tags = list(dict.fromkeys([t.strip().lower() for t in tags if t.strip()]))
        tags = tags[:7]
    except Exception as e:
        print(f"Keyphrase extraction error: {e}")
        tags = []

    return {
        "short_description": short_description,
        "tags": tags,
    }


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

    async def seed(self, count: int = 10, fixture_path: str = "seed/fixtures") -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                if fixture_path:
                    await import_articles_from_json(
                        os.path.join(fixture_path, "articles.json"), session
                    )
                else:
                    articles_data = self.fetch_articles_from_openai(count)
                    for article_data in articles_data:
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

def parse_vietnamese_datetime(date_str: str) -> datetime | None:
    """
    Tries to parse common Vietnamese datetime string formats, including RFC 1123.
    Returns a timezone-aware datetime object (UTC) or None if parsing fails.
    """
    if not date_str or not isinstance(date_str, str):
        return None

    # First: handle ISO8601 with 'T' and timezone info
    if "T" in date_str and ("Z" in date_str or "+" in date_str or "-" in date_str[10:]):
        try:
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    # Try known formats, including RFC 1123
    formats_to_try = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%a, %d %b %Y %H:%M:%S GMT",  # RFC 1123 (e.g., "Sun, 01 Jun 2025 01:16:00 GMT")
    ]

    for fmt in formats_to_try:
        try:
            dt_naive = datetime.strptime(date_str.strip(), fmt)
            dt_aware = dt_naive.replace(tzinfo=timezone.utc)
            return dt_aware
        except ValueError:
            continue

    print(f"Warning: Could not parse date string: {date_str}")
    return None

async def get_or_create_tags(session: AsyncSession, tag_names: List[str]) -> List[Tag]:
    """
    Retrieves existing Tag objects or creates new ones for each tag name.
    """
    tags_to_return: List[Tag] = []
    for name in tag_names:
        if not name or not isinstance(name, str):
            print(f"Skipping invalid tag name: {name}")
            continue
        name_cleaned = name.strip().lower()
        if not name_cleaned:
            continue
        tag = (
            await Tag.as_unique_async(
                session=session,
                name=name,
                slug=slugify(name),
            )
        ).id
        tags_to_return.append(tag)
    return tags_to_return


async def process_article_data(session: AsyncSession, article_data: Dict[str, Any]):
    """
    Processes a single article dictionary, generates tags/summary, and prepares for DB insertion.
    """
    title = article_data.get("tieu_de")
    html_content = article_data.get("clean_html")
    source_name = article_data.get("nguon_tin")
    published_date_str = article_data.get("ngay_dang_tai_raw")

    if not all([title, html_content, source_name]):
        print(f"Skipping article due to missing essential data: {title or 'N/A'}")
        return None
    gemini_data = await generate_tags_and_summary(html_content)
    tag_names = gemini_data.get("tags", [])
    short_description = gemini_data.get("short_description")
    if not short_description:
        short_description = "Tóm tắt không có sẵn."
    if not tag_names:
        print(f"No tags generated for article: {title}")
    publish_date = parse_vietnamese_datetime(published_date_str)
    if publish_date is None:
        print(
            f"Using current time for article '{title}' due to unparseable date: {published_date_str}"
        )
        publish_date = datetime.now(timezone.utc)
    # db_tags = await get_or_create_tags(session, tag_names)
    new_article = Article(
        title=title,
        publish_date=publish_date,
        content=html_content,
        short_description=short_description[:499],
        author=source_name,
        tags=[],
    )
    return new_article


async def import_articles_from_json(json_filepath: str, session: AsyncSession):
    """
    Loads articles from a JSON file, processes them, and adds them to the database.
    """
    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            data_from_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_filepath}")
        return

    if not isinstance(data_from_json, list):
        if (
            isinstance(data_from_json, dict) and "tieu_de" in data_from_json
        ):  # Heuristic
            data_from_json = [data_from_json]
        else:
            print(
                "Error: JSON data is not a list of articles or a single article object."
            )
            return

    articles_to_add = []
    for i, item_data in enumerate(data_from_json):
        if i > 2:
            break
        print(f"\n--- Processing item {i+1}/{len(data_from_json)} ---")
        article_obj = await process_article_data(session, item_data)
        if article_obj:
            articles_to_add.append(article_obj)
            session.add(article_obj)

    if articles_to_add:
        print(f"\nCommitting {len(articles_to_add)} new articles to the database...")
    else:
        print("No new articles to add.")
    print("Import process finished.")
