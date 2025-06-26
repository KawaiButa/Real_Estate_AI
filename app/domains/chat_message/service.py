from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Dict, List
import uuid
from sqlalchemy.dialects import postgresql  # or mysql, sqlite depending on your DB
from sqlalchemy import and_, desc, or_, select
from sqlalchemy.orm import noload
from transformers import pipeline

from database.models.property import Property
from domains.properties.service import PropertyService
from domains.chat_session.service import ChatSessionService
from database.models.chat_message import ChatMessage, ChatMessageSchema
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from database.models.chat_session import ChatSession
from domains.chat_message.dtos import AskAIDTO, CreateMessageDTO
from domains.image.service import ImageService
from domains.supabase.service import SupabaseService, provide_supabase_service
from sqlalchemy.ext.asyncio import AsyncSession
from google.genai import types
from configs.gemai import client
import re
from litestar.exceptions import ValidationException, InternalServerException
import requests
from google.genai.types import (
    Tool,
    GenerateContentConfig,
    GoogleSearch,
    UserContent,
    ModelContent,
)
from litestar.repository.filters import LimitOffset
from litestar.pagination import OffsetPagination


class ChatMessageRepository(SQLAlchemyAsyncRepository[ChatMessage]):
    model_type = ChatMessage


class ChatMessageService(SQLAlchemyAsyncRepositoryService[ChatMessage]):
    repository_type = ChatMessageRepository
    supabase_service: SupabaseService = provide_supabase_service(bucket_name="chat")

    async def create_message(
        self, data: CreateMessageDTO, user_id: uuid.UUID, auto_commit: bool = True
    ) -> ChatMessage:
        try:
            image_service = ImageService(session=self.repository.session)
            if not data.session_id:
                chat_session_service = ChatSessionService(
                    session=self.repository.session
                )
                chat_session = await chat_session_service.get_one_or_none(
                    or_(
                        and_(
                            ChatSession.user_1_id == data.user_id,
                            ChatSession.user_2_id == user_id,
                        ),
                        and_(
                            ChatSession.user_1_id == user_id,
                            ChatSession.user_2_id == data.user_id,
                        ),
                    )
                )
                if not chat_session:
                    user_1_id = (
                        min(str(user_id), str(data.user_id))
                        if data.user_id
                        else data.user_id
                    )
                    user_2_id = (
                        max(str(user_id), str(data.user_id))
                        if data.user_id
                        else user_id
                    )
                    chat_session = await chat_session_service.create(
                        {
                            "user_1_id": user_1_id,
                            "user_2_id": user_2_id,
                            "last_message": data.content if data.content else "Image",
                            "last_message_time": datetime.now(),
                        }
                    )
                data.session_id = chat_session.id
            message = await self.create(
                {
                    "content": data.content if data.content != None else "",
                    "session_id": data.session_id,
                    "sender_id": uuid.UUID(user_id),
                },
            )
            if data.image_list:
                await image_service.create_many(
                    [
                        {
                            "url": await self.supabase_service.upload_file(image),
                            "model_type": "chat_message",
                            "model_id": message.id,
                        }
                        for image in data.image_list
                    ], 
                )
            return message
        except Exception as e:
            if auto_commit:
                await self.repository.session.rollback()
            raise e
        finally:
            if auto_commit:
                await self.repository.session.commit()

    async def chat_messages_by_user_id(
        self, user_1_id: uuid.UUID, user_2_id: uuid.UUID, limit_offset: LimitOffset
    ) -> OffsetPagination[ChatMessage]:
        query = select(ChatMessage)
        query = query.join(ChatMessage.session)
        query = query.where(
            or_(
                and_(
                    ChatSession.user_1_id == user_1_id,
                    ChatSession.user_2_id == user_2_id,
                ),
                and_(
                    ChatSession.user_1_id == user_2_id,
                    ChatSession.user_2_id == user_1_id,
                ),
            )
        )
        paginated = (
            query.order_by(desc(ChatSession.created_at))
            .offset(limit_offset.offset)
            .limit(limit_offset.limit)
        )
        result = await self.repository.session.execute(paginated)
        items = result.scalars().all()
        total = await self.count()
        return OffsetPagination(
            items=self.to_schema(data=items, schema_type=ChatMessageSchema).items,
            total=total,
            limit=limit_offset.limit,
            offset=limit_offset.offset,
        )

    async def chat_messages_by_session_id(
        self, session_id: uuid.UUID, limit_offset: LimitOffset
    ) -> OffsetPagination[ChatMessage]:
        query = select(ChatMessage).options(noload(ChatMessage.sender))
        query = query.where(ChatMessage.session_id == session_id)
        paginated = (
            query.order_by(desc(ChatMessage.created_at))
            .offset(limit_offset.offset)
            .limit(limit_offset.limit)
        )
        print(
            paginated.compile(
                dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
            )
        )
        result = await self.repository.session.execute(paginated)
        items = result.scalars().all()
        total = await self.count()
        return OffsetPagination(
            items=self.to_schema(data=items, schema_type=ChatMessageSchema).items,
            total=total,
            limit=limit_offset.limit,
            offset=limit_offset.offset,
        )

    def extract_hashtags(self, text: str) -> List[str]:
        """
        Extracts hashtags (including UUID-like ones) and returns hashtags and cleaned text.
        A hashtag is defined as '#' followed by alphanumerics, dashes, or underscores.

        :param text: The input string
        :return: A tuple of (hashtags list, cleaned text)
        """
        hashtag_pattern = re.compile(r"#([\w\-]+)")
        hashtags = hashtag_pattern.findall(text)
        cleaned_text = hashtag_pattern.sub("", text).strip()
        return hashtags, cleaned_text

    def find_uuids_in_list(self, strings: List[str]) -> List[str]:
        """
        Extracts UUIDs from a list of strings.

        :param strings: List of strings to search
        :return: List of UUID strings found
        """
        print(strings)
        uuid_pattern = re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        )
        uuids = []
        for s in strings:
            found = uuid_pattern.findall(s)
            uuids.extend(found)
        print(uuids)
        return uuids

    async def ask_ai(self, data: AskAIDTO, save_to_session: bool = True) -> str:
        hashtags, cleaned_text = self.extract_hashtags(data.content)
        uuid_list = self.find_uuids_in_list(hashtags)
        property_service = PropertyService(session=self.repository.session)
        if "Analyze" in hashtags:
            if len(uuid_list) == 0:
                raise ValidationException("Missing property id in the message")
            property = await property_service.get_one_or_none(
                Property.id == uuid_list[0]
            )
            if not property:
                raise ValidationException("The property is not found")
            answer = self.ask_gemini_about_property(cleaned_text, property)
            return answer
        if "Compare" in hashtags:
            if len(uuid_list) <= 1:
                raise ValidationException("Not enough property to compare")
            properties = await property_service.list(Property.id.in_(uuid_list))
            if len(properties) < 2:
                raise ValidationException("Not enough valid property to compare")
            return self.compare_properties_with_gemini(properties, data.content)

    def compare_properties_with_gemini(
        self, properties: List[Property], question: str
    ) -> str:
        """
        Sends a comparison question along with details and images of multiple properties to Gemini.

        :param properties: List of Property instances to compare
        :param question: The comparison question to ask (e.g., "Which property is best for a family?")
        :return: Gemini's response text
        """
        image_parts = []
        property_descriptions = []

        for idx, prop in enumerate(properties):
            property_descriptions.append(f"Property {idx + 1}:\n{prop.to_string()}\n")
            for image in prop.images:
                try:
                    image_bytes = requests.get(image.url).content
                    image_parts.append(
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg",
                        )
                    )
                except Exception as e:
                    print(f"Error downloading image: {image.url} -> {e}")
        text_content = "\n\n".join(property_descriptions)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                f"You are a helpful assistant tasked with comparing the following real estate listings:\n\n{text_content}",
                f"Question: {question}",
                *image_parts,
            ],
            config=GenerateContentConfig(
                tools=[Tool(google_search=GoogleSearch())], response_modalities=["TEXT"]
            ),
        )

        return response.text

    def ask_gemini_about_property(self, question: str, property: Property) -> str:
        """
        Sends a question with property description and image to Gemini API and returns the answer.

        :param description: The textual description of the property
        :param question: The question to ask about the property
        :param property: Thew property the user is question about
        :return: Gemini's response as a string, or None if request fails
        """
        image_list = []
        for image_path in [image.url for image in property.images]:
            image_bytes = requests.get(image_path).content
            image_list.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
            )
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                f"Property detail: {property.to_string()}",
                *image_list,
                f"You are a professional real estate advisor. Based on the provided property details, summarize the listing clearly, comment on its strengths and weaknesses, and offer helpful suggestions to the user. The question the user asked you is: {question}",
            ],
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            ),
        )
        return response.text

    async def summarize_session(self, session_id: uuid.UUID) -> str:
        """
        Summarize the entire chat session using a lightweight summarization model.
        """
        # Fetch all messages ordered oldest first
        query = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        result = await self.repository.session.execute(query)
        messages: List[ChatMessage] = result.scalars().all()
        # Concatenate speaker labels
        transcript = "\n".join(
            f"{ 'User' if msg.sender_id else 'Assistant' }: {msg.content}"
            for msg in messages
        )
        # Load summarizer (T5-small) on CPU
        summarizer = pipeline("summarization", model="t5-small", device=-1)
        summary_out = summarizer(
            transcript, max_length=150, min_length=50, do_sample=False
        )
        return summary_out[0]["summary_text"]

    async def build_chat_context(
        self, session_id: uuid.UUID, window_size: int = 10
    ) -> List[Dict[str, str]]:
        """
        Build a message context: start with system summary, then last `window_size` messages.
        Returns a list of dicts with 'author' and 'content'.
        """
        context = []
        subq = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(desc(ChatMessage.created_at))
            .limit(window_size)
        )
        result = await self.repository.session.execute(subq)
        recent_msgs = result.scalars().all()[::-1]
        for msg in recent_msgs:
            context.append(
                UserContent(msg.content) if msg.sender_id else ModelContent(msg.content)
            )
        return context

    async def ai_respond_to_user(
        self, data: CreateMessageDTO, user_id: uuid.UUID, window_size: int = 10
    ) -> ChatMessage:
        """
        Unified response handler:
        - Summarize past.
        - Build context.
        - Always include a system instruction to detect suggestion intent across languages.
        - Call Gemini, persist messages.

        The LLM is instructed: if the user requests property suggestions, at the end of its reply include a single hashtag line:
        #PROPERTY_CRITERIA:<json>
        where <json> exactly matches the PropertySchema fields:
        {
            "title": string,
            "property_category": string,
            "transaction_type": string,
            "price": number,
            "bedrooms": integer,
            "bathrooms": number,
            "sqm": number,
            "description": string,
            "average_rating": number,
            "status": boolean,
        }
        Otherwise omit the tag.
        """
        try:
            if data.session_id:
                summary = await self.summarize_session(data.session_id)
                context = await self.build_chat_context(data.session_id, window_size)
            else:
                context = []
            context.append(UserContent(data.content))
            system_instruction = """You are a real estate assistant that help user choose and find the best match properties. Detect if the user wants property suggestions in any language.
            Always respond helpfully. If suggestions are requested, at the very end append exactly one line with 
            #PROPERTY_CRITERIA:<json>
            where <json> exactly matches the PropertySchema fields:
            {
                "title": string,
                "property_category": string,
                "transaction_type": string,
                "price": number,
                "bedrooms": integer,
                "bathrooms": number,
                "sqm": number,
                "description": string,
                "average_rating": number,
                "status": boolean,
            }
            If not, do not append the tag."""
            system_instruction += f"Also, here is there summary of the conversation between you and this customer {summary}"
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=context,
                config=GenerateContentConfig(
                    tools=[Tool(google_search=GoogleSearch())],
                    system_instruction=system_instruction,
                ),
            )
            assistant_text = response.text
            message = await self.create_message(
                CreateMessageDTO(session_id=data.session_id, content=data.content),
                user_id,
                auto_commit=False,
            )
            message = await self.create(
                {
                    "content": assistant_text,
                    "session_id": message.session_id,
                    "sender_id": None,
                }
            )

            return message
        except Exception as e:
            await self.repository.session.rollback()
            raise InternalServerException("There is something wrong, try again later")
        finally:
            await self.repository.session.commit()


async def provide_chat_message_service(
    db_session: AsyncSession,
) -> AsyncGenerator[ChatMessageService]:

    async with ChatMessageService.new(session=db_session) as service:
        yield service
