# notification_service.py

from firebase_admin import messaging
from typing import List, Optional
import logging
import os

class NotificationService:
    def _build_message(
        self,
        token: str,
        title: str,
        body: str,
        image_url: Optional[str] = None,
        data: Optional[dict] = None
    ) -> messaging.Message:
        return messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
                image=image_url,
            ),
            token=token,
            data=data or {},
        )

    def send_to_token(
        self,
        token: str,
        title: str,
        body: str,
        image_url: Optional[str] = None,
        data: Optional[dict] = None
    ):
        message = self._build_message(token, title, body, image_url, data)
        response = messaging.send(message)
        logging.info(f"Sent notification to token: {token} | Response: {response}")
        return response

    def send_to_multiple_tokens(
        self,
        tokens: List[str],
        title: str,
        body: str,
        image_url: Optional[str] = None,
        data: Optional[dict] = None
    ):
        messages = [
            self._build_message(token, title, body, image_url, data)
            for token in tokens
        ]
        responses = []
        for i in range(0, len(messages), 500):
            batch = messages[i:i+500]
            batch_response = messaging.send_all(batch)
            responses.append(batch_response)
            logging.info(f"Batch {i // 500 + 1} sent with {len(batch_response.responses)} messages")
        return responses
