from datetime import date, datetime
from email.mime.text import MIMEText
import os
import smtplib
from typing import Optional
import logging
from litestar.response import Template
from configs.template import template_config
from database.models.user import User

logger = logging.getLogger(__name__)


class MailService:
    def __init__(
        self,
        use_tls: bool = True,
        sender_email: str = None,
    ):
        self.smtp_server = os.environ.get("SMTP_SERVER")
        self.smtp_port = os.environ.get("SMTP_PORT")
        self.smtp_username = os.environ.get("SMTP_USERNAME")
        self.smtp_password = os.environ.get("SMTP_PASSWORD")
        self.use_tls = use_tls
        self.sender_email = self.smtp_username

    async def send_email(
        self,
        recipient: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
    ) -> bool:
        """Send an email asynchronously"""
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                # Create a MIMEText object with UTF-8 encoding
                msg = MIMEText(html_content, "plain", "utf-8")
                msg["Subject"] = subject
                msg["From"] = "Real estate - AI"
                msg["To"] = recipient
                # Connect to SMTP server
                if self.use_tls:
                    server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail("Real estate - AI", recipient, msg.as_string())
                server.quit()

            logger.info(f"Email sent successfully to {recipient}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {recipient}: {str(e)}")
            return False

    async def send_forget_password_email(self, user: User, token: str) -> bool:
        content = template_config.engine_instance.get_template(
            template_name="forget_password.html"
        ).render(
            {
                "user_name": user.name,
                "reset_link": f"{os.environ.get('BASE_URL')}/auth/reset_password?token={token}",
                "current_year": date.today().year,
            }
        )
        return await self.send_email(
            recipient=user.email,
            subject="Real Estate: Password Reset",
            html_content=content,
        )



async def provide_mail_service() -> MailService:
    return MailService()
