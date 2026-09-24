import asyncio
import smtplib
from email.message import EmailMessage

from app.core.config import settings


class PasswordResetEmailDeliveryError(Exception):
    pass


class PasswordResetEmailService:
    async def send_code(self, email: str, code: str) -> None:
        if (
            not settings.SMTP_USERNAME
            or not settings.SMTP_PASSWORD
            or not settings.SMTP_FROM_EMAIL
        ):
            raise PasswordResetEmailDeliveryError(
                "Password recovery email delivery is not configured"
            )

        try:
            await asyncio.to_thread(self._send_message, email, code)
        except Exception as error:
            raise PasswordResetEmailDeliveryError(
                "Unable to send password recovery email"
            ) from error

    @staticmethod
    def _send_message(email: str, code: str) -> None:
        message = EmailMessage()
        message["From"] = settings.SMTP_FROM_EMAIL
        message["To"] = email
        message["Subject"] = "Your LeaderCoach password reset code"
        message.set_content(
            "Use this code to reset your LeaderCoach password:\n\n"
            f"{code}\n\n"
            f"This code expires in {settings.PASSWORD_RESET_CODE_TTL_MINUTES} minutes."
        )

        with smtplib.SMTP_SSL(
            settings.SMTP_HOST,
            settings.SMTP_PORT,
            timeout=settings.SMTP_TIMEOUT_SECONDS,
        ) as smtp:
            smtp.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            smtp.send_message(message)
