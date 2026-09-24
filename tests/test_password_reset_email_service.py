import unittest
from unittest.mock import Mock, patch

from app.services.password_reset_email_service import (
    PasswordResetEmailService,
)


class PasswordResetEmailServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_the_reset_code_through_gmail_smtp(self):
        smtp_client = Mock()
        smtp_client.__enter__ = Mock(return_value=smtp_client)
        smtp_client.__exit__ = Mock(return_value=None)

        with (
            patch(
                "app.services.password_reset_email_service.smtplib.SMTP_SSL",
                return_value=smtp_client,
            ) as smtp_ssl,
            patch(
                "app.services.password_reset_email_service.settings.SMTP_USERNAME",
                "security@example.com",
            ),
            patch(
                "app.services.password_reset_email_service.settings.SMTP_PASSWORD",
                "gmail-app-password",
            ),
            patch(
                "app.services.password_reset_email_service.settings.SMTP_FROM_EMAIL",
                "LeaderCoach <security@example.com>",
            ),
        ):
            await PasswordResetEmailService().send_code("member@example.com", "123456")

        smtp_ssl.assert_called_once_with("smtp.gmail.com", 465, timeout=10)
        smtp_client.login.assert_called_once_with(
            "security@example.com", "gmail-app-password"
        )
        sent_message = smtp_client.send_message.call_args.args[0]
        self.assertEqual(sent_message["To"], "member@example.com")
        self.assertEqual(sent_message["Subject"], "Your LeaderCoach password reset code")
        self.assertIn("123456", sent_message.get_content())
