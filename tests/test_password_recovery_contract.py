import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException

from app.api.v1 import auth
from app.models.password_reset_code import PasswordResetCode
from app.models.user import User
from app.services.password_reset_email_service import PasswordResetEmailDeliveryError
from app.schemas.user import (
    PasswordResetRequest,
    ResetPasswordRequest,
    VerifyResetOtpRequest,
)


class PasswordRecoveryContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 24, 12, 0, 0)
        self.db = Mock()
        self.db.commit = AsyncMock()
        self.db.rollback = AsyncMock()

    async def test_unknown_account_receives_generic_accepted_response(self):
        with patch.object(
            auth.user_repo,
            "get_by_email",
            new=AsyncMock(return_value=None),
        ), patch.object(
            auth.password_reset_email_service,
            "send_code",
            new=AsyncMock(),
        ) as send_code:
            response = await auth.forgot_password(
                PasswordResetRequest(email="missing@example.com"),
                self.db,
            )

        self.assertEqual(response.status_code, 202)
        send_code.assert_not_awaited()

    async def test_password_account_receives_a_code_that_is_not_persisted_plaintext(self):
        user = User(
            id=7,
            email="person@example.com",
            full_name="Test Person",
            hashed_password="hashed",
            is_active=True,
            password_login_enabled=True,
        )
        created_code = None

        async def capture_code(_db, code):
            nonlocal created_code
            created_code = code

        with (
            patch.object(auth, "_utcnow", return_value=self.now),
            patch.object(auth, "_new_reset_code", return_value="123456"),
            patch.object(
                auth.user_repo,
                "get_by_email",
                new=AsyncMock(return_value=user),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "count_requests_since",
                new=AsyncMock(return_value=0),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "get_latest_for_user",
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "invalidate_active_for_user",
                new=AsyncMock(),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "create",
                new=capture_code,
            ),
            patch.object(
                auth.password_reset_email_service,
                "send_code",
                new=AsyncMock(),
            ) as send_code,
        ):
            response = await auth.forgot_password(
                PasswordResetRequest(email="person@example.com"),
                self.db,
            )

        self.assertEqual(response.status_code, 202)
        self.assertIsNotNone(created_code)
        self.assertNotEqual(created_code.code_hash, "123456")
        self.assertEqual(created_code.code_hash, auth._reset_code_hash(7, "123456"))
        self.assertEqual(created_code.expires_at, self.now + timedelta(minutes=10))
        send_code.assert_awaited_once_with("person@example.com", "123456")
        self.db.commit.assert_awaited_once()

    async def test_google_only_account_is_indistinguishable_from_unknown_account(self):
        user = User(
            id=7,
            email="person@example.com",
            full_name="Test Person",
            hashed_password="hashed",
            is_active=True,
            password_login_enabled=False,
            google_subject="google-subject",
        )
        with patch.object(
            auth.user_repo,
            "get_by_email",
            new=AsyncMock(return_value=user),
        ), patch.object(
            auth.password_reset_email_service,
            "send_code",
            new=AsyncMock(),
        ) as send_code:
            response = await auth.forgot_password(
                PasswordResetRequest(email="person@example.com"),
                self.db,
            )

        self.assertEqual(response.status_code, 202)
        send_code.assert_not_awaited()

    async def test_delivery_failure_keeps_the_response_generic(self):
        user = User(
            id=7,
            email="person@example.com",
            full_name="Test Person",
            hashed_password="hashed",
            is_active=True,
            password_login_enabled=True,
        )
        with (
            patch.object(auth, "_utcnow", return_value=self.now),
            patch.object(
                auth.user_repo,
                "get_by_email",
                new=AsyncMock(return_value=user),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "count_requests_since",
                new=AsyncMock(return_value=0),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "get_latest_for_user",
                new=AsyncMock(return_value=None),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "invalidate_active_for_user",
                new=AsyncMock(),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "create",
                new=AsyncMock(),
            ),
            patch.object(
                auth.password_reset_email_service,
                "send_code",
                new=AsyncMock(side_effect=PasswordResetEmailDeliveryError("failed")),
            ),
        ):
            response = await auth.forgot_password(
                PasswordResetRequest(email="person@example.com"),
                self.db,
            )

        self.assertEqual(response.status_code, 202)
        self.db.rollback.assert_awaited_once()

    async def test_invalid_code_consumes_an_attempt(self):
        user = User(
            id=7,
            email="person@example.com",
            full_name="Test Person",
            hashed_password="hashed",
            is_active=True,
            password_login_enabled=True,
        )
        record = PasswordResetCode(
            user_id=7,
            code_hash=auth._reset_code_hash(7, "123456"),
            expires_at=self.now + timedelta(minutes=10),
            requested_at=self.now,
            attempt_count=0,
        )
        with (
            patch.object(auth, "_utcnow", return_value=self.now),
            patch.object(
                auth.user_repo,
                "get_by_email",
                new=AsyncMock(return_value=user),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "get_latest_for_user",
                new=AsyncMock(return_value=record),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "increment_attempts",
                new=AsyncMock(),
            ) as increment_attempts,
        ):
            with self.assertRaises(HTTPException) as raised:
                await auth.verify_reset_otp(
                    VerifyResetOtpRequest(
                        email="person@example.com",
                        code="654321",
                    ),
                    self.db,
                )

        self.assertEqual(raised.exception.status_code, 400)
        increment_attempts.assert_awaited_once_with(self.db, record)
        self.db.commit.assert_awaited_once()

    async def test_verifying_a_valid_code_marks_it_verified(self):
        record = PasswordResetCode(
            user_id=7,
            code_hash=auth._reset_code_hash(7, "123456"),
            expires_at=self.now + timedelta(minutes=10),
            requested_at=self.now,
            attempt_count=0,
        )
        user = User(
            id=7,
            email="person@example.com",
            full_name="Test Person",
            hashed_password="hashed",
            is_active=True,
            password_login_enabled=True,
        )

        with (
            patch.object(auth, "_utcnow", return_value=self.now),
            patch.object(
                auth.user_repo,
                "get_by_email",
                new=AsyncMock(return_value=user),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "get_latest_for_user",
                new=AsyncMock(return_value=record),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "mark_verified",
                new=AsyncMock(),
            ) as mark_verified,
        ):
            response = await auth.verify_reset_otp(
                VerifyResetOtpRequest(email="person@example.com", code="123456"),
                self.db,
            )

        self.assertEqual(response.status_code, 204)
        mark_verified.assert_awaited_once_with(self.db, record, self.now)

    async def test_reset_consumes_verified_code_and_revokes_existing_sessions(self):
        user = User(
            id=7,
            email="person@example.com",
            full_name="Test Person",
            hashed_password="old-hash",
            is_active=True,
            password_login_enabled=True,
        )
        record = PasswordResetCode(
            user_id=7,
            code_hash=auth._reset_code_hash(7, "123456"),
            expires_at=self.now + timedelta(minutes=10),
            requested_at=self.now,
            verified_at=self.now,
            attempt_count=0,
        )

        with (
            patch.object(auth, "_utcnow", return_value=self.now),
            patch.object(
                auth.user_repo,
                "get_by_email",
                new=AsyncMock(return_value=user),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "get_latest_for_user",
                new=AsyncMock(return_value=record),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "consume",
                new=AsyncMock(),
            ) as consume,
            patch("app.api.v1.auth.has_password", return_value="new-hash"),
            patch.object(
                auth.refresh_token_repo,
                "revoke_all_for_user",
                new=AsyncMock(),
            ) as revoke_all,
        ):
            response = await auth.reset_password(
                ResetPasswordRequest(
                    email="person@example.com",
                    code="123456",
                    new_password="new-password",
                ),
                self.db,
            )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(user.hashed_password, "new-hash")
        consume.assert_awaited_once_with(self.db, record, self.now)
        revoke_all.assert_awaited_once_with(self.db, 7)

    async def test_expired_code_is_rejected(self):
        user = User(
            id=7,
            email="person@example.com",
            full_name="Test Person",
            hashed_password="hashed",
            is_active=True,
            password_login_enabled=True,
        )
        record = PasswordResetCode(
            user_id=7,
            code_hash=auth._reset_code_hash(7, "123456"),
            expires_at=self.now - timedelta(seconds=1),
            requested_at=self.now - timedelta(minutes=10),
            attempt_count=0,
        )

        with (
            patch.object(auth, "_utcnow", return_value=self.now),
            patch.object(
                auth.user_repo,
                "get_by_email",
                new=AsyncMock(return_value=user),
            ),
            patch.object(
                auth.password_reset_code_repo,
                "get_latest_for_user",
                new=AsyncMock(return_value=record),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await auth.verify_reset_otp(
                    VerifyResetOtpRequest(
                        email="person@example.com",
                        code="123456",
                    ),
                    self.db,
                )

        self.assertEqual(raised.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
