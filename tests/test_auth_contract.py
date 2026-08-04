import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from app.api.v1 import auth
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.schemas.user import GoogleLoginRequest, RefreshTokenRequest, UserLogin
from app.services.google_auth_service import GoogleAuthService, GoogleIdentityError


class AuthenticationContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_login_returns_the_nextauth_token_contract(self):
        user = User(
            id=7,
            email="person@example.com",
            full_name="Test Person",
            role="candidate",
            is_active=True,
            hashed_password="hashed",
        )

        with (
            patch.object(auth.user_repo, "get_by_email", new=AsyncMock(return_value=user)),
            patch.object(auth.auth_service, "authenticate_user", return_value=user),
            patch.object(
                auth,
                "_issue_tokens",
                new=AsyncMock(
                    return_value={
                        "access_token": "access",
                        "refresh_token": "refresh",
                        "access_token_expires_in": 3600,
                        "user": {
                            "id": 7,
                            "email": "person@example.com",
                            "full_name": "Test Person",
                            "role": "candidate",
                        },
                    }
                ),
            ),
        ):
            response = await auth.login(
                UserLogin(email="person@example.com", password="password"),
                Mock(),
            )

        self.assertEqual(response["access_token"], "access")
        self.assertEqual(response["refresh_token"], "refresh")
        self.assertEqual(response["access_token_expires_in"], 3600)
        self.assertEqual(response["user"]["id"], 7)

    async def test_issued_refresh_token_is_stored_only_as_a_hash(self):
        user = User(
            id=7,
            email="person@example.com",
            full_name="Test Person",
            role="candidate",
            is_active=True,
            hashed_password="hashed",
        )
        persisted_token = None

        async def capture_token(_db, token):
            nonlocal persisted_token
            persisted_token = token
            return token

        with patch.object(auth.refresh_token_repo, "create", new=capture_token):
            response = await auth._issue_tokens(Mock(), user)

        self.assertTrue(response.access_token)
        self.assertTrue(response.refresh_token)
        self.assertNotEqual(persisted_token.token_hash, response.refresh_token)
        self.assertEqual(persisted_token.token_hash, auth._token_hash(response.refresh_token))

    async def test_refresh_rotates_an_active_token(self):
        user = User(
            id=7,
            email="person@example.com",
            full_name="Test Person",
            role="candidate",
            is_active=True,
            hashed_password="hashed",
        )
        stored_token = RefreshToken(
            user_id=7,
            token_hash=auth._token_hash("refresh"),
            expires_at=datetime.utcnow() + timedelta(days=1),
        )

        with (
            patch.object(
                auth.refresh_token_repo,
                "get_active_by_hash",
                new=AsyncMock(return_value=stored_token),
            ),
            patch.object(auth.user_repo, "get_by_id", new=AsyncMock(return_value=user)),
            patch.object(auth.refresh_token_repo, "revoke", new=AsyncMock()) as revoke,
            patch.object(
                auth,
                "_issue_tokens",
                new=AsyncMock(
                    return_value={
                        "access_token": "new-access",
                        "refresh_token": "new-refresh",
                        "access_token_expires_in": 3600,
                        "user": {
                            "id": 7,
                            "email": "person@example.com",
                            "full_name": "Test Person",
                            "role": "candidate",
                        },
                    }
                ),
            ),
        ):
            response = await auth.refresh_tokens(
                RefreshTokenRequest(refresh_token="refresh"),
                Mock(),
            )

        revoke.assert_awaited_once()
        self.assertEqual(response["refresh_token"], "new-refresh")

    async def test_google_login_uses_the_same_token_contract(self):
        user = User(
            id=7,
            email="person@example.com",
            full_name="Test Person",
            role="candidate",
            is_active=True,
            hashed_password="hashed",
            google_subject="google-subject",
        )
        identity = Mock(subject="google-subject", email="person@example.com")

        with (
            patch.object(auth.settings, "GOOGLE_CLIENT_ID", "google-client"),
            patch.object(
                auth.google_auth_service,
                "verify_id_token",
                return_value=identity,
            ),
            patch.object(
                auth.user_repo,
                "get_by_google_subject",
                new=AsyncMock(return_value=user),
            ),
            patch.object(
                auth,
                "_issue_tokens",
                new=AsyncMock(
                    return_value={
                        "access_token": "access",
                        "refresh_token": "refresh",
                        "access_token_expires_in": 3600,
                        "user": {
                            "id": 7,
                            "email": "person@example.com",
                            "full_name": "Test Person",
                            "role": "candidate",
                        },
                    }
                ),
            ),
        ):
            response = await auth.google_login(
                GoogleLoginRequest(id_token="google-id-token"),
                Mock(),
            )

        self.assertEqual(response["user"]["email"], "person@example.com")

    def test_google_identity_rejects_an_unverified_email(self):
        claims = {
            "iss": "https://accounts.google.com",
            "sub": "google-subject",
            "email": "person@example.com",
            "email_verified": False,
        }
        with patch(
            "app.services.google_auth_service.verify_oauth2_token",
            return_value=claims,
        ):
            with self.assertRaises(GoogleIdentityError):
                GoogleAuthService().verify_id_token("google-id-token", "google-client")


if __name__ == "__main__":
    unittest.main()
