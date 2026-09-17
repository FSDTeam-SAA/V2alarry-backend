import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from app.api.dependencies.auth import require_admin
from app.core.agreements import CURRENT_AGREEMENT_VERSION
from app.schemas.user import AgreementAcceptanceCreate, UserCreate, UserResponse
from app.services.auth_service import AuthService
from app.api.v1 import users
from app.schemas.user import PasswordChangeRequest


class IdentityConsentContractTests(unittest.IsolatedAsyncioTestCase):
    def test_credential_registration_requires_the_current_agreement(self):
        valid = UserCreate(
            email="leader@example.com",
            password="a sufficiently long password",
            full_name="Test Leader",
            agreement_version=CURRENT_AGREEMENT_VERSION,
        )
        self.assertEqual(valid.agreement_version, CURRENT_AGREEMENT_VERSION)

        with self.assertRaises(ValidationError):
            UserCreate(
                email="leader@example.com",
                password="a sufficiently long password",
                full_name="Test Leader",
                agreement_version="outdated-version",
            )

    def test_agreement_endpoint_only_accepts_the_current_version(self):
        acceptance = AgreementAcceptanceCreate(
            agreement_version=CURRENT_AGREEMENT_VERSION,
            source="consent-gate",
        )
        self.assertEqual(acceptance.agreement_version, CURRENT_AGREEMENT_VERSION)

        with self.assertRaises(ValidationError):
            AgreementAcceptanceCreate(
                agreement_version="future-or-unknown",
                source="consent-gate",
            )

    async def test_admin_dependency_enforces_exact_admin_role(self):
        admin = Mock(role="admin")
        user = Mock(role="user")

        self.assertIs(await require_admin(admin), admin)
        with self.assertRaises(HTTPException) as error:
            await require_admin(user)
        self.assertEqual(error.exception.status_code, 403)

    def test_google_accounts_do_not_gain_a_local_password_capability(self):
        user = AuthService().register_google_user(
            email="leader@example.com",
            full_name="Test Leader",
            google_subject="google-subject",
        )
        self.assertFalse(user.password_login_enabled)

    def test_profile_contract_includes_identity_and_consent_capabilities(self):
        fields = UserResponse.model_fields
        for field in (
            "full_name",
            "role",
            "is_active",
            "auth_provider",
            "password_login_enabled",
            "accepted_agreement_version",
        ):
            self.assertIn(field, fields)

    async def test_google_only_account_cannot_change_a_local_password(self):
        account = Mock(id=7, password_login_enabled=False)
        with self.assertRaises(HTTPException) as error:
            await users.change_password(
                PasswordChangeRequest(
                    current_password="current-password",
                    new_password="new-password",
                ),
                current_user=account,
                db=AsyncMock(),
            )
        self.assertEqual(error.exception.status_code, 409)

    async def test_password_change_revokes_all_refresh_tokens(self):
        account = Mock(
            id=7,
            password_login_enabled=True,
            hashed_password="old-hash",
        )
        db = AsyncMock()
        with (
            patch("app.api.v1.users.verify_password", side_effect=[True, False]),
            patch("app.api.v1.users.has_password", return_value="new-hash"),
            patch.object(
                users.refresh_token_repo,
                "revoke_all_for_user",
                new=AsyncMock(),
            ) as revoke_all,
        ):
            response = await users.change_password(
                PasswordChangeRequest(
                    current_password="current-password",
                    new_password="new-password",
                ),
                current_user=account,
                db=db,
            )

        self.assertEqual(response.status_code, 204)
        self.assertEqual(account.hashed_password, "new-hash")
        revoke_all.assert_awaited_once_with(db, 7)


if __name__ == "__main__":
    unittest.main()
