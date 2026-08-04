from dataclasses import dataclass

from google.auth.transport.requests import Request
from google.oauth2.id_token import verify_oauth2_token


class GoogleIdentityError(ValueError):
    pass


@dataclass(frozen=True)
class GoogleIdentity:
    subject: str
    email: str
    full_name: str


class GoogleAuthService:
    def verify_id_token(self, id_token: str, audience: str) -> GoogleIdentity:
        try:
            claims = verify_oauth2_token(id_token, Request(), audience=audience)
        except Exception as exc:
            raise GoogleIdentityError("Google identity verification failed") from exc

        issuer = claims.get("iss")
        subject = claims.get("sub")
        email = claims.get("email")
        if (
            issuer not in {"accounts.google.com", "https://accounts.google.com"}
            or not isinstance(subject, str)
            or not subject
            or not isinstance(email, str)
            or not email
            or claims.get("email_verified") is not True
        ):
            raise GoogleIdentityError("Google identity verification failed")

        full_name = claims.get("name")
        return GoogleIdentity(
            subject=subject,
            email=email.lower(),
            full_name=full_name if isinstance(full_name, str) and full_name else email,
        )
