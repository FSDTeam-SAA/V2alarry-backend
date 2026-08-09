# Interview-Platform-Fastapi
On Ubuntu, you can start PostgreSQL with:

sudo systemctl start postgresql

Check if it's running:

sudo systemctl status postgresql

<!-- run command -->
uv run uvicorn app.main:app --reload


### Api docs:
http://127.0.0.1:8000/docs

### Embedding model requirement

Chat runs with `all-MiniLM-L6-v2` in offline-only mode. Preload the model in
the deployed image or runtime cache before starting the backend; the server
will fail startup if it is unavailable rather than downloading it during a
chat request.


# Docker & Qdrant Quick Guide

## 1. Start Docker

Check if Docker is installed:

```bash
docker --version
```

Check if Docker is running:

```bash
docker info
```

If Docker is not running, start the Docker Desktop application or start the Docker service on Linux:

```bash
sudo systemctl start docker
```

Check running containers:

```bash
docker ps
```

Check all containers (running and stopped):

```bash
docker ps -a
```

---

# 2. Run Qdrant

Run Qdrant for the first time:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

This command:

* Downloads the Qdrant image if needed.
* Starts the container in the background.
* Stores data in `qdrant_storage`.
* Exposes the API on port `6333`.

---

# 3. Stop Qdrant

```bash
docker stop qdrant
```

---

# 4. Start an Existing Qdrant Container

If you have already created the container:

```bash
docker start qdrant
```

---

# 5. Restart Qdrant

```bash
docker restart qdrant
```

---

# 6. Remove Qdrant Container

> Only remove it if you no longer need the container.

```bash
docker rm qdrant
```

If it is running:

```bash
docker stop qdrant
docker rm qdrant
```

---

# 7. View Logs

```bash
docker logs qdrant
```

Follow logs live:

```bash
docker logs -f qdrant
```

---

# 8. Open Qdrant Dashboard

Open your browser:

```
http://localhost:6333/dashboard
```

Or check if the server is running:

```
http://localhost:6333
```

---

# 9. Check Docker Images

```bash
docker images
```

---

# 10. Remove the Qdrant Image

```bash
docker rmi qdrant/qdrant
```

---

# 11. Useful Docker Commands

List running containers:

```bash
docker ps
```

List all containers:

```bash
docker ps -a
```

List downloaded images:

```bash
docker images
```

Stop all running containers:

```bash
docker stop $(docker ps -q)
```

Remove all stopped containers:

```bash
docker container prune
```

---

# 12. Verify Qdrant is Running

```bash
curl http://localhost:6333
```

Expected output:

```json
{
  "title": "qdrant - vector search engine"
}
```

---

# Typical Development Workflow

Start Docker (if needed):

```bash
sudo systemctl start docker
```

Start Qdrant:

```bash
docker start qdrant
```

Run your backend:

```bash
python main.py
```

When finished:

```bash
docker stop qdrant
```

(Optional) Stop Docker:

```bash
sudo systemctl stop docker
```
# Alembic Migration Commands

## 1. Create a new migration

```bash
alembic revision --autogenerate -m "Your migration message"
```

Example:

```bash
alembic revision --autogenerate -m "Add rag tables"
```

---

# Backend change log

This section records the current backend changes that support the dashboard,
authenticated chat, and Google sign-in integrations. It is intended as a
rollout reference: apply the migrations before deploying the corresponding
frontend changes.

## Migration order

Run migrations from the backend virtual environment:

```bash
alembic upgrade head
```

The current migration chain adds:

| Revision | Change | Why it is needed |
| --- | --- | --- |
| `b9e4a22c91d3` | `documents.category`, `documents.file_size_bytes`, and the `refresh_tokens` table | Supplies document metadata to the dashboard and allows refresh tokens to be revoked and rotated server-side. |
| `c3f091d4ea62` | `users.google_subject` with a unique index | Stores Google’s stable account identifier so Google sign-in does not rely on an email address as an account key. |

Do not deploy the Google sign-in endpoint before `c3f091d4ea62` has been
applied. Do not deploy refresh-token issuing code before `b9e4a22c91d3` has
been applied.

## Authentication and session security

### Access and refresh tokens

Changed files:

- `app/utils/jwt.py`
- `app/api/dependencies/auth.py`
- `app/api/v1/auth.py`
- `app/models/refresh_token.py`
- `app/repositories/refresh_token_repository.py`
- `app/models/__init__.py`
- `app/core/config.py`

What changed:

- Access JWTs now contain `token_type: "access"`.
- Refresh JWTs contain `token_type: "refresh"`, a unique `jti`, and their own
  expiry based on `REFRESH_TOKEN_EXPIRE_MINUTES` (default: 10080 minutes).
- Protected routes reject tokens that are not access tokens. A refresh token
  therefore cannot be used as a bearer token for a normal API request.
- `POST /api/v1/auth/login` now returns a `TokenResponse`, which contains both
  an access token and refresh token, their expiry durations, and the user.
- `POST /api/v1/auth/refresh` validates the refresh JWT, hashes the presented
  token, checks its server-side record, revokes that record, and issues a new
  token pair. This is refresh-token rotation.
- Only SHA-256 hashes of refresh tokens are persisted. A database read does not
  expose a usable refresh token.

Why:

The frontend uses short-lived access tokens for API calls and needs a secure
way to continue a valid signed-in session. Separating token types and rotating
refresh tokens limits misuse of a leaked token and gives the backend a
server-side revocation point.

### Credential and account management

Changed files:

- `app/services/auth_service.py`
- `app/repositories/user_repository.py`
- `app/schemas/user.py`
- `app/api/v1/users.py`

What changed:

- Inactive users can no longer authenticate with email and password.
- `GET /api/v1/users/me` returns the complete `UserResponse`, including
  `full_name`.
- `PATCH /api/v1/users/me` updates the signed-in user’s name and/or email.
  A conflicting email returns `409 Email is already in use`.
- `POST /api/v1/users/me/password` requires the current password, validates a
  new password between 8 and 72 characters, and returns `204 No Content` on
  success.
- User repository helpers now support updating a user and looking up a user by
  Google subject.

Why:

These endpoints provide the dashboard settings UI with real persistence while
protecting account ownership and preventing duplicate email addresses.

## Google OAuth sign-in

Changed files:

- `app/services/google_auth_service.py`
- `app/api/v1/auth.py`
- `app/models/user.py`
- `app/schemas/user.py`
- `app/services/auth_service.py`
- `app/repositories/user_repository.py`
- `app/core/config.py`
- `requirements.txt`
- `alembic/versions/c3f091d4ea62_add_google_subject_to_users.py`

### API contract

`POST /api/v1/auth/google`

Request body:

```json
{
  "id_token": "Google OpenID Connect ID token"
}
```

Success response: the same `TokenResponse` returned by `/auth/login` and
`/auth/refresh`.

Failure behavior:

- `422` for an omitted or blank `id_token`.
- `503 Google sign-in is not configured` when `GOOGLE_CLIENT_ID` is absent.
- `401 Unable to sign in with Google` when token verification or identity
  matching fails.

### Verification and linking flow

1. The backend verifies the ID token signature, expiry, and audience with
   `google-auth`.
2. It accepts only Google issuers (`accounts.google.com` or
   `https://accounts.google.com`) and requires `email_verified: true`.
3. It finds returning users by `google_subject` (`sub` claim), which is the
   stable Google identifier.
4. If an existing local account has the same verified email and no Google
   subject yet, it is linked to that subject.
5. Otherwise, the backend creates a user with a generated, hashed unusable
   password and records the Google subject.
6. The backend issues its normal application access and refresh tokens. Google
   tokens are not used as bearer tokens for V2alarry APIs.

Why:

The Google profile returned to a browser is not sufficient proof for the API.
Verification occurs on the backend, the audience must match this application’s
Google client ID, and the Google `sub` value prevents an email-address change
from creating or taking over an account.

### Required configuration

Set this environment variable in the backend deployment:

```dotenv
GOOGLE_CLIENT_ID=your-google-oauth-web-client-id
```

`GOOGLE_CLIENT_ID` must be the same OAuth web-client ID configured by the
NextAuth application. The Google client secret belongs in the Next.js server
environment, not in this backend endpoint.

## Document metadata and dashboard API support

Changed files:

- `app/models/document.py`
- `app/schemas/document.py`
- `app/services/document_service.py`
- `app/api/v1/admin/documents.py`
- `alembic/versions/b9e4a22c91d3_add_dashboard_auth_and_document_metadata.py`

What changed:

- Document uploads accept an optional multipart `category` field (maximum 100
  characters).
- The backend stores `category` and `file_size_bytes` alongside each document.
- Document list/status responses include both fields.

Why:

The dashboard needs category and file-size information without inferring it
from filenames or client-side upload state. The database is the source of
truth for these values.

## Authenticated chat contract

Changed files:

- `app/api/v1/user/chat.py`
- `app/workflows/chat_workflow.py`
- `tests/test_chat_contract.py`

What changed:

- `POST /api/v1/chat/` trims messages and rejects blank or whitespace-only
  input before generation starts.
- Unexpected chat errors now return generic client-safe messages instead of
  exposing internal exception text.
- Streaming responses send zero or more `token` events, persist the turn, then
  send exactly one `done` event containing the persisted `conversation_id` and
  `message_id`, followed by the SSE sentinel `data: [DONE]`.
- Conversation list, transcript, and delete routes remain in place.
- Deleting a missing conversation correctly preserves the intended `404
  Conversation not found` response.

Why:

The frontend must only reconcile an optimistic message after the database has
saved it. The persisted identifiers in `done` make that reconciliation safe,
and generic errors avoid leaking implementation details.

## Tests and current verification

The following focused tests pass in the backend virtual environment:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_google_auth.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_chat_contract.py' -v
.venv/bin/python -m compileall -q app tests
```

These tests cover verified/unverified Google identity claims, untrusted Google
issuers, blank ID tokens, missing Google configuration, blank chat messages,
stream completion order, and missing-conversation deletion behavior.

They do not prove a deployed Google OAuth callback, a live database migration,
or a real external Google token exchange. Before release, apply migrations in
the target environment and test a complete sign-in, refresh, chat, and logout
flow against the deployed API.

---

## 2. Apply the latest migration

```bash
alembic upgrade head
```

---

## 3. Check the current migration version

```bash
alembic current
```

---

## 4. View migration history

```bash
alembic history
```

---

## 5. Roll back one migration

```bash
alembic downgrade -1
```

---

## 6. Roll back to a specific revision

```bash
alembic downgrade <revision_id>
```

Example:

```bash
alembic downgrade ff1b893406a8
```

---

## Typical Workflow

```bash
# 1. Update SQLAlchemy models

# 2. Generate a migration
alembic revision --autogenerate -m "Describe your changes"

# 3. Apply the migration
alembic upgrade head
```



## How this is working 
Upload: Admin uploads a file (PDF/DOCX/TXT/MD)
Extract: Text is extracted from the file
Chunk: Text is split into ~500-word chunks with overlap
Embed: Each chunk is converted to a vector (384-dim) using all-MiniLM-L6-v2
Store: Vectors + metadata saved to Qdrant cloud
Chunks + doc metadata saved to PostgreSQL
At query time (chat):
User message → embedded to vector
Qdrant finds top 5 similar chunks
Chunks + message sent to OpenAI GPT
LLM answers using retrieved context
That's RAG: Retrieval-Augmented Generation — fetch relevant docs first, then generate.


### Setup commands: 
```
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential libpq-dev

cd /path/to/V2alarry-backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
