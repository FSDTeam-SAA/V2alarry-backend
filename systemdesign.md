# V2alarry Backend System Design

## Overview
V2alarry Backend provides an AI-powered knowledge base and conversational interface. It is built on FastAPI with support for user authentication, document ingestion, vector search, conversational retrieval, and assisted chat.

## Core Components

### 1. API Layer
- **FastAPI application** (`app/main.py`)
- Exposes endpoints for:
  - Authentication: `/api/v1/auth`
  - User profile: `/api/v1/users`
  - Admin document management: `/api/v1/admin/documents`
  - User chat: `/api/v1/chat`
- Middleware:
  - CORS support configured with allowed origins from environment settings.
- Startup hook ensures database schema compatibility.

### 2. Authentication & Authorization
- **JWT-based auth** with access tokens and refresh tokens.
- Password-based login and Google OAuth login support.
- Refresh tokens are hashed and stored in the database.
- Protected routes use bearer token validation via `OAuth2PasswordBearer`.
- Admin-only endpoints are enforced in the document upload and management router.

### 3. Database Persistence
- SQLAlchemy models define the relational schema.
- Key entities include:
  - `User`: authenticated application users with roles and Google subject mapping.
  - `Document`: uploaded files tracked with metadata, status, and chunk counts.
  - `DocumentChunk`: individual text chunks linked to vector IDs.
  - `Conversation`: chat sessions scoped to a user.
  - `Message`: conversation messages with role metadata.
- Database access is asynchronous through `AsyncSession` and repository patterns.

### 4. Document Ingestion Pipeline
- Document upload endpoint accepts PDF, DOCX, TXT, and MD.
- Files are validated for type and size.
- Document processing includes:
  - Saving the file locally in `uploads/documents`
  - Text extraction from file content
  - Chunking text into overlapping segments
  - Generating embeddings using Sentence-Transformers
  - Writing embeddings to a vector store
  - Persisting document and chunk metadata in the database
- Document deletion removes both DB records and vector store entries.

### 5. Embeddings & Vector Search
- Embedding model: `all-MiniLM-L6-v2` by default.
- Embeddings are cached in Redis to reduce repeated computation.
- The vector store supports adding and deleting vectors, and semantic search.
- Search uses similarity thresholding and top-k retrieval to identify relevant document chunks.

### 6. Conversational Workflow
- Chat requests can run in normal or streaming mode.
- Workflow stages:
  1. Retrieve user conversation history and document context.
  2. Build a combined prompt context from relevant document chunks and recent user history.
  3. Generate an LLM response.
  4. Persist conversation turns and update conversation state.
- The workflow is implemented using `langgraph.graph.StateGraph` and custom nodes.
- Support for streaming responses using Server-Sent Events (SSE).

### 7. Caching
- Redis caching is used for:
  - Embedding results
  - Recent conversation history
  - Search results
- Cache TTLs are configurable in settings.

## Current Functional Flow

1. **User Registration / Login**
   - New users register with email/password.
   - Existing users authenticate and receive JWT access/refresh tokens.
   - Google login verifies tokens and maps or creates users.

2. **Document Upload**
   - Admin uploads knowledge base documents.
   - System extracts, chunks, embeds, and stores document content.
   - Document metadata and status are tracked in the relational database.

3. **Chat Request**
   - User sends a message to `/api/v1/chat`.
   - The system retrieves conversation history and similar document chunks.
   - It generates a context-aware prompt for the LLM.
   - A response is returned and the chat turn is persisted.

4. **Conversation History**
   - Users can list conversations.
   - Users can fetch messages within a conversation.
   - Users can delete conversations.

## Architecture Diagram (Conceptual)

- Client (frontend or API client)
  - Sends requests to FastAPI endpoints

- FastAPI Backend
  - Auth module
  - Document admin router
  - Chat router

- Services
  - `DocumentService`
  - `EmbeddingService`
  - `ChatHistoryService`
  - `VectorStore`
  - `LLMService`

- Data stores
  - Relational DB for users, documents, chats
  - Qdrant / vector DB for embeddings
  - Redis for caching

- External systems
  - OpenAI / LLM provider
  - Google token verification for social login

## Strengths
- Modular separation of responsibilities across services.
- RAG-enabled chat with document retrieval and history.
- Streaming support for interactive user experience.
- Document lifecycle management with status tracking.
- Extensible settings for embeddings, LLM, and caching.

## Current Limitations
- Document processing is synchronous and may block on large files.
- No background worker system for long-running ingestion tasks.
- Limited admin operations and knowledge base metadata.
- Vector search filter is simplistic and only supports active documents.
- No explicit rate limiting or advanced security hardening.
- Conversation context is limited to a fixed history window.

## Future Scope

### 1. Scalability & Reliability
- Add asynchronous background jobs for document ingestion and vector indexing.
- Introduce task queueing with Celery, RQ, or Dramatiq.
- Add health checks and readiness probes for DB, Redis, and vector store.
- Support horizontal scaling with Kubernetes or container orchestration.

### 2. Enhanced Knowledge Base
- Add document versioning, tagging, and richer metadata.
- Support more file formats, including HTML, PPTX, and spreadsheets.
- Implement document search and filtering in admin endpoints.
- Add a knowledge base browser with categories, status, and preview.

### 3. Better Retrieval and RAG
- Add query rewriting and semantic search improvements.
- Add multi-vector retrieval for multi-hop reasoning.
- Support document chunk ranking and answer attribution.
- Expose a dedicated search endpoint for semantic document search.

### 4. Conversation Improvements
- Add multi-session user context and conversation summary generation.
- Support message editing, pinning, and annotation.
- Add fine-grained conversation privacy controls.
- Support multi-user and team workspaces.

### 5. Security and Governance
- Add role-based access control beyond admin/user.
- Add OAuth provider integrations: GitHub, Microsoft, LinkedIn.
- Add logging, auditing, and admin activity tracking.
- Introduce rate limiting, request quotas, and abuse protection.

### 6. Model & LLM Enhancements
- Make LLM providers pluggable with a provider interface.
- Support model selection and dynamic temperature/tokens per request.
- Add response caching and prompt tuning for improved latency.
- Add fallback or ensemble approaches for failover.

### 7. Observability and Monitoring
- Add request tracing and metrics for API, ingestion, and search.
- Add logging for vector retrieval quality and prompt usage.
- Add dashboards for document ingestion, vector count, and chat activity.

## Recommendations for Next Iteration
- Implement a background processing queue for document uploads.
- Add explicit conversation summary and context refresh logic.
- Upgrade the vector search pipeline to support document filtering by metadata and more advanced similarity scoring.
- Harden auth flows with token revocation, refresh token rotation, and password reset.
- Add more complete API documentation and OpenAPI schema coverage.

## Summary
The current backend is a solid foundation for an AI-assisted knowledge platform, with strong core features: authenticated chat, document ingestion, vector search, and RAG workflows. The next phase should focus on scaling ingestion, improving retrieval relevance, expanding knowledge base management, and adding operational resiliency.
