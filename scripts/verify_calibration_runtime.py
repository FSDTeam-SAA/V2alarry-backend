"""Verify real PostgreSQL/auth/HTTP persistence with deterministic model and retrieval doubles."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


async def verify(url: str, output: Path, serve_port: int = 0, live_model: bool = False):
    from sqlalchemy.engine import make_url
    parsed = make_url(url)
    if parsed.host not in {"127.0.0.1", "localhost"} or not (parsed.database or "").startswith("leadercoach_calibration"):
        raise ValueError("Use a disposable localhost leadercoach_calibration database")
    os.environ["DATABASE_URL"] = url
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from httpx import ASGITransport, AsyncClient
    from fastapi import FastAPI
    from sqlalchemy import select
    from app.api.v1.user import chat as chat_module
    from app.core.agreements import CURRENT_AGREEMENT_VERSION
    from app.db.base import Base
    from app.db.database import engine, AsyncSessionLocal
    from app.models.user import User
    from app.models.message import Message
    from app.models.coaching_summary import CoachingSummary
    from app.models.agreement_acceptance import AgreementAcceptance
    from app.services.chat_history_service import ChatHistoryService
    from app.services.coaching_memory_service import CoachingMemoryService
    from app.schemas.coaching import CoachingWorkingState, CoachingSummaryData, CoachingSummaryRevision, CoachingTurnAssessment, CoachingEvidenceState
    from app.utils.jwt import create_access_token
    from app.utils.security import has_password
    from app.workflows.chat_workflow import ChatWorkflow, WorkflowNodes

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with AsyncSessionLocal() as db:
        run_id = uuid.uuid4().hex[:8]
        users = [User(email=f"runtime-{run_id}-{number}@example.com", full_name="Synthetic Test",
                      hashed_password=has_password("Calibration-browser-2026!"), password_login_enabled=True) for number in (1, 2)]
        db.add_all(users)
        await db.flush()
        for user in users:
            db.add(AgreementAcceptance(user_id=user.id, agreement_version=CURRENT_AGREEMENT_VERSION, source="runtime_test"))
        await db.commit()
        user_ids = [user.id for user in users]
        browser_email = users[0].email

    class TestLLM:
        async def generate(self, **kwargs):
            if "query rewriting" in kwargs["system_prompt"]:
                return "Synthetic review workload"
            return "What matters most to you about having that influence?"

        async def generate_stream(self, **kwargs):
            yield "What matters most to you "
            yield "about having that influence?"

        async def generate_structured(self, *, schema, **kwargs):
            payload = json.loads(kwargs["user_message"])
            if issubclass(schema, CoachingTurnAssessment):
                wording = "How would I open that meeting?" == payload["new_user_message"]
                actions = payload["extracted_state"].get("action_context", [])
                previous = payload["previous_state"].get("understanding_context")
                continuing = bool(payload.get("latest_assistant_response"))
                return CoachingTurnAssessment(
                    focus_assessment={"focus": "Client's desired clarity", "reason": "Client wants to think clearly",
                        "evidence_ids": ["current_message"]},
                    interaction_assessment=None,
                    inquiry_assessment={"progress": "advancing", "reason": "Client is clarifying their purpose",
                                        "client_requested_pause": False},
                    coaching_stage="possibilities" if wording else "discovery",
                    next_coaching_move="support_choice" if wording else "explore_desired_help",
                    client_response_to_last_move="not_applicable", client_generation="emerging",
                    essential_unknowns=[] if wording else ["desired_help"], transition_basis=["grounded_evidence"],
                    knowledge_gap_observed=False, user_owned_outcome=False, tested_understanding=False,
                    transition_evidence_ids=[],
                    requested_support="action_wording" if wording else "exploration",
                    support_action_id=actions[0]["id"] if wording else None,
                    progress_assessment={"established": ["The user wants clarity before deciding on action"],
                        "material_unknown": None, "discovery_useful": not wording,
                        "understanding_effect": "retain" if previous else "establish" if continuing else "unknown",
                        "recommended_direction": "support" if wording else "discovery",
                        "reason": "Separate clarity, preparation and action", "evidence_ids": ["current_message"]},
                    understanding_assessment={"picture": "The user wants clarity about priorities before deciding how to approach the boss",
                        "basis": "natural", "evidence_ids": ["current_message"]} if continuing and not previous else None,
                    support_assessment={"scope": "preparation", "permitted": True,
                        "reason": "The user requests an opening, without commitment", "evidence_ids": ["current_message"]} if wording else None,
                )
            if schema is CoachingEvidenceState:
                message = payload["new_user_message"]
                updates = []
                if "not comfortable requesting" in message or "decided to abandon" in message:
                    old = payload["previous_state"].get("action_context", [])
                    updates = [{"id": old[0]["id"] if old else None, "actor": "client",
                        "action": "request a meeting with the boss",
                        "status": "abandoned" if "decided to abandon" in message else "considering",
                        "readiness": None if "decided to abandon" in message else "not_ready",
                        "basis": "reported_action" if "decided to abandon" in message else "readiness_boundary",
                        "boundary_scope": None if "decided to abandon" in message else "action",
                        "evidence": [{"source": "current_message", "excerpt": message}]}]
                return CoachingEvidenceState(reported_facts=["The client can negotiate priorities"], action_updates=updates)
            if schema is CoachingSummaryRevision:
                changes = {}
                previous = payload["previous_summary"] or {}
                abandoned = next((item for item in payload["working_state"]["action_context"] if item["status"] == "abandoned"), None)
                if abandoned:
                    changes = {f"{field}_change": {"operation": "clear", "previous_value": previous[field],
                        "action_id": abandoned["id"], "evidence": [{"source": "current_message", "excerpt": payload["latest_turn"]["leader"]}]}
                        for field in ("commitment", "next_experiment") if previous.get(field)}
                return CoachingSummaryRevision(presenting_focus="Client's influence over priorities", **changes)
            return schema()

    timings = {}
    if live_model:
        from app.core.llm import LLMService

        class LiveLLM(LLMService):
            def __init__(self):
                from app.core.config import settings
                from langchain_openai import ChatOpenAI
                import tiktoken
                self.llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.MAX_TOKENS, openai_api_key=settings.OPENAI_API_KEY, timeout=25, max_retries=0)
                self.encoder = tiktoken.get_encoding("cl100k_base")
                self.calls = 0
                self.reserved_tokens = 0

            def reserve(self, *, schema=None, **kwargs):
                from app.core.config import settings
                estimate = sum(len(self.encoder.encode(str(m.content))) + 16 for m in self.build_messages(**kwargs))
                if schema:
                    estimate += len(self.encoder.encode(json.dumps(schema.model_json_schema())))
                reservation = estimate + settings.MAX_TOKENS
                if self.calls >= 32 or self.reserved_tokens + reservation > 450000:
                    raise RuntimeError("Isolated runtime model budget exhausted")
                self.calls += 1
                self.reserved_tokens += reservation

            async def generate(self, **kwargs):
                self.reserve(**kwargs)
                return await super().generate(**kwargs)

            async def generate_structured(self, *, schema, **kwargs):
                self.reserve(schema=schema, **kwargs)
                return await super().generate_structured(schema=schema, **kwargs)

            async def generate_stream(self, **kwargs):
                self.reserve(**kwargs)
                async for token in super().generate_stream(**kwargs):
                    timings.setdefault("sse_first_token_seconds", round(time.monotonic() - request_started, 3))
                    yield token

    nodes = WorkflowNodes.__new__(WorkflowNodes)
    nodes.chat_history_service = ChatHistoryService()
    nodes.llm_service = LiveLLM() if live_model else TestLLM()
    summary_results = []

    class ObservedMemory(CoachingMemoryService):
        async def refresh_summary(self, **kwargs):
            try:
                result = await super().refresh_summary(**kwargs)
            except Exception as exc:
                summary_results.append(getattr(exc, "category", type(exc).__name__))
                raise
            summary_results.append("updated")
            return result

    nodes.coaching_memory_service = ObservedMemory(nodes.llm_service)
    nodes.document_service = SimpleNamespace(get_accessible_document_ids=AsyncMock(return_value=[]))
    nodes._fetch_documents = AsyncMock(return_value=[])
    workflow = ChatWorkflow.__new__(ChatWorkflow)
    workflow.nodes = nodes
    workflow.workflow = workflow._create_workflow()
    workflow.app = workflow.workflow.compile()
    app = FastAPI()
    if serve_port:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3330", "http://127.0.0.1:3330",
                                                       "http://localhost:3331", "http://127.0.0.1:3331"],
                           allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.include_router(chat_module.router, prefix="/api/v1")
    checks = {}
    with (
        patch.object(chat_module, "get_chat_workflow", return_value=workflow),
        patch("app.workflows.chat_workflow.cache_get", new=AsyncMock(return_value=None)),
        patch("app.workflows.chat_workflow.cache_set", new=AsyncMock()),
        patch("app.workflows.chat_workflow.cache_delete_safe", new=AsyncMock()),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime.test") as client:
            headers = {"Authorization": "Bearer " + create_access_token({"sub": str(user_ids[0])})}
            other = {"Authorization": "Bearer " + create_access_token({"sub": str(user_ids[1])})}
            denied = await client.post("/api/v1/chat/", json={"message": "Hello"})
            assert denied.status_code == 401
            checks["authentication"] = True
            request_started = time.monotonic()
            response = await client.post("/api/v1/chat/", headers=headers, json={"message": "I can negotiate priorities, and I want to understand what I need before speaking to my director."})
            timings["json_total_seconds"] = round(time.monotonic() - request_started, 3)
            assert response.status_code == 200, response.text
            result = response.json()
            conversation_id = result["conversation_id"]
            assert "coaching_trace" not in result["metadata"]
            checks["json_transport"] = True
            request_started = time.monotonic()
            stream = await client.post("/api/v1/chat/", headers=headers, json={"message": "I want clarity about my own priorities first, rather than a plan for the director.", "conversation_id": conversation_id, "stream": True})
            timings["sse_total_seconds"] = round(time.monotonic() - request_started, 3)
            events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
            streamed_reply = "".join(event.get("content", "") for event in events if event["type"] == "token")
            assert streamed_reply
            if not live_model:
                assert streamed_reply == "What matters most to you about having that influence?"
            assert events[-1]["type"] == "done" and events[-1]["persisted"] is True
            checks["sse_transport"] = True
            messages = (await client.get(f"/api/v1/chat/conversations/{conversation_id}/messages", headers=headers)).json()
            assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
            assert messages[-1]["content"] == streamed_reply
            assert all(m.get("metadata") is None for m in messages)
            checks["reload_order_and_privacy"] = True
            forbidden = await client.post("/api/v1/chat/", headers=other, json={"message": "Attempt", "conversation_id": conversation_id})
            assert forbidden.status_code == 400
            leaked = await client.get(f"/api/v1/chat/conversations/{conversation_id}/messages", headers=other)
            assert leaked.json() == []
            checks["ownership_isolation"] = True
            await ChatHistoryService().save_summary(conversation_id=conversation_id, user_id=str(user_ids[0]),
                summary=CoachingSummaryData(commitment="Request a meeting with the boss."), source_turn_count=2)
            for message in ("I am not comfortable requesting a meeting with my boss yet.", "How would I open that meeting?"):
                response = await client.post("/api/v1/chat/", headers=headers,
                    json={"message": message, "conversation_id": conversation_id})
                assert response.status_code == 200, response.text
            assert response.json()["metadata"]["next_coaching_move"] == "support_choice"
            boundary_state = await ChatHistoryService().get_working_state(conversation_id=conversation_id, user_id=str(user_ids[0]))
            assert any(item["readiness"] == "not_ready" for item in boundary_state.state["action_context"])
            checks["action_boundary_resume"] = True
            assert boundary_state.state["understanding_context"]["status"] == "supported"
            assert all(item["status"] != "planned" for item in boundary_state.state["action_context"])
            checks["preparation_without_commitment"] = True
            response = await client.post("/api/v1/chat/", headers=headers, json={
                "message": "I have decided to abandon that meeting request. I no longer want to do it.", "conversation_id": conversation_id})
            assert response.status_code == 200, response.text
    async with AsyncSessionLocal() as db:
        import uuid as uuid_module
        saved = list((await db.execute(select(Message).where(Message.conversation_id == uuid_module.UUID(conversation_id)).order_by(Message.created_at))).scalars())
        assert len(saved) == 10
        for user, assistant in zip(saved[::2], saved[1::2]):
            assert user.doc_metadata["coaching_trace"]["version"] == "calibration-5"
            assert user.doc_metadata["coaching_trace"]["policy_reason_codes"]
            assert user.doc_metadata["assistant_message_id"] == str(assistant.id)
            assert assistant.doc_metadata["user_message_id"] == str(user.id)
            assert user.doc_metadata["turn_id"] == assistant.doc_metadata["turn_id"]
            if user.doc_metadata["source_turn_count"] > 1:
                assert isinstance(user.doc_metadata["coaching_trace"]["inquiry_assessment"], dict)
            if live_model:
                assert user.doc_metadata["coaching_trace"]["extraction_status"] == "updated"
                assert user.doc_metadata["coaching_trace"]["assessment_status"] == "updated"
        checks["postgres_readback"] = True
        summary = (await db.execute(select(CoachingSummary).where(CoachingSummary.conversation_id == uuid_module.UUID(conversation_id)))).scalar_one()
        assert summary.source_turn_count == 5 and summary.generation_status == "current"
        assert summary.commitment is None
        checks["abandoned_commitment_cleared"] = True
        checks["continuity_refresh"] = True
        # Reproduce the timestamp tie in legacy rows using only this run's synthetic pair.
        saved[1].created_at = saved[0].created_at
        await db.commit()
    service = ChatHistoryService()
    tied = await service.get_conversation_messages(conversation_id, str(user_ids[0]))
    assert [message["role"] for message in tied] == ["user", "assistant"] * 5
    latest = await service.get_conversation_history(conversation_id, str(user_ids[0]), limit=2)
    assert [message["role"] for message in latest] == ["user", "assistant"]
    checks["legacy_timestamp_order"] = True
    # A new service reads persisted continuity rather than relying on an in-memory workflow.
    persisted = await ChatHistoryService().get_working_state(conversation_id=conversation_id, user_id=str(user_ids[0]))
    assert persisted.source_turn_count == 5
    assert not {"focus_assessment", "interaction_assessment", "inquiry_assessment", "action_updates", "requested_support", "support_action_id", "progress_assessment", "support_assessment"} & persisted.state.keys()
    checks["state_resume"] = True
    from app.core.config import settings
    output.write_text(json.dumps({"checks": checks, "external_model": settings.LLM_MODEL if live_model else "doubled",
                                 "production_database": "untouched", "timings": timings,
                                 "working_state": persisted.state, "summary_results": summary_results,
                                 "model_budget": {"calls": nodes.llm_service.calls, "reserved_tokens": nodes.llm_service.reserved_tokens} if live_model else None}, indent=2))
    print(json.dumps(checks))
    if serve_port:
        import uvicorn
        from app.api.v1.auth import router as auth_router
        from app.api.v1.users import router as users_router
        import app.workflows.chat_workflow as workflow_module
        app.include_router(auth_router, prefix="/api/v1/auth")
        app.include_router(users_router, prefix="/api/v1/users")
        chat_module.get_chat_workflow = lambda: workflow
        workflow_module.cache_get = AsyncMock(return_value=None)
        workflow_module.cache_set = AsyncMock()
        workflow_module.cache_delete_safe = AsyncMock()
        output.write_text(json.dumps({"checks": checks, "browser_email": browser_email,
                                      "browser_password": "Calibration-browser-2026!", "external_model": "doubled"}, indent=2))
        await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=serve_port, log_level="warning")).serve()
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serve-port", type=int, default=0, help="Optionally keep the isolated app running for a browser test")
    parser.add_argument("--live-model", action="store_true", help="Use the configured provider for five synthetic authenticated turns; retrieval remains doubled")
    arguments = parser.parse_args()
    asyncio.run(verify(arguments.database_url, arguments.output, arguments.serve_port, arguments.live_model))
