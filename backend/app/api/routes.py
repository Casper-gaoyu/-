from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.api.auth import auth_service, bearer_scheme, get_current_user
from app.core.database import get_db
from app.models.db_models import User
from app.models.schemas import (
    AdvanceRequest,
    AgentBehaviorRecordsUpdateRequest,
    AgentConfigUpdateRequest,
    AgentConceptImportApplyRequest,
    AgentConceptImportPreviewResponse,
    AgentDraftUpdateRequest,
    AgentFactsUpdateRequest,
    AgentHostManualSchema,
    AgentLlmDiagnosticsResponse,
    AgentLlmRuntimeProfileResponse,
    AgentLlmStrategyResponse,
    AgentOnboardingControlRequest,
    AgentRollbackRequest,
    AgentMessageRequest,
    AgentRoleCardsUpdateRequest,
    AgentRoleScriptsUpdateRequest,
    AgentRoleScriptPromptBlueprintsUpdateRequest,
    AgentSessionCreateRequest,
    AgentSessionListItem,
    AgentStateResponse,
    AgentTemplateConfigResponse,
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthResponse,
    CreateSessionResponse,
    ReviewResponse,
    ScriptStateResponse,
    UserProfileResponse,
    WorldConfig,
)
from app.services.agent_service import AgentService
from app.services.script_reference_service import ScriptReferenceService
from app.services.script_service import ScriptService
from app.services.template_config_service import TemplateConfigService

router = APIRouter()
logger = logging.getLogger(__name__)
script_service = ScriptService()
agent_service = AgentService()
script_reference_service = ScriptReferenceService()
template_config_service = TemplateConfigService()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/auth/register", response_model=AuthResponse)
def register(payload: AuthRegisterRequest, db: Session = Depends(get_db)) -> AuthResponse:
    return auth_service.register(db, payload.username, payload.password)


@router.post("/auth/login", response_model=AuthResponse)
def login(payload: AuthLoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    return auth_service.login(db, payload.username, payload.password)


@router.get("/auth/me", response_model=UserProfileResponse)
def get_me(current_user: User = Depends(get_current_user)) -> UserProfileResponse:
    return auth_service.serialize_user(current_user)


@router.post("/auth/logout")
def logout(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme), db: Session = Depends(get_db)) -> dict[str, str]:
    if credentials and credentials.credentials.strip():
        auth_service.logout(db, credentials.credentials.strip())
    return {"status": "ok"}


@router.post("/scripts/sessions", response_model=CreateSessionResponse)
async def create_session(payload: WorldConfig, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> CreateSessionResponse:
    return await script_service.create_session(db, current_user.id, payload)


@router.get("/scripts/{session_id}", response_model=ScriptStateResponse)
def get_session(session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> ScriptStateResponse:
    return script_service.get_session_state(db, current_user.id, session_id)


@router.post("/scripts/{session_id}/advance", response_model=ScriptStateResponse)
async def advance_session(session_id: str, payload: AdvanceRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> ScriptStateResponse:
    return await script_service.advance_session(db, current_user.id, session_id, payload.choice_index)


@router.post("/scripts/{session_id}/review", response_model=ReviewResponse)
async def generate_review(session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> ReviewResponse:
    return await script_service.generate_review(db, current_user.id, session_id)


@router.post("/agent/sessions", response_model=AgentStateResponse)
async def create_agent_session(payload: AgentSessionCreateRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    logger.info(
        "create_agent_session request user_id=%s template_key=%s provider=%s model=%s brief_len=%s",
        current_user.id,
        payload.template_key,
        payload.llm_provider,
        payload.llm_model,
        len(payload.brief or ""),
    )
    try:
        state = await agent_service.create_session(
            db,
            current_user.id,
            payload.brief,
            payload.llm_provider,
            payload.llm_model,
            payload.template_key,
            payload.template_form_data,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("create_agent_session failed user_id=%s template_key=%s", current_user.id, payload.template_key)
        raise HTTPException(status_code=500, detail=f"创建会话失败：{exc}") from exc

    if not state.session_id:
        logger.error("create_agent_session returned empty session_id user_id=%s template_key=%s", current_user.id, payload.template_key)
        raise HTTPException(status_code=500, detail="创建会话失败：未返回有效 session_id")

    logger.info("create_agent_session success user_id=%s session_id=%s", current_user.id, state.session_id)
    return state


@router.post("/agent/sessions/stream")
async def create_agent_session_stream(payload: AgentSessionCreateRequest, current_user: User = Depends(get_current_user)) -> StreamingResponse:
    return StreamingResponse(
        agent_service.create_session_stream(
            current_user.id,
            payload.brief,
            payload.llm_provider,
            payload.llm_model,
            payload.template_key,
            payload.template_form_data,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/agent/templates", response_model=list[AgentTemplateConfigResponse])
def list_agent_templates() -> list[AgentTemplateConfigResponse]:
    return template_config_service.list_templates()


@router.get("/agent/templates/{template_key}", response_model=AgentTemplateConfigResponse)
def get_agent_template(template_key: str) -> AgentTemplateConfigResponse:
    return template_config_service.get_template(template_key)


@router.get("/agent/sessions/{session_id}", response_model=AgentStateResponse)
def get_agent_session(session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return agent_service.get_session_state(db, current_user.id, session_id)


@router.get("/agent/sessions/{session_id}/host-manual", response_model=AgentHostManualSchema)
def get_agent_host_manual(session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentHostManualSchema:
    state = agent_service.get_session_state(db, current_user.id, session_id)
    return state.host_manual


@router.post("/agent/sessions/{session_id}/messages", response_model=AgentStateResponse)
async def send_agent_message(session_id: str, payload: AgentMessageRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return await agent_service.send_message(db, current_user.id, session_id, payload.message)


@router.post("/agent/sessions/{session_id}/messages/stream")
async def send_agent_message_stream(session_id: str, payload: AgentMessageRequest, current_user: User = Depends(get_current_user)) -> StreamingResponse:
    return StreamingResponse(
        agent_service.send_message_stream(current_user.id, session_id, payload.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.patch("/agent/sessions/{session_id}/draft", response_model=AgentStateResponse)
def update_agent_draft(session_id: str, payload: AgentDraftUpdateRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return agent_service.update_draft(db, current_user.id, session_id, payload.draft_content, payload.title)


@router.patch("/agent/sessions/{session_id}/config", response_model=AgentStateResponse)
async def update_agent_config(session_id: str, payload: AgentConfigUpdateRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return await agent_service.update_config(db, current_user.id, session_id, payload.model_dump())


@router.patch("/agent/sessions/{session_id}/role-cards", response_model=AgentStateResponse)
def update_agent_role_cards(session_id: str, payload: AgentRoleCardsUpdateRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return agent_service.update_role_cards(db, current_user.id, session_id, payload.role_cards)


@router.patch("/agent/sessions/{session_id}/role-scripts", response_model=AgentStateResponse)
def update_agent_role_scripts(session_id: str, payload: AgentRoleScriptsUpdateRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return agent_service.update_role_scripts(db, current_user.id, session_id, payload.role_scripts)


@router.patch("/agent/sessions/{session_id}/behavior-records", response_model=AgentStateResponse)
def update_agent_behavior_records(session_id: str, payload: AgentBehaviorRecordsUpdateRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return agent_service.update_behavior_records(db, current_user.id, session_id, payload.behavior_records)


@router.patch("/agent/sessions/{session_id}/facts", response_model=AgentStateResponse)
def update_agent_facts(session_id: str, payload: AgentFactsUpdateRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return agent_service.update_facts(db, current_user.id, session_id, payload.facts)


@router.patch("/agent/sessions/{session_id}/role-script-prompts", response_model=AgentStateResponse)
def update_agent_role_script_prompts(session_id: str, payload: AgentRoleScriptPromptBlueprintsUpdateRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return agent_service.update_role_script_prompt_blueprints(db, current_user.id, session_id, payload.role_script_prompt_blueprints)


@router.post("/agent/sessions/{session_id}/import-file", response_model=AgentStateResponse)
async def import_agent_concept_file(session_id: str, file: UploadFile = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    content = await file.read()
    return await agent_service.import_concept_file(
        db,
        current_user.id,
        session_id,
        file.filename or "import.txt",
        file.content_type or "application/octet-stream",
        content,
    )


@router.post("/agent/sessions/{session_id}/import-file/preview", response_model=AgentConceptImportPreviewResponse)
async def preview_agent_concept_files(session_id: str, files: list[UploadFile] = File(...), db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentConceptImportPreviewResponse:
    uploads: list[tuple[str, str, bytes]] = []
    for file in files:
        uploads.append((file.filename or "import.txt", file.content_type or "application/octet-stream", await file.read()))
    return await agent_service.preview_concept_files(db, current_user.id, session_id, uploads)


@router.post("/agent/sessions/{session_id}/import-file/apply", response_model=AgentStateResponse)
async def apply_agent_concept_import(session_id: str, payload: AgentConceptImportApplyRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return await agent_service.apply_imported_concept(
        db,
        current_user.id,
        session_id,
        payload.analysis.model_dump(),
        payload.apply_fields,
        payload.replace_role_names,
        payload.selected_facts,
    )


@router.post("/agent/sessions/{session_id}/onboarding", response_model=AgentStateResponse)
def control_agent_onboarding(session_id: str, payload: AgentOnboardingControlRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return agent_service.control_onboarding(db, current_user.id, session_id, payload.action, payload.question_id)


@router.post("/agent/sessions/{session_id}/rollback", response_model=AgentStateResponse)
def rollback_agent_session(session_id: str, payload: AgentRollbackRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> AgentStateResponse:
    return agent_service.rollback_to_revision(db, current_user.id, session_id, payload.revision_id)


@router.patch("/agent/sessions/{session_id}/hide")
def hide_agent_session(session_id: str, hidden: bool, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> dict[str, str]:
    return agent_service.set_session_hidden(db, current_user.id, session_id, hidden)


@router.delete("/agent/sessions/{session_id}")
def delete_agent_session(session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> dict[str, str]:
    return agent_service.delete_session(db, current_user.id, session_id)


@router.get("/agent/sessions", response_model=list[AgentSessionListItem])
def list_agent_sessions(include_hidden: bool = False, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> list[AgentSessionListItem]:
    return agent_service.list_sessions(db, current_user.id, include_hidden)


@router.get("/agent/llm/runtime-profile", response_model=AgentLlmRuntimeProfileResponse)
def get_agent_llm_runtime_profile() -> AgentLlmRuntimeProfileResponse:
    return agent_service.llm_service.get_runtime_profile()


@router.get("/agent/llm/diagnostics", response_model=AgentLlmDiagnosticsResponse)
async def get_agent_llm_diagnostics() -> AgentLlmDiagnosticsResponse:
    return await agent_service.llm_service.get_runtime_diagnostics()


@router.get("/agent/llm/strategy", response_model=AgentLlmStrategyResponse)
def get_agent_llm_strategy() -> AgentLlmStrategyResponse:
    return agent_service.llm_service.get_model_strategy()


@router.get("/agent/reference-library/status")
def get_reference_library_status(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> dict:
    try:
        rebuild_status = script_reference_service.get_rebuild_status()
    except Exception:
        rebuild_status = {
            "running": False,
            "completed": False,
            "root": "",
            "documents": 0,
            "fragments": 0,
            "processed_files": 0,
            "total_files": 0,
            "skipped": 0,
            "last_error": "",
            "started_at": "",
            "finished_at": "",
        }
    try:
        stats = script_reference_service.get_library_stats(db)
    except Exception:
        stats = {
            "documents": 0,
            "fragments": 0,
            "documents_by_category": {},
            "fragments_by_type": {},
        }
    return {
        "rebuild_status": rebuild_status,
        "stats": stats,
    }


@router.post("/agent/reference-library/rebuild")
def rebuild_reference_library(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> dict:
    return script_reference_service.ingest_library(db)


@router.get("/agent/reference-library/search")
def search_reference_library(query: str, limit: int = 5, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)) -> dict:
    return {"query": query, "results": script_reference_service.search_fragments(db, query=query, limit=limit)}
