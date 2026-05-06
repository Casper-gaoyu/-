import asyncio
import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.db_models import ScriptSession
from app.models.schemas import AgentStateResponse
from app.services.dm_service import DMService
from app.services.host_manual_service import HostManualService
from app.services.intent_service import IntentService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.concept_import_service import ConceptImportService
from app.services.role_script_service import RoleScriptLongformService
from app.services.render_service import RenderService
from app.services.rule_engine import RuleEngine
from app.services.structure_engine import StructureEngine
from app.services.template_config_service import DEFAULT_TEMPLATE_KEY, TemplateConfigService
from app.services.template_runtime_service import TemplateRuntimeService


class AgentService:
    MAX_SUGGESTIONS = 6
    PLAN_PREVIEW_TOKENS = (
        "先给计划",
        "先出计划",
        "先列计划",
        "先看计划",
        "先说说计划",
        "先规划一下",
        "先规划",
        "先别执行",
        "先不要执行",
        "不要直接改",
        "不要直接生成",
        "确认后再执行",
        "确认后再改",
        "先告诉我你准备怎么改",
        "先告诉我怎么改",
    )
    SIMPLE_RENAME_PATTERNS = (
        re.compile(
            r"(?:请|麻烦你|麻烦)?(?:把|将)(?P<old>[^，。；：“”\"'、]{1,24}?)(?:这个)?(?:名字)?(?:改成|改为|换成|替换成|改叫)(?P<new>[^，。；！？!?]{1,24})(?:，.*)?"
        ),
        re.compile(
            r"(?P<old>[^，。；：“”\"'、]{1,24}?)(?:改成|改为|换成)(?P<new>[^，。；！？!?]{1,24})(?:，.*)?"
        ),
    )
    TITLE_RENAME_PATTERNS = (
        re.compile(
            r"(?:请|麻烦你|麻烦)?(?:把|将)?(?:剧本|标题|题目|名称)(?:改成|改为|换成|命名为|改叫)(?P<new>[^，。；！？!?]{1,40})(?:，.*)?"
        ),
        re.compile(
            r"(?:请|麻烦你|麻烦)?(?:把|将)(?P<old>[^，。；：“”\"'、]{1,40}?)(?:这个)?(?:剧本|标题|题目|名称)(?:改成|改为|换成|命名为|改叫)(?P<new>[^，。；！？!?]{1,40})(?:，.*)?"
        ),
    )
    QUESTION_FLOW = [
        {
            "id": "genre",
            "question": "第 1 步：你想做什么类型的剧本？",
            "placeholder": "例如：恐怖、情感、硬核、欢乐、现代悬疑",
            "options": ["现代悬疑", "恐怖", "情感", "硬核", "欢乐"],
            "explanation": "剧本类型决定整体体验重心，比如偏推理、偏情绪、偏恐怖氛围，还是偏轻松互动。",
            "examples": ["欢乐本", "硬核推理本", "情感沉浸本"],
        },
        {
            "id": "era",
            "question": "第 2 步：故事发生在什么时代或世界观？",
            "placeholder": "例如：现代校园、民国小镇、未来空间站、古风山门",
            "options": ["现代都市", "现代校园", "民国小镇", "古风武侠", "未来科幻"],
            "explanation": "这里决定故事发生的舞台，也会影响人物身份、线索样式和语言风格。",
            "examples": ["中式现代都市", "民国戏班", "未来空间站"],
        },
        {
            "id": "players",
            "question": "第 3 步：预计几人参与？",
            "placeholder": "例如：6 人、8 人、10 人",
            "options": ["4 人", "6 人", "8 人", "10 人"],
            "explanation": "人数会直接影响角色数量、信息分配和流程复杂度，先给一个大致范围就够。",
            "examples": ["6 人", "6-8 人", "8 人左右"],
        },
        {
            "id": "conflict",
            "question": "第 4 步：这局最核心的冲突是什么？",
            "placeholder": "例如：密室命案、旧案重启、相亲翻车、家族争产",
            "options": ["密室命案", "旧案重启", "相亲翻车", "家族争产"],
            "explanation": "核心冲突就是整局故事最主要的矛盾来源，决定玩家为什么会卷进这场局，以及剧情为什么会持续推进。",
            "examples": ["密室命案背后的旧案重启", "相亲局上的身份翻车", "家族争产引发互相拆台"],
        },
        {
            "id": "protagonist",
            "question": "第 5 步：你希望主控视角或关键角色是什么身份？",
            "placeholder": "例如：记者、医生、学生、侦探、组织者",
            "options": ["记者", "医生", "学生", "侦探", "组织者"],
            "explanation": "主控视角就是最适合带着玩家理解剧情的关键角色身份，不一定是主角，但通常掌握重要信息或推动关键选择。",
            "examples": ["知道部分真相的记者", "控制局面的主持人", "意外卷入的学生"],
        },
        {
            "id": "goal",
            "question": "第 6 步：玩家最后要完成什么目标？",
            "placeholder": "例如：找出真凶、揭开真相、完成配对、达成站队",
            "options": ["找出真凶", "揭开真相", "完成配对", "达成站队"],
            "explanation": "这个目标决定结局怎么收束，也决定系统后面该怎么生成分支和复盘。",
            "examples": ["找出真凶", "揭开真相", "完成配对"],
        },
    ]

    def __init__(self) -> None:
        self.memory_service = MemoryService()
        self.llm_service = LLMService(RuleEngine())
        self.intent_service = IntentService()
        self.structure_engine = StructureEngine()
        self.dm_service = DMService()
        self.render_service = RenderService()
        self.host_manual_service = HostManualService()
        self.role_script_service = RoleScriptLongformService()
        self.concept_import_service = ConceptImportService()
        self.template_config_service = TemplateConfigService()
        self.template_runtime_service = TemplateRuntimeService()

    async def create_session(
        self,
        db: Session,
        owner_id: int,
        brief: str,
        llm_provider: str = "",
        llm_model: str = "",
        template_key: str = "",
        template_form_data: dict[str, str] | None = None,
    ) -> AgentStateResponse:
        record = ScriptSession(
            id=str(uuid4()),
            owner_id=owner_id,
            title="待命名剧本",
            world_config={},
            memory={},
            latest_segment="",
            latest_branches=[],
        )
        db.add(record)
        db.flush()

        memory = self._build_initial_memory(
            record.title,
            brief,
            llm_provider,
            llm_model,
            template_key,
            template_form_data or {},
        )
        if memory["onboarding"]["completed"]:
            if self._co_creation_enabled(memory):
                payload = self._build_co_creation_payload(memory, intro=True)
                state = self._apply_payload(db, record, memory, payload, "guide")
            else:
                state = await self._generate_turn(db, record, memory)
        else:
            payload = self._build_onboarding_payload(memory, brief, is_initial=True)
            state = self._apply_payload(db, record, memory, payload, "guide")
        db.commit()
        db.refresh(record)
        return state

    async def create_session_stream(
        self,
        owner_id: int,
        brief: str,
        llm_provider: str = "",
        llm_model: str = "",
        template_key: str = "",
        template_form_data: dict[str, str] | None = None,
    ):
        session_id = str(uuid4())
        record_title = "待命名剧本"
        memory = self._build_initial_memory(
            record_title,
            brief,
            llm_provider,
            llm_model,
            template_key,
            template_form_data or {},
        )
        async for event in self._stream_session(session_id, owner_id, record_title, memory, is_new_session=True):
            yield event

    def get_session_state(self, db: Session, owner_id: int, session_id: str) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        role_cards_changed = self._sanitize_memory_role_cards(memory)
        original_role_scripts = copy.deepcopy(memory.get("role_scripts", []))
        repaired_role_scripts = self.role_script_service.build(record.title, memory)
        role_scripts_changed = [item.model_dump() for item in repaired_role_scripts] != original_role_scripts
        if role_scripts_changed:
            memory["role_scripts"] = [item.model_dump() for item in repaired_role_scripts]
        if role_cards_changed or role_scripts_changed:
            db.add(record)
            self.memory_service.save(record, memory)
            db.commit()
            db.refresh(record)
        return self._build_state(record, memory)

    def list_sessions(self, db: Session, owner_id: int, include_hidden: bool = False) -> list[dict]:
        records = db.execute(
            select(ScriptSession).where(ScriptSession.owner_id == owner_id).order_by(ScriptSession.updated_at.desc())
        ).scalars().all()
        items: list[dict] = []
        for record in records:
            memory = record.memory or {}
            if memory.get("mode") != "agent":
                continue
            if record.is_hidden and not include_hidden:
                continue

            messages = memory.get("messages", [])
            assistant_messages = [msg["content"] for msg in messages if msg.get("role") == "assistant" and msg.get("content")]
            last_message = assistant_messages[-1] if assistant_messages else ""
            onboarding = memory.get("onboarding", self._empty_onboarding_state())
            summary = memory.get("summary", {})
            status = "drafting"
            if summary.get("focus") == "手动编辑当前稿件":
                status = "edited"
            elif onboarding.get("completed", False):
                status = "completed"
            items.append(
                {
                    "session_id": record.id,
                    "title": record.title,
                    "last_message": last_message[:120],
                    "updated_at": record.updated_at.isoformat(sep=" ", timespec="seconds"),
                    "provider": memory.get("llm_profile", {}).get("provider") or memory.get("provider", "mock"),
                    "is_hidden": bool(record.is_hidden),
                    "status": status,
                    "onboarding_completed": onboarding.get("completed", False),
                    "revision_count": int(summary.get("revision_count", 0)),
                    "template_key": summary.get("template_key", memory.get("template_key", DEFAULT_TEMPLATE_KEY)),
                    "template_label": summary.get("template_label", memory.get("template_label", "通用剧本模板")),
                }
            )
        return items

    async def send_message(self, db: Session, owner_id: int, session_id: str, message: str) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        self._sanitize_memory_role_cards(memory)

        self._append_message(memory, "user", message)
        if not memory.get("onboarding", {}).get("completed"):
            if self._is_clarification_request(memory, message):
                payload = self._build_clarification_payload(memory)
                state = self._apply_payload(db, record, memory, payload, "guide")
            else:
                validation_error = self._validate_onboarding_answer(
                    memory.get("onboarding", {}).get("current_question_id", ""),
                    message,
                )
                if validation_error:
                    payload = self._build_validation_payload(memory, validation_error)
                    state = self._apply_payload(db, record, memory, payload, "guide")
                else:
                    self._record_onboarding_answer(memory, message)
                    if not memory["onboarding"]["completed"]:
                        payload = self._build_onboarding_payload(memory, message)
                        state = self._apply_payload(db, record, memory, payload, "guide")
                    else:
                        if self._co_creation_enabled(memory):
                            payload, provider = self._process_co_creation_turn(memory, message, intro=True)
                            state = self._apply_payload(db, record, memory, payload, provider)
                        else:
                            state = await self._generate_turn(db, record, memory)
        else:
            if (local_payload := self._maybe_build_local_edit_payload(memory, message)) is not None:
                state = self._apply_payload(db, record, memory, local_payload, "local")
                db.commit()
                db.refresh(record)
                return state
            if (title_payload := self._maybe_build_local_title_refresh_payload(memory, message)) is not None:
                state = self._apply_help_payload(db, record, memory, title_payload, "local")
                db.commit()
                db.refresh(record)
                return state
            if self._co_creation_enabled(memory):
                if self._is_co_creation_finish_request(message):
                    state = await self._generate_co_creation_stage_output(db, record, memory, force_full_script=True)
                    db.commit()
                    db.refresh(record)
                    return state
                if self._is_co_creation_generate_request(memory, message):
                    state = await self._generate_co_creation_stage_output(db, record, memory, force_full_script=False)
                    db.commit()
                    db.refresh(record)
                    return state
                current_stage = self._get_current_co_creation_stage(memory) or {}
                if str(current_stage.get("id") or "").strip() == "detail_iteration" and message.strip():
                    memory["active_generation_request"] = message.strip()
                    state = await self._generate_turn(db, record, memory)
                    db.commit()
                    db.refresh(record)
                    return state
                payload, provider = self._process_co_creation_turn(memory, message, intro=False)
                state = self._apply_payload(db, record, memory, payload, provider)
                db.commit()
                db.refresh(record)
                return state
            route = self._resolve_runtime_route(memory, message)
            if not memory.get("draft_content", "").strip():
                state = await self._generate_turn(db, record, memory)
            elif memory.get("pending_plan"):
                if self._is_plan_confirmation(message):
                    memory["active_generation_request"] = memory["pending_plan"]["request"]
                    memory["pending_plan"] = None
                    state = await self._generate_turn(db, record, memory)
                elif self._is_plan_cancellation(message):
                    payload = self._build_plan_cancel_payload(memory)
                    state = self._apply_help_payload(db, record, memory, payload, "guide")
                else:
                    payload = self._build_pending_plan_payload(memory, message, route)
                    state = self._apply_help_payload(db, record, memory, payload, "guide")
            elif (local_payload := self._maybe_build_local_edit_payload(memory, message)) is not None:
                state = self._apply_payload(db, record, memory, local_payload, "local")
            elif (title_payload := self._maybe_build_local_title_refresh_payload(memory, message)) is not None:
                state = self._apply_help_payload(db, record, memory, title_payload, "local")
            elif route.get("response_mode") == "help":
                payload = self._build_workspace_help_payload(memory, message)
                state = self._apply_help_payload(db, record, memory, payload, "guide")
            elif route.get("response_mode") == "answer":
                payload = self._build_analysis_payload(memory, message, route)
                state = self._apply_help_payload(db, record, memory, payload, "guide")
            elif self._should_require_plan_confirmation(memory, message, route):
                payload = self._build_pending_plan_payload(memory, message, route)
                state = self._apply_help_payload(db, record, memory, payload, "guide")
            else:
                state = await self._generate_turn(db, record, memory)

        db.commit()
        db.refresh(record)
        return state

    async def send_message_stream(self, owner_id: int, session_id: str, message: str):
        record_title, memory = self._load_agent_memory(owner_id, session_id)
        self._append_message(memory, "user", message)
        async for event in self._stream_session(session_id, owner_id, record_title, memory, is_new_session=False):
            yield event

    def update_draft(self, db: Session, owner_id: int, session_id: str, draft_content: str, title: str = "") -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        self._sanitize_memory_role_cards(memory)

        resolved_title = title.strip() or record.title
        memory["pending_plan"] = None
        memory["role_scripts"] = []
        memory["role_script_prompt_blueprints"] = []
        memory["draft_content"] = draft_content.strip()
        memory.setdefault("summary", {})["title"] = resolved_title
        memory["summary"]["focus"] = "手动编辑当前稿件"
        memory["summary"]["revision_count"] = int(memory["summary"].get("revision_count", 0)) + 1
        self._append_memory_node(
            memory,
            "manual_edit",
            "手动编辑成稿",
            memory["draft_content"][:120],
            tags=["draft", "manual"],
        )
        self._record_co_creation_sync(memory, "draft_content", memory["draft_content"])
        self._refresh_agent_memory_layers(memory)
        assistant_message = self._append_message(
            memory,
            "assistant",
            "已保存你手动修改后的当前稿件。你可以继续对话细化，也可以再次直接编辑。",
        )
        self._refresh_workspace_blueprint(memory, resolved_title)
        self._bind_snapshot_to_message(memory, resolved_title, assistant_message, "manual_edit")

        record.title = resolved_title
        record.latest_segment = memory["draft_content"]
        record.latest_branches = memory.get("suggestions", [])
        record.world_config = memory.get("summary", {})
        db.add(record)
        self.memory_service.save(record, memory)
        db.commit()
        db.refresh(record)
        return self._build_state(record, memory)

    async def update_config(self, db: Session, owner_id: int, session_id: str, config: dict) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        self._sanitize_memory_role_cards(memory)

        template_config = self._get_template_config(memory)
        onboarding = memory.get("onboarding", self._empty_onboarding_state(str(template_config.get("template_key") or DEFAULT_TEMPLATE_KEY)))
        answers = onboarding.setdefault("answers", {})
        previous_answers = copy.deepcopy(answers)
        llm_profile_changed = self._update_llm_profile(memory, config)
        draft_mode_changed = self._update_draft_mode(memory, config)
        role_script_length_mode_changed = self._update_role_script_length_mode(memory, config)
        co_creation_changed = self._update_co_creation_mode(memory, config)
        memory["pending_plan"] = None

        mapping = {
            "genre": "genre",
            "era": "era",
            "players": "players",
            "conflict": "conflict",
            "protagonist": "protagonist",
            "goal": "goal",
        }
        for payload_key, answer_key in mapping.items():
            value = (config.get(payload_key) or "").strip()
            if value:
                validation_error = self._validate_onboarding_answer(answer_key, value)
                if validation_error:
                    raise HTTPException(status_code=400, detail=validation_error)
                answers[answer_key] = self._normalize_onboarding_answer(answer_key, value)

        title = (config.get("title") or "").strip()
        if title:
            record.title = title

        self._sync_onboarding_state(onboarding, self._get_template_config(memory))
        memory["onboarding"] = onboarding
        current_facts = memory.get("facts", [])
        memory["facts"] = self._merge_text_items(self._build_onboarding_facts(answers), current_facts)
        memory["summary"] = self._build_onboarding_summary(record.title, onboarding)
        memory["role_scripts"] = []
        memory["role_script_prompt_blueprints"] = []
        self._sync_co_creation_state(memory.setdefault("co_creation", self._empty_co_creation_state()), self._get_template_config(memory))
        self._lock_co_creation_foundation(memory)
        self._append_memory_node(
            memory,
            "config_update",
            "参数更新",
            "；".join(self._build_onboarding_facts(answers)) or f"手动更新了配置参数。当前模式：{memory.get('draft_mode', 'complete')}",
            tags=["config", memory.get("draft_mode", "complete"), *sorted(answers.keys())],
        )
        self._refresh_agent_memory_layers(memory)
        answers_changed = any(previous_answers.get(key) != answers.get(key) for key in mapping.values())
        if onboarding.get("completed") and answers_changed and not self._co_creation_enabled(memory):
            generation_memory, memory_stats = self.memory_service.prepare_agent_memory(memory)
            payload, provider = await self.llm_service.generate_agent_turn(generation_memory)
            payload["memory_stats"] = memory_stats
            payload["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
            await self._attach_generated_role_scripts_to_payload(record.title, memory, payload, provider)
            memory, resolved_title = self._apply_payload_to_memory(
                record.title,
                memory,
                payload,
                provider,
                append_assistant_message=False,
            )
            record.title = title or resolved_title
            memory.setdefault("summary", {})["title"] = record.title
            self._refresh_workspace_blueprint(memory, record.title)
            self._capture_revision_snapshot(memory, record.title, "config_regeneration")
        else:
            if not onboarding.get("completed"):
                memory["suggestions"] = self._build_question_suggestions(onboarding, self._get_template_config(memory))
            elif self._co_creation_enabled(memory):
                memory["suggestions"] = self._build_co_creation_suggestions(memory)
            else:
                memory["suggestions"] = memory.get("suggestions", [])
            self._refresh_workspace_blueprint(memory, record.title)
            if onboarding.get("completed") and (draft_mode_changed or role_script_length_mode_changed or co_creation_changed):
                memory.setdefault("summary", {})["focus"] = "完整成稿模式" if memory.get("draft_mode") == "complete" else "迭代模式"
                if self._co_creation_enabled(memory):
                    memory["summary"]["focus"] = f"共创进行中：{memory.get('co_creation', {}).get('current_stage_title', '细节共创')}"
                blueprint = self._build_workspace_blueprint(memory, record.title)
                memory["suggestions"] = self._build_contextual_suggestions(memory, {"suggestions": memory.get("suggestions", []), "summary": memory.get("summary", {}), "draft_content": memory.get("draft_content", ""), "role_cards": memory.get("role_cards", []), "logic_checks": memory.get("logic_checks", [])}, blueprint)

        record.title = memory.get("summary", {}).get("title", record.title)
        record.world_config = memory.get("summary", {})
        record.latest_branches = memory.get("suggestions", [])
        record.latest_segment = memory.get("draft_content", "")
        db.add(record)
        self.memory_service.save(record, memory)
        db.commit()
        db.refresh(record)
        return self._build_state(record, memory)

    def update_role_cards(self, db: Session, owner_id: int, session_id: str, role_cards: list[dict]) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")

        normalized_cards = []
        for item in role_cards:
            if hasattr(item, "model_dump"):
                normalized_cards.append(item.model_dump())
            else:
                normalized_cards.append(dict(item))
        normalized_cards = self.llm_service.sanitize_role_cards(
            normalized_cards,
            genre=self._memory_role_card_genre(memory),
            conflict=self._memory_role_card_conflict(memory),
        )

        memory["pending_plan"] = None
        memory["role_cards"] = normalized_cards
        memory["role_scripts"] = []
        memory["role_script_prompt_blueprints"] = []
        memory["draft_content"] = self.llm_service.sync_draft_role_card_section(
            memory.get("draft_content", ""),
            normalized_cards,
        )
        memory.setdefault("summary", {})["focus"] = "手动编辑角色卡"
        memory["summary"]["revision_count"] = int(memory["summary"].get("revision_count", 0)) + 1
        self._append_memory_node(
            memory,
            "role_cards",
            "角色卡更新",
            f"已更新 {len(normalized_cards)} 张角色卡。",
            tags=["roles"],
        )
        self._record_co_creation_sync(memory, "role_cards", f"已同步 {len(normalized_cards)} 张角色卡修改")
        self._refresh_agent_memory_layers(memory)
        assistant_message = self._append_message(
            memory,
            "assistant",
            "已保存角色卡与人设标签体系。后续生成会继续参考这些人设约束。",
        )
        self._refresh_workspace_blueprint(memory, record.title)
        self._bind_snapshot_to_message(memory, record.title, assistant_message, "role_card_edit")

        record.world_config = memory.get("summary", {})
        record.latest_segment = memory["draft_content"]
        db.add(record)
        self.memory_service.save(record, memory)
        db.commit()
        db.refresh(record)
        return self._build_state(record, memory)

    def update_role_scripts(self, db: Session, owner_id: int, session_id: str, role_scripts: list[dict]) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        self._sanitize_memory_role_cards(memory)

        role_cards = memory.get("role_cards", [])
        role_id_map = {str(item.get("id") or ""): item for item in role_cards}
        normalized_scripts: list[dict[str, Any]] = []
        for item in role_scripts:
            payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            script_id = str(payload.get("id") or "").strip()
            role_name = str(payload.get("role_name") or "").strip()
            content = str(payload.get("content") or "").strip()
            if not script_id or not content:
                continue
            card = role_id_map.get(script_id, {})
            normalized_scripts.append(
                {
                    "id": script_id,
                    "role_name": role_name or str(card.get("name") or "").strip(),
                    "title": str(payload.get("title") or f"{role_name or card.get('name') or '角色'} 角色剧本").strip(),
                    "cover_age": str(payload.get("cover_age") or "").strip(),
                    "cover_identity": str(payload.get("cover_identity") or card.get("public_identity") or "").strip(),
                    "cover_tagline": str(payload.get("cover_tagline") or "").strip(),
                    "source": "manual",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "content": content,
                }
            )

        memory["pending_plan"] = None
        memory["role_scripts"] = normalized_scripts
        memory.setdefault("summary", {})["focus"] = "手动编辑角色剧本"
        memory["summary"]["revision_count"] = int(memory["summary"].get("revision_count", 0)) + 1
        self._append_memory_node(
            memory,
            "role_scripts",
            "角色剧本更新",
            f"已更新 {len(normalized_scripts)} 份角色剧本正文。",
            tags=["role_scripts", "manual"],
        )
        self._refresh_agent_memory_layers(memory)
        assistant_message = self._append_message(
            memory,
            "assistant",
            "已保存当前角色剧本正文。刷新页面后仍会保留这次修改。",
        )
        self._refresh_workspace_blueprint(memory, record.title)
        self._bind_snapshot_to_message(memory, record.title, assistant_message, "role_script_edit")

        record.world_config = memory.get("summary", {})
        record.latest_segment = memory.get("draft_content", "")
        db.add(record)
        self.memory_service.save(record, memory)
        db.commit()
        db.refresh(record)
        return self._build_state(record, memory)

    def update_behavior_records(self, db: Session, owner_id: int, session_id: str, behavior_records: list[dict]) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        self._sanitize_memory_role_cards(memory)

        normalized_records = []
        for item in behavior_records:
            payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            normalized_records.append(
                {
                    "id": str(payload.get("id") or uuid4()),
                    "timestamp": str(payload.get("timestamp", "")).strip(),
                    "role_name": str(payload.get("role_name", "")).strip(),
                    "action_type": str(payload.get("action_type", "")).strip(),
                    "key_output": str(payload.get("key_output", "")).strip(),
                    "dm_note": str(payload.get("dm_note", "")).strip(),
                }
            )

        memory["pending_plan"] = None
        memory["behavior_records"] = normalized_records
        memory.setdefault("summary", {})["focus"] = "记录玩家行为"
        memory["summary"]["revision_count"] = int(memory["summary"].get("revision_count", 0)) + 1
        self._append_memory_node(
            memory,
            "behavior_records",
            "玩家行为记录更新",
            f"已记录 {len(normalized_records)} 条玩家行为。",
            tags=["review", "behavior"],
        )
        self._record_co_creation_sync(memory, "behavior_records", f"已同步 {len(normalized_records)} 条行为记录")
        self._refresh_agent_memory_layers(memory)
        assistant_message = self._append_message(
            memory,
            "assistant",
            "已保存玩家行为记录，复盘检查与导出内容会自动同步这些记录。",
        )
        self._refresh_workspace_blueprint(memory, record.title)
        self._bind_snapshot_to_message(memory, record.title, assistant_message, "behavior_record_edit")

        record.world_config = memory.get("summary", {})
        record.latest_segment = memory.get("draft_content", "")
        db.add(record)
        self.memory_service.save(record, memory)
        db.commit()
        db.refresh(record)
        return self._build_state(record, memory)

    def update_facts(self, db: Session, owner_id: int, session_id: str, facts: list[str]) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        self._sanitize_memory_role_cards(memory)

        normalized_facts = self._merge_text_items(facts)
        memory["pending_plan"] = None
        memory["facts"] = normalized_facts
        memory.setdefault("summary", {})["focus"] = "手动编辑设定"
        memory["summary"]["revision_count"] = int(memory["summary"].get("revision_count", 0)) + 1
        self._append_memory_node(
            memory,
            "facts",
            "设定更新",
            "；".join(normalized_facts[:6]),
            tags=["facts", "manual"],
        )
        self._record_co_creation_sync(memory, "facts", "；".join(normalized_facts[:6]))
        self._refresh_agent_memory_layers(memory)
        assistant_message = self._append_message(
            memory,
            "assistant",
            "已同步设定区修改，后续对话会严格参考这些已确认事实。",
        )
        self._refresh_workspace_blueprint(memory, record.title)
        self._bind_snapshot_to_message(memory, record.title, assistant_message, "facts_edit")

        record.world_config = memory.get("summary", {})
        record.latest_segment = memory.get("draft_content", "")
        db.add(record)
        self.memory_service.save(record, memory)
        db.commit()
        db.refresh(record)
        return self._build_state(record, memory)

    def update_role_script_prompt_blueprints(self, db: Session, owner_id: int, session_id: str, blueprints: list[dict]) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        self._sanitize_memory_role_cards(memory)

        role_cards = memory.get("role_cards", [])
        role_card_ids = {str(item.get("id") or "") for item in role_cards if str(item.get("id") or "").strip()}
        role_name_map = {str(item.get("id") or ""): str(item.get("name") or "").strip() for item in role_cards}
        normalized_blueprints: list[dict[str, Any]] = []
        for item in blueprints:
            payload = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            blueprint_id = str(payload.get("id") or "").strip()
            if not blueprint_id or (role_card_ids and blueprint_id not in role_card_ids):
                continue
            normalized_blueprints.append(
                {
                    "id": blueprint_id,
                    "role_name": str(payload.get("role_name") or role_name_map.get(blueprint_id) or "").strip(),
                    "user_requirement": str(payload.get("user_requirement") or "").strip(),
                    "objective": str(payload.get("objective") or "").strip(),
                    "style": str(payload.get("style") or "").strip(),
                    "must_keep": [str(entry).strip() for entry in payload.get("must_keep", []) if str(entry).strip()],
                    "focus": [str(entry).strip() for entry in payload.get("focus", []) if str(entry).strip()],
                    "negative_constraints": [str(entry).strip() for entry in payload.get("negative_constraints", []) if str(entry).strip()],
                    "source": "manual",
                }
            )

        memory["pending_plan"] = None
        memory["role_script_prompt_blueprints"] = normalized_blueprints
        memory.setdefault("summary", {})["focus"] = "手动编辑角色剧本提示词"
        memory["summary"]["revision_count"] = int(memory["summary"].get("revision_count", 0)) + 1
        self._append_memory_node(
            memory,
            "role_script_prompt_blueprints",
            "角色剧本提示词更新",
            f"已更新 {len(normalized_blueprints)} 份角色剧本提示词蓝图。",
            tags=["role_scripts", "prompt"],
        )
        self._refresh_agent_memory_layers(memory)
        assistant_message = self._append_message(
            memory,
            "assistant",
            "已保存角色剧本提示词蓝图。下次重新生成角色剧本时，会优先使用你编辑后的版本。",
        )
        self._refresh_workspace_blueprint(memory, record.title)
        self._bind_snapshot_to_message(memory, record.title, assistant_message, "role_script_prompt_edit")

        record.world_config = memory.get("summary", {})
        record.latest_segment = memory.get("draft_content", "")
        db.add(record)
        self.memory_service.save(record, memory)
        db.commit()
        db.refresh(record)
        return self._build_state(record, memory)

    async def import_concept_file(
        self,
        db: Session,
        owner_id: int,
        session_id: str,
        filename: str,
        content_type: str,
        file_bytes: bytes,
    ) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        self._sanitize_memory_role_cards(memory)

        try:
            imported_text = self.concept_import_service.extract_text(filename, content_type, file_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        analysis = self.concept_import_service.analyze_text(imported_text, filename)
        return await self.apply_imported_concept(
            db,
            owner_id,
            session_id,
            analysis,
            ["title", "genre", "era", "players", "conflict", "protagonist", "goal", "role_cards", "facts"],
            filename=filename,
            imported_text=imported_text,
        )

    async def preview_concept_file(
        self,
        db: Session,
        owner_id: int,
        session_id: str,
        filename: str,
        content_type: str,
        file_bytes: bytes,
    ) -> dict[str, Any]:
        return await self.preview_concept_files(
            db,
            owner_id,
            session_id,
            [(filename, content_type, file_bytes)],
        )

    async def preview_concept_files(
        self,
        db: Session,
        owner_id: int,
        session_id: str,
        uploads: list[tuple[str, str, bytes]],
    ) -> dict[str, Any]:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        if not uploads:
            raise HTTPException(status_code=400, detail="请至少上传一个构想文件。")

        analyses: list[dict[str, Any]] = []
        filenames: list[str] = []
        sources: list[dict[str, Any]] = []
        for filename, content_type, file_bytes in uploads:
            try:
                imported_text = self.concept_import_service.extract_text(filename, content_type, file_bytes)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{filename}：{exc}") from exc
            analysis = self.concept_import_service.analyze_text(imported_text, filename)
            analyses.append(analysis)
            filenames.append(filename)
            sources.append(
                {
                    "filename": filename,
                    "category": self.concept_import_service.classify_filename(filename),
                    "analysis": analysis,
                }
            )

        merged = self.concept_import_service.merge_analyses(analyses, filenames)
        label = filenames[0] if len(filenames) == 1 else f"{filenames[0]} 等 {len(filenames)} 个文件"
        return {"filename": label, "filenames": filenames, "analysis": merged, "sources": sources}

    async def apply_imported_concept(
        self,
        db: Session,
        owner_id: int,
        session_id: str,
        analysis: dict[str, Any],
        apply_fields: list[str],
        replace_role_names: list[str] | None = None,
        selected_facts: list[str] | None = None,
        *,
        filename: str = "导入构想",
        imported_text: str = "",
    ) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        self._sanitize_memory_role_cards(memory)

        selected = set(apply_fields or [])
        replace_names = {str(item).strip() for item in (replace_role_names or []) if str(item).strip()}
        selected_fact_set = None if selected_facts is None else {str(item).strip() for item in selected_facts if str(item).strip()}
        if not selected:
            raise HTTPException(status_code=400, detail="请至少选择一项导入内容。")

        onboarding = memory.get("onboarding", self._empty_onboarding_state())
        answers = onboarding.setdefault("answers", {})

        mapping = {
            "genre": "genre",
            "era": "era",
            "players": "players",
            "conflict": "conflict",
            "protagonist": "protagonist",
            "goal": "goal",
        }
        recognized_fields: list[str] = []
        for source_key, answer_key in mapping.items():
            if source_key not in selected:
                continue
            candidate = str(analysis.get(source_key) or "").strip()
            if not candidate or not self._can_fill_imported_answer(answers.get(answer_key)):
                continue
            validation_error = self._validate_onboarding_answer(answer_key, candidate)
            if validation_error:
                continue
            answers[answer_key] = self._normalize_onboarding_answer(answer_key, candidate)
            recognized_fields.append(f"{answer_key}={answers[answer_key]}")

        imported_title = str(analysis.get("title") or "").strip()
        if "title" in selected and imported_title and self._can_fill_imported_title(record.title):
            record.title = imported_title[:200]
            recognized_fields.append(f"title={record.title}")

        self._sync_onboarding_state(onboarding, self._get_template_config(memory))
        memory["onboarding"] = onboarding
        imported_facts = []
        if "facts" in selected:
            for item in analysis.get("facts", []):
                text = str(item).strip()
                if not text:
                    continue
                if selected_fact_set is not None and text not in selected_fact_set:
                    continue
                imported_facts.append(text)
        base_facts = self._build_onboarding_facts(answers)
        memory["facts"] = self._merge_text_items(base_facts, memory.get("facts", []), imported_facts)
        if recognized_fields or imported_facts:
            memory["role_scripts"] = []
            memory["role_script_prompt_blueprints"] = []

        existing_roles = memory.get("role_cards", [])
        imported_roles = analysis.get("role_cards", []) if "role_cards" in selected else []
        merged_roles = self._merge_imported_role_cards(existing_roles, imported_roles, replace_names)
        if merged_roles != existing_roles:
            memory["role_cards"] = self.llm_service.sanitize_role_cards(
                merged_roles,
                genre=self._memory_role_card_genre(memory),
                conflict=self._memory_role_card_conflict(memory),
            )
            memory["role_scripts"] = []
            memory["role_script_prompt_blueprints"] = []
            recognized_fields.append(f"role_cards={len(memory['role_cards'])}")

        previous_summary = memory.get("summary", {})
        memory["summary"] = {
            **self._build_onboarding_summary(record.title, onboarding),
            "title": record.title,
            "genre": previous_summary.get("genre") or answers.get("genre") or "待确认",
            "era": previous_summary.get("era") or answers.get("era") or "待确认",
            "focus": "导入构想文件",
            "revision_count": int(previous_summary.get("revision_count", 0)) + 1,
        }
        imported_summary = str(analysis.get("summary") or "").strip()
        self._append_memory_node(
            memory,
            "concept_import",
            "导入构想文件",
            f"{filename}：{imported_summary or imported_text[:180]}",
            tags=["import", Path(filename).suffix.lower().lstrip(".") or "text"],
        )
        self._refresh_agent_memory_layers(memory)
        assistant_message = self._append_message(
            memory,
            "assistant",
            self._build_concept_import_reply(filename, analysis, recognized_fields),
        )
        self._refresh_workspace_blueprint(memory, record.title)
        self._bind_snapshot_to_message(memory, record.title, assistant_message, "concept_import")

        record.world_config = memory.get("summary", {})
        record.latest_segment = memory.get("draft_content", "")
        record.latest_branches = memory.get("suggestions", [])
        db.add(record)
        self.memory_service.save(record, memory)
        db.commit()
        db.refresh(record)
        return self._build_state(record, memory)

    def control_onboarding(self, db: Session, owner_id: int, session_id: str, action: str, question_id: str) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
        self._sanitize_memory_role_cards(memory)

        onboarding = memory.get("onboarding", self._empty_onboarding_state())
        if not onboarding.get("enabled", True):
            raise HTTPException(status_code=400, detail="当前会话未启用引导问答。")
        template_config = self._get_template_config(memory)

        if action == "jump":
            target_id = question_id or onboarding.get("current_question_id", "")
            self._jump_to_question(onboarding, target_id, template_config)
            memory["summary"] = self._build_onboarding_summary(record.title, onboarding)
            memory["suggestions"] = self._build_question_suggestions(onboarding, template_config)
            memory["facts"] = self._build_onboarding_facts(onboarding.get("answers", {}))
            assistant_message = self._append_message(
                memory,
                "assistant",
                f"已切换到 {onboarding.get('current_question') or '指定步骤'}。你可以重新回答这一项，后续步骤会按新的设定继续生成。",
            )
            self._append_memory_node(memory, "onboarding_control", "回退步骤", target_id, tags=["jump", target_id])
        elif action == "skip":
            target_id = question_id or onboarding.get("current_question_id", "")
            self._skip_question(onboarding, target_id, template_config)
            memory["summary"] = self._build_onboarding_summary(record.title, onboarding)
            memory["suggestions"] = self._build_question_suggestions(onboarding, template_config)
            memory["facts"] = self._build_onboarding_facts(onboarding.get("answers", {}))
            assistant_message = self._append_message(
                memory,
                "assistant",
                "这一项我先标记为“待补充”，你后续可以随时回到这一步再改。",
            )
            self._append_memory_node(memory, "onboarding_control", "跳过步骤", target_id, tags=["skip", target_id])
        elif action == "reset":
            onboarding = self._empty_onboarding_state(str(template_config.get("template_key") or DEFAULT_TEMPLATE_KEY))
            memory["onboarding"] = onboarding
            memory["facts"] = []
            memory["summary"] = self._build_onboarding_summary("待命名剧本", onboarding)
            memory["suggestions"] = self._build_question_suggestions(onboarding, template_config)
            memory["draft_content"] = ""
            memory["role_cards"] = []
            memory["role_scripts"] = []
            memory["role_script_prompt_blueprints"] = []
            memory["task_queue"] = []
            memory["structure_plan"] = []
            memory["dm_script"] = []
            memory["output_outline"] = []
            assistant_message = self._append_message(
                memory,
                "assistant",
                "已重置顶部核心参数。你可以重新填写类型、人数、冲突与目标，然后再生成新稿。",
            )
            memory["memory_nodes"] = []
            self._append_memory_node(memory, "onboarding_control", "重置参数", "已清空现有设定并重新开始。", tags=["reset"])
        else:
            raise HTTPException(status_code=400, detail="不支持的引导控制动作。")

        self._refresh_agent_memory_layers(memory)
        self._refresh_workspace_blueprint(memory, record.title)
        self._bind_snapshot_to_message(memory, record.title, assistant_message, "onboarding_control")
        record.title = memory.get("summary", {}).get("title", record.title)
        record.latest_branches = memory.get("suggestions", [])
        record.world_config = memory.get("summary", {})
        db.add(record)
        self.memory_service.save(record, memory)
        db.commit()
        db.refresh(record)
        return self._build_state(record, memory)

    def delete_session(self, db: Session, owner_id: int, session_id: str) -> dict[str, str]:
        record = self._get_record(db, owner_id, session_id)
        db.delete(record)
        db.commit()
        self.memory_service.delete(session_id)
        return {"status": "deleted"}

    def set_session_hidden(self, db: Session, owner_id: int, session_id: str, hidden: bool) -> dict[str, str]:
        record = self._get_record(db, owner_id, session_id)
        record.is_hidden = bool(hidden)
        db.add(record)
        db.commit()
        return {"status": "hidden" if hidden else "visible"}

    def rollback_to_revision(self, db: Session, owner_id: int, session_id: str, revision_id: str) -> AgentStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        if memory.get("mode") != "agent":
            raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")

        snapshot = self._find_revision_snapshot(memory, revision_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="未找到可回滚的历史版本。")

        self._restore_snapshot_to_memory(memory, snapshot)
        self._sanitize_memory_role_cards(memory)
        self._append_memory_node(
            memory,
            "rollback",
            "版本回滚",
            snapshot.get("revision_id", ""),
            tags=["rollback"],
        )
        self._refresh_agent_memory_layers(memory)
        record.title = snapshot.get("title") or record.title
        record.latest_segment = memory.get("draft_content", "")
        record.latest_branches = memory.get("suggestions", [])
        record.world_config = memory.get("summary", {})
        db.add(record)
        self.memory_service.save(record, memory)
        db.commit()
        db.refresh(record)
        return self._build_state(record, memory)

    async def _generate_turn(self, db: Session, record: ScriptSession, memory: dict) -> AgentStateResponse:
        self._refresh_agent_memory_layers(memory)
        generation_memory, memory_stats = self.memory_service.prepare_agent_memory(memory)
        payload, provider = await self.llm_service.generate_agent_turn(generation_memory)
        payload["memory_stats"] = memory_stats
        payload["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
        await self._attach_generated_role_scripts_to_payload(record.title, memory, payload, provider)
        return self._apply_payload(db, record, memory, payload, provider)

    async def _generate_co_creation_stage_output(
        self,
        db: Session,
        record: ScriptSession,
        memory: dict,
        *,
        force_full_script: bool = False,
    ) -> AgentStateResponse:
        co_creation = memory.setdefault("co_creation", self._empty_co_creation_state(str(memory.get("template_key") or DEFAULT_TEMPLATE_KEY)))
        stage = self._get_current_co_creation_stage(memory) or {}
        stage_id = str(stage.get("id") or "").strip()

        if force_full_script:
            memory["active_generation_request"] = "请直接生成完整剧本，跳过剩余共创阶段，输出可继续编辑的完整剧本稿、人物、分幕和交付骨架。"
            co_creation["current_stage_id"] = "detail_iteration"
            self._sync_co_creation_state(co_creation, self._get_template_config(memory))
            stage_id = "detail_iteration"
        elif stage_id == "worldview":
            self._apply_co_creation_worldview(memory, "已确认世界观，生成世界观框架")
            self._advance_co_creation_stage(memory)
        elif stage_id == "roles":
            self._apply_co_creation_roles(memory, "已确认角色方向，生成角色卡初稿")
            self._advance_co_creation_stage(memory)
        elif stage_id == "core_plot":
            self._apply_co_creation_core_plot(memory, "已确认剧情核心，生成分幕剧情框架")
            self._advance_co_creation_stage(memory)
        else:
            memory["active_generation_request"] = "请基于当前已锁定设定生成完整剧本内容，并输出最终交付结构。"

        if stage_id in {"detail_iteration", "core_plot"} or force_full_script:
            return await self._generate_turn(db, record, memory)

        template_config = self._get_template_config(memory)
        co_creation["sync_notice"] = str((template_config.get("co_creation_config") or {}).get("panel_sync_notice") or "").strip()
        self._lock_co_creation_foundation(memory)
        payload = self._build_co_creation_payload(memory, intro=False)
        return self._apply_payload(db, record, memory, payload, "guide")

    async def _stream_session(self, session_id: str, owner_id: int, record_title: str, memory: dict, is_new_session: bool):
        try:
            latest_user_input = memory.get("messages", [{}])[-1].get("content", "")
            route = {"response_mode": "generate"}
            if not memory.get("onboarding", {}).get("completed"):
                if not is_new_session:
                    if self._is_clarification_request(memory, latest_user_input):
                        payload = self._build_clarification_payload(memory)
                        provider = "guide"
                        yield self._sse_event("meta", {"session_id": session_id, "provider": provider})
                        async for event in self._yield_stream_chunks(payload):
                            yield event
                        final_state = self._persist_existing_session(session_id, owner_id, record_title, memory, payload, provider)
                        yield self._sse_event("complete", final_state.model_dump())
                        return

                    validation_error = self._validate_onboarding_answer(
                        memory.get("onboarding", {}).get("current_question_id", ""),
                        latest_user_input,
                    )
                    if validation_error:
                        payload = self._build_validation_payload(memory, validation_error)
                        provider = "guide"
                        yield self._sse_event("meta", {"session_id": session_id, "provider": provider})
                        async for event in self._yield_stream_chunks(payload):
                            yield event
                        final_state = self._persist_existing_session(session_id, owner_id, record_title, memory, payload, provider)
                        yield self._sse_event("complete", final_state.model_dump())
                        return

                    self._record_onboarding_answer(memory, latest_user_input)

                if not memory["onboarding"]["completed"]:
                    payload = self._build_onboarding_payload(memory, latest_user_input, is_initial=is_new_session)
                    provider = "guide"
                else:
                    if self._co_creation_enabled(memory):
                        payload, provider = self._process_co_creation_turn(memory, latest_user_input, intro=True)
                    else:
                        self._refresh_agent_memory_layers(memory)
                        generation_memory, memory_stats = self.memory_service.prepare_agent_memory(memory)
                        payload, provider = await self.llm_service.generate_agent_turn(generation_memory)
                        payload["memory_stats"] = memory_stats
                        payload["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
                        await self._attach_generated_role_scripts_to_payload(record_title, memory, payload, provider)
            else:
                if (local_payload := self._maybe_build_local_edit_payload(memory, latest_user_input)) is not None:
                    payload = local_payload
                    provider = "local"
                    yield self._sse_event("meta", {"session_id": session_id, "provider": provider})
                    async for event in self._yield_stream_chunks(payload):
                        yield event
                    final_state = self._persist_existing_session(session_id, owner_id, record_title, memory, payload, provider)
                    yield self._sse_event("complete", final_state.model_dump())
                    return
                if (title_payload := self._maybe_build_local_title_refresh_payload(memory, latest_user_input)) is not None:
                    payload = title_payload
                    provider = "local"
                    yield self._sse_event("meta", {"session_id": session_id, "provider": provider})
                    async for event in self._yield_stream_chunks(payload):
                        yield event
                    final_state = self._persist_existing_help_session(session_id, owner_id, record_title, memory, payload, provider)
                    yield self._sse_event("complete", final_state.model_dump())
                    return
                if self._co_creation_enabled(memory):
                    if self._is_co_creation_finish_request(latest_user_input):
                        memory["active_generation_request"] = "请直接生成完整剧本，跳过剩余共创阶段，输出可继续编辑的完整剧本稿、人物、分幕和交付骨架。"
                        memory.setdefault("co_creation", {})["current_stage_id"] = "detail_iteration"
                        self._sync_co_creation_state(memory["co_creation"], self._get_template_config(memory))
                        self._refresh_agent_memory_layers(memory)
                        generation_memory, memory_stats = self.memory_service.prepare_agent_memory(memory)
                        payload, provider = await self.llm_service.generate_agent_turn(generation_memory)
                        payload["memory_stats"] = memory_stats
                        payload["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
                        await self._attach_generated_role_scripts_to_payload(record_title, memory, payload, provider)
                        yield self._sse_event("meta", {"session_id": session_id, "provider": provider})
                        async for event in self._yield_stream_chunks(payload):
                            yield event
                        final_state = self._persist_existing_session(session_id, owner_id, record_title, memory, payload, provider)
                        yield self._sse_event("complete", final_state.model_dump())
                        return
                    current_stage = self._get_current_co_creation_stage(memory) or {}
                    if str(current_stage.get("id") or "").strip() == "detail_iteration" and latest_user_input.strip():
                        memory["active_generation_request"] = latest_user_input.strip()
                        self._refresh_agent_memory_layers(memory)
                        generation_memory, memory_stats = self.memory_service.prepare_agent_memory(memory)
                        payload, provider = await self.llm_service.generate_agent_turn(generation_memory)
                        payload["memory_stats"] = memory_stats
                        payload["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
                        await self._attach_generated_role_scripts_to_payload(record_title, memory, payload, provider)
                        yield self._sse_event("meta", {"session_id": session_id, "provider": provider})
                        async for event in self._yield_stream_chunks(payload):
                            yield event
                        final_state = self._persist_existing_session(session_id, owner_id, record_title, memory, payload, provider)
                        yield self._sse_event("complete", final_state.model_dump())
                        return
                    payload, provider = self._process_co_creation_turn(memory, latest_user_input, intro=False)
                    if provider == "finish":
                        memory["active_generation_request"] = "请直接生成完整剧本，基于当前已锁定设定输出完整剧本稿、人物、分幕和交付骨架。"
                        self._refresh_agent_memory_layers(memory)
                        generation_memory, memory_stats = self.memory_service.prepare_agent_memory(memory)
                        payload, provider = await self.llm_service.generate_agent_turn(generation_memory)
                        payload["memory_stats"] = memory_stats
                        payload["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
                        await self._attach_generated_role_scripts_to_payload(record_title, memory, payload, provider)
                    if self._is_co_creation_generate_request(memory, latest_user_input):
                        generation_memory, memory_stats = self.memory_service.prepare_agent_memory(memory)
                        payload, provider = await self.llm_service.generate_agent_turn(generation_memory)
                        payload["memory_stats"] = memory_stats
                        payload["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
                        await self._attach_generated_role_scripts_to_payload(record_title, memory, payload, provider)
                    yield self._sse_event("meta", {"session_id": session_id, "provider": provider})
                    async for event in self._yield_stream_chunks(payload):
                        yield event
                    final_state = self._persist_existing_session(session_id, owner_id, record_title, memory, payload, provider)
                    yield self._sse_event("complete", final_state.model_dump())
                    return
                route = self._resolve_runtime_route(memory, latest_user_input)
                if not memory.get("draft_content", "").strip():
                    self._refresh_agent_memory_layers(memory)
                    generation_memory, memory_stats = self.memory_service.prepare_agent_memory(memory)
                    payload, provider = await self.llm_service.generate_agent_turn(generation_memory)
                    payload["memory_stats"] = memory_stats
                    payload["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
                    await self._attach_generated_role_scripts_to_payload(record_title, memory, payload, provider)
                elif memory.get("pending_plan"):
                    if self._is_plan_confirmation(latest_user_input):
                        memory["active_generation_request"] = memory["pending_plan"]["request"]
                        memory["pending_plan"] = None
                        self._refresh_agent_memory_layers(memory)
                        generation_memory, memory_stats = self.memory_service.prepare_agent_memory(memory)
                        payload, provider = await self.llm_service.generate_agent_turn(generation_memory)
                        payload["memory_stats"] = memory_stats
                        payload["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
                        await self._attach_generated_role_scripts_to_payload(record_title, memory, payload, provider)
                    elif self._is_plan_cancellation(latest_user_input):
                        payload = self._build_plan_cancel_payload(memory)
                        provider = "guide"
                    else:
                        payload = self._build_pending_plan_payload(memory, latest_user_input, route)
                        provider = "guide"
                elif (local_payload := self._maybe_build_local_edit_payload(memory, latest_user_input)) is not None:
                    payload = local_payload
                    provider = "local"
                elif (title_payload := self._maybe_build_local_title_refresh_payload(memory, latest_user_input)) is not None:
                    payload = title_payload
                    provider = "local"
                elif route.get("response_mode") == "help":
                    payload = self._build_workspace_help_payload(memory, latest_user_input)
                    provider = "guide"
                elif route.get("response_mode") == "answer":
                    payload = self._build_analysis_payload(memory, latest_user_input, route)
                    provider = "guide"
                elif self._should_require_plan_confirmation(memory, latest_user_input, route):
                    payload = self._build_pending_plan_payload(memory, latest_user_input, route)
                    provider = "guide"
                else:
                    self._refresh_agent_memory_layers(memory)
                    generation_memory, memory_stats = self.memory_service.prepare_agent_memory(memory)
                    payload, provider = await self.llm_service.generate_agent_turn(generation_memory)
                    payload["memory_stats"] = memory_stats
                    payload["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
                    await self._attach_generated_role_scripts_to_payload(record_title, memory, payload, provider)

            yield self._sse_event("meta", {"session_id": session_id, "provider": provider})
            async for event in self._yield_stream_chunks(payload):
                yield event

            if is_new_session:
                if provider == "guide" and route.get("response_mode") in {"help", "answer"}:
                    final_state = self._persist_new_help_session(session_id, owner_id, record_title, memory, payload, provider)
                else:
                    final_state = self._persist_new_session(session_id, owner_id, record_title, memory, payload, provider)
            else:
                if provider == "guide" and route.get("response_mode") in {"help", "answer"}:
                    final_state = self._persist_existing_help_session(session_id, owner_id, record_title, memory, payload, provider)
                else:
                    final_state = self._persist_existing_session(session_id, owner_id, record_title, memory, payload, provider)
            yield self._sse_event("complete", final_state.model_dump())
        except HTTPException as exc:
            yield self._sse_event("error", {"detail": str(exc.detail), "status_code": exc.status_code})

    def _apply_payload(self, db: Session, record: ScriptSession, memory: dict, payload: dict, provider: str) -> AgentStateResponse:
        memory, resolved_title = self._apply_payload_to_memory(record.title, memory, payload, provider)
        record.title = resolved_title
        record.latest_segment = memory["draft_content"]
        record.latest_branches = memory["suggestions"]
        record.world_config = memory["summary"]
        db.add(record)
        self.memory_service.save(record, memory)
        return self._build_state(record, memory, provider)

    def _apply_help_payload(self, db: Session, record: ScriptSession, memory: dict, payload: dict, provider: str) -> AgentStateResponse:
        memory, resolved_title = self._apply_help_payload_to_memory(record.title, memory, payload, provider)
        record.title = resolved_title
        record.latest_segment = memory.get("draft_content", "")
        record.latest_branches = memory.get("suggestions", [])
        record.world_config = memory.get("summary", {})
        db.add(record)
        self.memory_service.save(record, memory)
        return self._build_state(record, memory, provider)

    def _build_state(self, record: ScriptSession, memory: dict, provider: str | None = None) -> AgentStateResponse:
        summary = memory.get("summary", {})
        llm_profile = memory.get("llm_profile", self.llm_service.build_llm_profile())
        host_manual = self.host_manual_service.build(record.title, memory)
        role_scripts = self.role_script_service.build(record.title, memory)
        return AgentStateResponse(
            session_id=record.id,
            title=record.title,
            messages=self._build_messages_for_state(memory),
            draft_content=memory.get("draft_content", ""),
            suggestions=memory.get("suggestions", []),
            facts=memory.get("facts", []),
            role_cards=memory.get("role_cards", []),
            role_scripts=role_scripts,
            behavior_records=memory.get("behavior_records", []),
            onboarding=memory.get("onboarding", self._empty_onboarding_state()),
            co_creation=memory.get("co_creation", self._empty_co_creation_state()),
            summary={
                "title": summary.get("title", record.title),
                "era": summary.get("era", "待确认"),
                "genre": summary.get("genre", "待确认"),
                "focus": summary.get("focus", "待补充"),
                "revision_count": int(summary.get("revision_count", 0)),
                "template_key": summary.get("template_key", memory.get("template_key", DEFAULT_TEMPLATE_KEY)),
                "template_label": summary.get("template_label", memory.get("template_label", "通用剧本模板")),
                "template_family": summary.get("template_family", memory.get("template_family", "general")),
            },
            intent=memory.get(
                "intent",
                {
                    "current_intent": "生成新剧本",
                    "next_action": "继续补全参数",
                    "progress_label": "待初始化",
                    "params": {},
                },
            ),
            task_queue=memory.get("task_queue", []),
            structure_plan=memory.get("structure_plan", []),
            dm_script=memory.get("dm_script", []),
            output_outline=memory.get("output_outline", []),
            logic_checks=memory.get("logic_checks", []),
            memory_summary=memory.get("memory_summary", []),
            memory_stats=memory.get(
                "memory_stats",
                {
                    "token_budget": settings.token_budget,
                    "estimated_tokens": 0,
                    "compression_applied": False,
                    "message_entries": len(memory.get("messages", [])),
                    "memory_nodes": len(memory.get("memory_nodes", [])),
                },
            ),
            pending_plan=memory.get("pending_plan"),
            host_manual=host_manual,
            provider=provider or memory.get("provider", "mock"),
            llm_provider=llm_profile.get("provider", "mock"),
            llm_model=llm_profile.get("model", self.llm_service.default_model_for_provider(llm_profile.get("provider", "mock"))),
            role_script_length_mode=str(memory.get("role_script_length_mode") or "normal"),
            template_config=memory.get("template_config"),
        )

    def _build_initial_memory(
        self,
        record_title: str,
        brief: str,
        llm_provider: str = "",
        llm_model: str = "",
        template_key: str = "",
        template_form_data: dict[str, str] | None = None,
    ) -> dict:
        resolved_template_key = self.template_config_service.resolve_template_key(template_key, brief)
        template_config = self.template_config_service.get_template(resolved_template_key)
        runtime_flags = self.template_runtime_service.build_runtime_flags(template_config)
        answers = self._sanitize_inferred_answers(self._infer_onboarding_answers(brief))
        answers = self._merge_template_defaults_into_answers(answers, template_config, template_form_data or {})
        onboarding = {
            "enabled": True,
            "completed": False,
            "progress": 0,
            "total_questions": len(self._get_question_flow(template_config)),
            "template_key": runtime_flags["template_key"],
            "current_question_id": "",
            "current_question": "",
            "current_placeholder": "",
            "answers": answers,
        }
        self._sync_onboarding_state(onboarding, template_config)
        co_creation = self._empty_co_creation_state(runtime_flags["template_key"])
        self._sync_co_creation_state(co_creation, template_config)
        memory = {
            "mode": "agent",
            "template_key": runtime_flags["template_key"],
            "template_label": runtime_flags["template_label"],
            "template_family": runtime_flags["template_family"],
            "template_config": template_config,
            "template_form_data": copy.deepcopy(template_form_data or {}),
            "template_runtime": runtime_flags,
            "draft_mode": "complete",
            "role_script_length_mode": "normal",
            "messages": [self._create_message("user", brief)],
            "draft_content": "",
            "suggestions": [],
            "pending_plan": None,
            "facts": self._build_onboarding_facts(answers),
            "role_cards": [],
            "role_scripts": [],
            "role_script_prompt_blueprints": [],
            "behavior_records": [],
            "onboarding": onboarding,
            "co_creation": co_creation,
            "summary": self._build_onboarding_summary(record_title, onboarding),
            "revision_history": [],
            "memory_nodes": [],
            "generation_rules": [],
            "logic_checks": [],
            "memory_summary": [],
            "llm_profile": self._resolve_llm_profile(llm_provider, llm_model),
            "memory_stats": {
                "token_budget": settings.token_budget,
                "estimated_tokens": 0,
                "compression_applied": False,
                "message_entries": 1,
                "memory_nodes": 0,
            },
        }
        self._append_memory_node(memory, "brief", "初始需求", brief, tags=["brief"])
        self._refresh_workspace_blueprint(memory, record_title)
        self._refresh_agent_memory_layers(memory)
        return memory

    def _resolve_llm_profile(self, provider: str = "", model: str = "") -> dict[str, str]:
        requested_provider = (provider or "").strip().lower()
        if requested_provider and requested_provider not in self.llm_service.SUPPORTED_PROVIDERS:
            raise HTTPException(status_code=400, detail="不支持的模型提供方。")
        return self.llm_service.build_llm_profile(requested_provider, model)

    def _get_template_config(self, memory: dict | None = None, template_key: str = "") -> dict[str, Any]:
        if memory and isinstance(memory.get("template_config"), dict):
            return copy.deepcopy(memory["template_config"])
        resolved_template_key = self.template_config_service.resolve_template_key(template_key)
        return self.template_config_service.get_template(resolved_template_key)

    def _get_question_flow(self, template_config: dict[str, Any]) -> list[dict[str, Any]]:
        guide_config = template_config.get("dialog_guide_config") or {}
        steps = guide_config.get("steps") or []
        return [dict(item) for item in steps if isinstance(item, dict) and item.get("id")]

    def _get_co_creation_stages(self, template_config: dict[str, Any]) -> list[dict[str, Any]]:
        co_creation_config = template_config.get("co_creation_config") or {}
        stages = co_creation_config.get("stages") or []
        return [dict(item) for item in stages if isinstance(item, dict) and item.get("id")]

    def _merge_template_defaults_into_answers(
        self,
        answers: dict[str, str],
        template_config: dict[str, Any],
        template_form_data: dict[str, str],
    ) -> dict[str, str]:
        merged = dict(answers)
        defaults = template_config.get("defaults") or {}
        for key, value in defaults.items():
            normalized = str(value or "").strip()
            if normalized:
                merged.setdefault(key, normalized)
        for key, value in (template_form_data or {}).items():
            normalized_key = str(key or "").strip()
            normalized_value = str(value or "").strip()
            if normalized_key and normalized_value:
                merged[normalized_key] = normalized_value
        return merged

    def _empty_co_creation_state(self, template_key: str = DEFAULT_TEMPLATE_KEY) -> dict[str, Any]:
        template_config = self.template_config_service.get_template(template_key)
        co_creation_config = template_config.get("co_creation_config") or {}
        return {
            "enabled": bool(co_creation_config.get("default_enabled", True)),
            "current_stage_id": "",
            "current_stage_title": "",
            "stage_index": 0,
            "total_stages": len(self._get_co_creation_stages(template_config)),
            "awaiting_user": True,
            "sync_notice": "",
            "locked_fields": [],
            "locked_values": {},
            "user_notes": {},
            "stage_options": [],
        }

    def _sync_co_creation_state(self, co_creation: dict[str, Any], template_config: dict[str, Any] | None = None) -> None:
        resolved_template = template_config or self.template_config_service.get_template(DEFAULT_TEMPLATE_KEY)
        stages = self._get_co_creation_stages(resolved_template)
        co_creation["total_stages"] = len(stages)
        if not stages:
            co_creation["current_stage_id"] = ""
            co_creation["current_stage_title"] = ""
            co_creation["stage_index"] = 0
            co_creation["stage_options"] = []
            return

        current_stage_id = str(co_creation.get("current_stage_id") or "").strip()
        stage_index = next((index for index, stage in enumerate(stages) if stage["id"] == current_stage_id), 0)
        current_stage = stages[stage_index]
        co_creation["current_stage_id"] = current_stage["id"]
        co_creation["current_stage_title"] = current_stage.get("title", current_stage["id"])
        co_creation["stage_index"] = stage_index + 1
        co_creation["stage_options"] = copy.deepcopy(current_stage.get("options", []))

    @staticmethod
    def _co_creation_enabled(memory: dict[str, Any]) -> bool:
        return bool(memory.get("co_creation", {}).get("enabled", True))

    def _update_llm_profile(self, memory: dict, config: dict) -> bool:
        provider_input = (config.get("llm_provider") or "").strip().lower()
        model_input = (config.get("llm_model") or "").strip()
        if not provider_input and not model_input:
            return False

        current_profile = copy.deepcopy(memory.get("llm_profile", self.llm_service.build_llm_profile()))
        if provider_input and provider_input not in self.llm_service.SUPPORTED_PROVIDERS:
            raise HTTPException(status_code=400, detail="不支持的模型提供方。")

        resolved_provider = provider_input or current_profile.get("provider") or self.llm_service.normalize_provider("")
        if model_input:
            resolved_model = model_input
        elif provider_input and provider_input != current_profile.get("provider"):
            resolved_model = self.llm_service.default_model_for_provider(resolved_provider)
        else:
            resolved_model = current_profile.get("model") or self.llm_service.default_model_for_provider(resolved_provider)

        next_profile = self.llm_service.build_llm_profile(resolved_provider, resolved_model)
        memory["llm_profile"] = next_profile
        if next_profile == current_profile:
            return False

        self._append_memory_node(
            memory,
            "llm_profile",
            "模型配置更新",
            f"provider={next_profile['provider']}；model={next_profile['model']}",
            tags=["llm", next_profile["provider"]],
        )
        return True

    @staticmethod
    def _update_draft_mode(memory: dict, config: dict) -> bool:
        mode_input = str(config.get("draft_mode") or "").strip().lower()
        if not mode_input:
            return False
        resolved_mode = "complete" if mode_input == "complete" else "iterative"
        if memory.get("draft_mode") == resolved_mode:
            return False
        memory["draft_mode"] = resolved_mode
        return True

    @staticmethod
    def _update_role_script_length_mode(memory: dict, config: dict) -> bool:
        mode_input = str(config.get("role_script_length_mode") or "").strip().lower()
        if not mode_input:
            return False
        resolved_mode = mode_input if mode_input in {"normal", "demo", "deep"} else "normal"
        if memory.get("role_script_length_mode") == resolved_mode:
            return False
        memory["role_script_length_mode"] = resolved_mode
        return True

    def _update_co_creation_mode(self, memory: dict, config: dict) -> bool:
        if "co_creation_enabled" not in config or config.get("co_creation_enabled") is None:
            return False
        co_creation = memory.setdefault("co_creation", self._empty_co_creation_state(str(memory.get("template_key") or DEFAULT_TEMPLATE_KEY)))
        next_enabled = bool(config.get("co_creation_enabled"))
        if bool(co_creation.get("enabled", True)) == next_enabled:
            return False
        co_creation["enabled"] = next_enabled
        self._sync_co_creation_state(co_creation, self._get_template_config(memory))
        return True

    def _build_co_creation_suggestions(self, memory: dict) -> list[dict[str, str]]:
        template_config = self._get_template_config(memory)
        co_creation = memory.setdefault("co_creation", self._empty_co_creation_state(str(template_config.get("template_key") or DEFAULT_TEMPLATE_KEY)))
        self._sync_co_creation_state(co_creation, template_config)
        suggestions = copy.deepcopy(co_creation.get("stage_options", []))
        stages = self._get_co_creation_stages(template_config)
        current_stage = next((item for item in stages if item.get("id") == co_creation.get("current_stage_id")), None)
        generate_prompt = str((current_stage or {}).get("generate_prompt") or "").strip()
        generate_label = str((template_config.get("co_creation_config") or {}).get("generate_current_label") or "一键生成当前片段").strip()
        if generate_prompt:
            suggestions.append(
                {
                    "id": f"{co_creation.get('current_stage_id', 'co_create')}_generate",
                    "label": generate_label,
                    "prompt": generate_prompt,
                    "category": "workflow",
                }
            )
        return self._normalize_suggestions(suggestions, default_category="co_create")

    def _lock_co_creation_foundation(self, memory: dict) -> None:
        co_creation = memory.setdefault("co_creation", self._empty_co_creation_state(str(memory.get("template_key") or DEFAULT_TEMPLATE_KEY)))
        locked_values = dict(co_creation.get("locked_values", {}))
        answers = memory.get("onboarding", {}).get("answers", {})
        for key in ("genre", "era", "players", "conflict", "protagonist", "goal"):
            value = str(answers.get(key) or "").strip()
            if value:
                locked_values[key] = value
        if memory.get("facts"):
            locked_values["facts"] = "；".join(self._merge_text_items(memory.get("facts", [])))
        if memory.get("role_cards"):
            locked_values["role_cards"] = json.dumps(memory.get("role_cards", []), ensure_ascii=False)
        co_creation["locked_values"] = locked_values
        co_creation["locked_fields"] = sorted(locked_values.keys())

    def _record_co_creation_sync(self, memory: dict, field: str, content: str) -> None:
        template_config = self._get_template_config(memory)
        sync_notice = str((template_config.get("co_creation_config") or {}).get("input_sync_notice") or "").strip()
        co_creation = memory.setdefault("co_creation", self._empty_co_creation_state(str(memory.get("template_key") or DEFAULT_TEMPLATE_KEY)))
        co_creation.setdefault("user_notes", {})[field] = str(content or "").strip()
        co_creation["sync_notice"] = sync_notice or "已同步你的修改，将基于新内容继续创作。"
        self._lock_co_creation_foundation(memory)

    def _get_current_co_creation_stage(self, memory: dict) -> dict[str, Any] | None:
        template_config = self._get_template_config(memory)
        co_creation = memory.setdefault("co_creation", self._empty_co_creation_state(str(template_config.get("template_key") or DEFAULT_TEMPLATE_KEY)))
        self._sync_co_creation_state(co_creation, template_config)
        stage_id = str(co_creation.get("current_stage_id") or "").strip()
        return next((item for item in self._get_co_creation_stages(template_config) if item.get("id") == stage_id), None)

    def _advance_co_creation_stage(self, memory: dict) -> None:
        template_config = self._get_template_config(memory)
        co_creation = memory.setdefault("co_creation", self._empty_co_creation_state(str(template_config.get("template_key") or DEFAULT_TEMPLATE_KEY)))
        stages = self._get_co_creation_stages(template_config)
        if not stages:
            return
        current_id = str(co_creation.get("current_stage_id") or "").strip()
        current_index = next((index for index, item in enumerate(stages) if item.get("id") == current_id), 0)
        next_index = min(current_index + 1, len(stages) - 1)
        co_creation["current_stage_id"] = stages[next_index]["id"]
        self._sync_co_creation_state(co_creation, template_config)

    def _is_co_creation_generate_request(self, memory: dict, message: str) -> bool:
        stage = self._get_current_co_creation_stage(memory) or {}
        generate_prompt = str(stage.get("generate_prompt") or "").strip()
        stripped = message.strip()
        if not stripped:
            return False
        return stripped == generate_prompt or "一键生成当前片段" in stripped or "补完整" in stripped

    @staticmethod
    def _is_co_creation_finish_request(message: str) -> bool:
        stripped = message.strip()
        if not stripped:
            return False
        finish_tokens = ("直接生成完整剧本", "强制成稿", "直接出稿", "确认", "可以了", "生成完整剧本", "补全剧本", "填充所有内容", "生成完整长剧本", "生成长剧情角色剧本", "生成3000字角色剧本")
        return any(token in stripped for token in finish_tokens)

    def _apply_co_creation_worldview(self, memory: dict, message: str) -> None:
        text = message.strip()
        if not text:
            return
        memory["facts"] = self._merge_text_items(memory.get("facts", []), [f"世界观方向：{text}"])
        memory.setdefault("co_creation", {}).setdefault("user_notes", {})["worldview"] = text

    def _apply_co_creation_roles(self, memory: dict, message: str) -> None:
        text = message.strip()
        if text:
            memory["facts"] = self._merge_text_items(memory.get("facts", []), [f"角色共创方向：{text}"])
            memory.setdefault("co_creation", {}).setdefault("user_notes", {})["roles"] = text
        if not memory.get("role_cards"):
            answers = memory.get("onboarding", {}).get("answers", {})
            memory["role_cards"] = self.llm_service.sanitize_role_cards(
                self.llm_service._build_role_cards(
                    answers.get("players", "6 人"),
                    answers.get("genre", "现代悬疑"),
                    answers.get("protagonist", "主控角色"),
                    answers.get("conflict", "核心冲突"),
                ),
                genre=answers.get("genre", ""),
                conflict=answers.get("conflict", ""),
            )
            memory["draft_content"] = self.llm_service.sync_draft_role_card_section(memory.get("draft_content", ""), memory["role_cards"])

    def _apply_co_creation_core_plot(self, memory: dict, message: str) -> None:
        text = message.strip()
        if text:
            memory["facts"] = self._merge_text_items(memory.get("facts", []), [f"剧情核心：{text}"])
            memory.setdefault("co_creation", {}).setdefault("user_notes", {})["core_plot"] = text
            memory.setdefault("onboarding", {}).setdefault("answers", {})["conflict"] = text
        answers = memory.get("onboarding", {}).get("answers", {})
        title = memory.get("summary", {}).get("title", "待命名剧本")
        memory["draft_content"] = self.llm_service._build_agent_draft(
            title,
            answers.get("era", "待确认"),
            answers.get("genre", "待确认"),
            answers.get("conflict", "核心冲突"),
            answers.get("protagonist", "主控角色"),
            answers.get("goal", "揭开真相"),
            memory.get("role_cards", []),
            memory,
        )

    def _apply_co_creation_detail_iteration(self, memory: dict, message: str) -> None:
        text = message.strip()
        if not text:
            return
        memory["facts"] = self._merge_text_items(memory.get("facts", []), [f"细节迭代：{text}"])
        memory.setdefault("co_creation", {}).setdefault("user_notes", {})["detail_iteration"] = text
        if memory.get("draft_content", "").strip():
            memory["draft_content"] = f"{memory['draft_content'].rstrip()}\n\n【细节迭代记录】\n- {text}"

    def _process_co_creation_turn(self, memory: dict, message: str, *, intro: bool = False) -> tuple[dict[str, Any], str]:
        if intro:
            self._lock_co_creation_foundation(memory)
            return self._build_co_creation_payload(memory, intro=True), "guide"

        stripped = message.strip()
        stage = self._get_current_co_creation_stage(memory) or {}
        stage_id = str(stage.get("id") or "").strip()
        template_config = self._get_template_config(memory)
        sync_notice = str((template_config.get("co_creation_config") or {}).get("panel_sync_notice") or "").strip()
        co_creation = memory.setdefault("co_creation", self._empty_co_creation_state(str(memory.get("template_key") or DEFAULT_TEMPLATE_KEY)))

        if stage_id == "detail_iteration" and self._is_co_creation_finish_request(stripped):
            co_creation["sync_notice"] = "已确认进入成稿流程，正在生成完整剧本。"
            self._lock_co_creation_foundation(memory)
            return self._build_co_creation_payload(memory, intro=False), "finish"

        if self._is_co_creation_generate_request(memory, stripped):
            if stage_id == "roles":
                self._apply_co_creation_roles(memory, stripped)
            elif stage_id == "core_plot":
                self._apply_co_creation_core_plot(memory, stripped)
            elif stage_id == "detail_iteration":
                self._apply_co_creation_detail_iteration(memory, stripped)
            else:
                self._apply_co_creation_worldview(memory, stripped)
            co_creation["sync_notice"] = sync_notice
            self._lock_co_creation_foundation(memory)
            return self._build_co_creation_payload(memory, intro=False), "local"

        if stage_id == "worldview":
            self._apply_co_creation_worldview(memory, stripped)
            co_creation["sync_notice"] = "已锁定世界观，并同步到设定面板。"
            self._advance_co_creation_stage(memory)
        elif stage_id == "roles":
            self._apply_co_creation_roles(memory, stripped)
            co_creation["sync_notice"] = "已同步角色设定，后续不会擅自覆盖这些核心人设。"
            self._advance_co_creation_stage(memory)
        elif stage_id == "core_plot":
            self._apply_co_creation_core_plot(memory, stripped)
            co_creation["sync_notice"] = "已锁定剧情核心，并同步到剧本稿。"
            self._advance_co_creation_stage(memory)
        else:
            self._apply_co_creation_detail_iteration(memory, stripped)
            co_creation["sync_notice"] = sync_notice or "已同步你的修改，将基于新内容继续创作。"

        self._lock_co_creation_foundation(memory)
        self._refresh_workspace_blueprint(memory, memory.get("summary", {}).get("title", "待命名剧本"))
        self._refresh_agent_memory_layers(memory)
        return self._build_co_creation_payload(memory, intro=False), "guide"

    def _record_onboarding_answer(self, memory: dict, message: str) -> None:
        onboarding = memory.get("onboarding", self._empty_onboarding_state())
        current_question_id = onboarding.get("current_question_id", "")
        normalized_answer = self._normalize_onboarding_answer(current_question_id, message)
        if current_question_id and normalized_answer:
            onboarding["answers"][current_question_id] = normalized_answer
            self._append_memory_node(
                memory,
                "onboarding_answer",
                f"步骤回答/{current_question_id}",
                normalized_answer,
                tags=[current_question_id],
            )

        inferred = self._sanitize_inferred_answers(self._infer_onboarding_answers(message))
        for key, value in inferred.items():
            onboarding["answers"].setdefault(key, value)

        self._sync_onboarding_state(onboarding, self._get_template_config(memory))
        memory["onboarding"] = onboarding
        memory["facts"] = self._build_onboarding_facts(onboarding["answers"])
        memory["summary"] = self._build_onboarding_summary(memory.get("summary", {}).get("title", "待命名剧本"), onboarding)
        self._refresh_agent_memory_layers(memory)

    def _build_onboarding_payload(self, memory: dict, latest_user_input: str, is_initial: bool = False) -> dict:
        template_config = self._get_template_config(memory)
        onboarding = memory.get("onboarding", self._empty_onboarding_state(str(template_config.get("template_key") or DEFAULT_TEMPLATE_KEY)))
        answers = onboarding.get("answers", {})
        progress = onboarding.get("progress", 0)
        total_questions = onboarding.get("total_questions", len(self._get_question_flow(template_config)))
        current_question = onboarding.get("current_question", "")

        if is_initial:
            leading = f"我会先通过 {total_questions} 个问题帮你完成世界观设定，然后再生成完整剧本草案。"
        else:
            current_question_id = onboarding.get("current_question_id", "")
            recorded = self._normalize_onboarding_answer(current_question_id, latest_user_input)
            leading = f"已记录你的回答：{recorded}。" if recorded else "已收到你的补充信息。"

        answer_preview = "；".join(self._build_onboarding_facts(answers)) or "当前还没有完整设定。"
        assistant_reply = (
            f"{leading} 当前进度 {progress}/{total_questions}。"
            f"已确认信息：{answer_preview}。"
            f"接下来请回答：{current_question}"
        )

        return {
            "assistant_reply": assistant_reply,
            "draft_content": memory.get("draft_content", ""),
            "suggestions": self._build_question_suggestions(onboarding, template_config),
            "facts": self._build_onboarding_facts(answers),
            "summary": self._build_onboarding_summary(memory.get("summary", {}).get("title", "待命名剧本"), onboarding),
            "logic_checks": self._build_onboarding_logic_checks(onboarding),
            "memory_summary": self.memory_service.build_agent_memory_summary(memory),
            "memory_stats": memory.get("memory_stats", {}),
            **self._build_workspace_blueprint(memory, memory.get("summary", {}).get("title", "待命名剧本")),
        }

    def _build_co_creation_payload(self, memory: dict, *, intro: bool = False) -> dict:
        template_config = self._get_template_config(memory)
        co_creation_config = template_config.get("co_creation_config") or {}
        co_creation = memory.setdefault("co_creation", self._empty_co_creation_state(str(template_config.get("template_key") or DEFAULT_TEMPLATE_KEY)))
        self._sync_co_creation_state(co_creation, template_config)
        stage_id = str(co_creation.get("current_stage_id") or "").strip()
        current_stage = next((item for item in self._get_co_creation_stages(template_config) if item.get("id") == stage_id), None)
        stage_title = str((current_stage or {}).get("title") or "当前阶段").strip()
        instruction = str((current_stage or {}).get("instruction") or "请继续补充当前阶段需求。").strip()
        placeholder = str((current_stage or {}).get("placeholder") or "").strip()
        answer_preview = "；".join(self._merge_text_items(memory.get("facts", []))) or "当前还没有稳定设定。"
        locked_notice = str(co_creation_config.get("stage_locked_notice") or "").strip()
        sync_notice = str(co_creation.get("sync_notice") or "").strip()
        prefix = "共创模式已开启。" if intro else f"已进入 {stage_title}。"
        if stage_id == "detail_iteration":
            instruction = (
                f"{instruction}\n"
                "当前可执行动作：\n"
                "1. 输入细化方向，我会只优化你指定的细节；\n"
                "2. 点击“生成完整剧本”直接进入成稿；\n"
                "3. 输入“确认”或“可以了”也可直接出稿。"
            )
        assistant_reply = (
            f"{prefix}\n"
            f"{instruction}\n"
            f"当前已锁定信息：{answer_preview}\n"
            f"{locked_notice}"
        ).strip()
        if placeholder:
            assistant_reply += f"\n可直接这样补充：{placeholder}"
        if sync_notice:
            assistant_reply += f"\n{sync_notice}"
        memory.setdefault("summary", {})["focus"] = f"共创进行中：{stage_title}"
        co_creation["awaiting_user"] = True
        co_creation["stage_options"] = copy.deepcopy((current_stage or {}).get("options", []))
        return {
            "assistant_reply": assistant_reply,
            "draft_content": memory.get("draft_content", ""),
            "suggestions": self._build_co_creation_suggestions(memory),
            "facts": memory.get("facts", []),
            "role_cards": memory.get("role_cards", []),
            "summary": {
                **memory.get("summary", {}),
                "focus": f"共创进行中：{stage_title}",
                "template_key": memory.get("template_key", DEFAULT_TEMPLATE_KEY),
                "template_label": memory.get("template_label", "通用剧本模板"),
                "template_family": memory.get("template_family", "general"),
            },
            "logic_checks": self._build_onboarding_logic_checks(memory.get("onboarding", {})),
            "memory_summary": self.memory_service.build_agent_memory_summary(memory),
            "memory_stats": memory.get("memory_stats", {}),
            **self._build_workspace_blueprint(memory, memory.get("summary", {}).get("title", "待命名剧本")),
        }

    def _build_clarification_payload(self, memory: dict) -> dict:
        template_config = self._get_template_config(memory)
        onboarding = memory.get("onboarding", self._empty_onboarding_state(str(template_config.get("template_key") or DEFAULT_TEMPLATE_KEY)))
        question = self._find_question(onboarding.get("current_question_id", ""), template_config)
        if not question:
            return self._build_onboarding_payload(memory, "")

        answer_preview = "；".join(self._build_onboarding_facts(onboarding.get("answers", {}))) or "当前还没有完整设定。"
        examples = "；".join(question.get("examples", []))
        explanation = str(question.get("explanation") or "这一项用于进一步收窄你的世界观与创作方向。").strip()
        assistant_reply = (
            f"我先解释一下。{question['question']} 的意思是：{explanation}"
            f" 你可以直接像这样回答：{examples}。"
            f" 当前已确认信息：{answer_preview}。"
            " 如果你不确定，也可以先给我一个大致方向，我会继续帮你收窄。"
        )
        return {
            "assistant_reply": assistant_reply,
            "draft_content": memory.get("draft_content", ""),
            "suggestions": self._build_question_suggestions(onboarding, template_config),
            "facts": self._build_onboarding_facts(onboarding.get("answers", {})),
            "summary": self._build_onboarding_summary(memory.get("summary", {}).get("title", "待命名剧本"), onboarding),
            "logic_checks": self._build_onboarding_logic_checks(onboarding),
            "memory_summary": self.memory_service.build_agent_memory_summary(memory),
            "memory_stats": memory.get("memory_stats", {}),
            **self._build_workspace_blueprint(memory, memory.get("summary", {}).get("title", "待命名剧本")),
        }

    def _build_question_suggestions(self, onboarding: dict, template_config: dict[str, Any] | None = None) -> list[dict[str, str]]:
        current_question = self._find_question(onboarding.get("current_question_id", ""), template_config)
        if not current_question:
            return []
        suggestions = [
            {"id": f"{current_question['id']}_{index}", "label": option, "prompt": option, "category": "onboarding"}
            for index, option in enumerate(current_question["options"], start=1)
        ]
        return self._normalize_suggestions(suggestions, default_category="onboarding")

    def _normalize_suggestions(self, suggestions: list[dict[str, str]], default_category: str = "create") -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        seen_keys: set[tuple[str, str]] = set()
        for item in suggestions:
            prompt = str(item.get("prompt", "")).strip()
            label = str(item.get("label", "")).strip()
            category = str(item.get("category", "")).strip() or default_category
            dedupe_key = (category, prompt)
            if not prompt or not label or dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            normalized.append(
                {
                    "id": str(item.get("id", f"suggestion_{len(normalized) + 1}")).strip() or f"suggestion_{len(normalized) + 1}",
                    "label": label,
                    "prompt": prompt,
                    "category": category,
                }
            )
            if len(normalized) == self.MAX_SUGGESTIONS:
                break
        return normalized

    def _ensure_three_suggestions(self, suggestions: list[dict[str, str]]) -> list[dict[str, str]]:
        return self._normalize_suggestions(suggestions)

    @staticmethod
    def _short_text(text: str, limit: int = 18) -> str:
        stripped = (text or "").strip()
        if len(stripped) <= limit:
            return stripped
        return stripped[:limit].rstrip() + "..."

    def _build_contextual_suggestions(self, memory: dict, payload: dict, blueprint: dict) -> list[dict[str, str]]:
        user_messages = [item.get("content", "") for item in memory.get("messages", []) if item.get("role") == "user"]
        latest_user_message = user_messages[-1] if user_messages else ""
        assistant_reply = payload.get("assistant_reply", "")
        summary = payload.get("summary", {})
        current_focus = summary.get("focus", "")
        intent = blueprint.get("intent", memory.get("intent", {}))
        task_queue = blueprint.get("task_queue", memory.get("task_queue", []))
        next_action = str(intent.get("next_action", "")).strip()
        draft_content = payload.get("draft_content", memory.get("draft_content", ""))
        role_cards = payload.get("role_cards", memory.get("role_cards", []))
        has_draft = bool((draft_content or "").strip())
        has_role_cards = bool(role_cards)

        primary_text = "\n".join(
            fragment for fragment in (latest_user_message, assistant_reply, current_focus) if fragment
        )
        progress_text = "\n".join(
            fragment for fragment in (latest_user_message, assistant_reply, current_focus, next_action) if fragment
        )
        request_text = latest_user_message.strip()
        candidates: list[dict[str, str]] = []

        keyword_recipes = [
            (("秘密", "动机", "身份"), "细化人物秘密", "请围绕当前草案继续细化 6 位角色的隐藏秘密、公开身份冲突和核心动机。"),
            (("线索", "搜证", "物证", "证据"), "补搜证线索", "请继续补充搜证线索、触发条件和误导线索，并标清每条线索的真实作用。"),
            (("氛围", "恐怖", "压迫", "惊悚"), "加强氛围", "请在不破坏逻辑的前提下，加强场景压迫感、声光细节和 DM 的氛围提示。"),
            (("结局", "反转", "真相"), "重写结局", "请在保留当前主线的前提下，给出一个反转更强、回收更完整的结局版本。"),
            (("角色卡", "角色", "人物关系"), "补角色卡", "请继续补齐角色卡的人设标签、行为约束、关系链和关键道具。"),
            (("dm", "主持", "控场", "话术"), "补DM手册", "请补充当前剧本的 DM 开场、转场、应急控场和结局宣读话术。"),
            (("复盘", "回顾", "行为记录"), "生成复盘", "请基于当前草案整理一版复盘报告草案，包含行为摘要、关键节点和真相揭示。"),
            (("分幕", "结构", "节奏", "时间线"), "梳理结构", "请把当前剧本按分幕、节奏和时间线重新梳理得更清晰。"),
        ]
        for keywords, label, prompt in keyword_recipes:
            if any(keyword in primary_text for keyword in keywords):
                candidates.append(
                    {
                        "id": f"context_{len(candidates) + 1}",
                        "label": label,
                        "prompt": prompt,
                    }
                )
        if len(candidates) < 2:
            for keywords, label, prompt in keyword_recipes:
                if any(keyword in progress_text for keyword in keywords):
                    candidates.append(
                        {
                            "id": f"context_{len(candidates) + 1}",
                            "label": label,
                            "prompt": prompt,
                        }
                    )

        active_task = next((item for item in task_queue if item.get("status") == "active"), None)
        pending_task = next((item for item in task_queue if item.get("status") == "pending"), None)
        task_hint = active_task or pending_task
        if not has_draft:
            candidates.append(
                {
                    "id": "progress_outline",
                    "label": "先出完整草案",
                    "prompt": "请先基于当前设定输出一版完整草案，再继续细化局部内容。",
                }
            )
        elif not has_role_cards:
            candidates.append(
                {
                    "id": "progress_roles",
                    "label": "补齐角色卡",
                    "prompt": "请优先补齐当前剧本的角色卡、人物关系和关键秘密。",
                }
            )
        elif task_hint:
            task_title = self._short_text(str(task_hint.get("title", "")), 16) or "当前重点"
            candidates.append(
                {
                    "id": "progress_next",
                    "label": f"推进{task_title}",
                    "prompt": f"请根据当前完成进度，优先推进：{task_hint.get('title', '当前重点')}。{task_hint.get('detail', '')}",
                }
            )
        elif next_action:
            candidates.append(
                {
                    "id": "progress_action",
                    "label": "按当前进度继续",
                    "prompt": f"请按当前完成进度继续推进，优先处理：{next_action}。",
                }
            )

        if current_focus:
            candidates.append(
                {
                    "id": "focus_followup",
                    "label": "继续当前重点",
                    "prompt": f"请围绕“{current_focus}”继续深化，保留已经确认的设定不变。",
                }
            )

        candidates.extend(payload.get("suggestions", []))
        return self._ensure_three_suggestions(candidates)

    @staticmethod
    def _pick_relevant_hint(text: str, values: list[str], fallback: str = "") -> str:
        for value in values:
            candidate = str(value or "").strip()
            if candidate and candidate in text:
                return candidate
        return fallback

    @staticmethod
    def _build_genre_atmosphere_recipe(genre: str, stage_hint: str, short_stage_hint: str) -> dict[str, str]:
        normalized = (genre or "").strip()
        if normalized == "欢乐":
            return {
                "keywords": ("氛围", "欢乐", "互动", "翻车", "抛梗"),
                "label": f"加强{short_stage_hint or '当前'}互动",
                "prompt": f"请在{stage_hint}加强抬杠、爆料、配对试探和翻车梗，让场上互动更热闹，但不要破坏已经确认的逻辑和人物关系。",
                "category": "create",
            }
        if normalized == "情感":
            return {
                "keywords": ("氛围", "情绪", "沉浸", "拉扯", "共情"),
                "label": f"加深{short_stage_hint or '当前'}情绪",
                "prompt": f"请在{stage_hint}加强情绪递进、关系拉扯和旧事回收，让玩家更容易代入当前情感冲突，但不要破坏既有逻辑。",
                "category": "create",
            }
        if normalized == "恐怖":
            return {
                "keywords": ("氛围", "恐怖", "压迫", "惊悚"),
                "label": f"加压{short_stage_hint or '当前'}氛围",
                "prompt": f"请在{stage_hint}加强压迫感、异常细节和 DM 控场提示，但不要破坏已经确认的逻辑和线索关系。",
                "category": "create",
            }
        return {
            "keywords": ("氛围", "张力", "悬疑", "节奏"),
            "label": f"加强{short_stage_hint or '当前'}张力",
            "prompt": f"请在{stage_hint}加强悬念、对峙感和信息张力，但不要破坏已经确认的逻辑和线索关系。",
            "category": "create",
        }

    def _build_contextual_suggestions(self, memory: dict, payload: dict, blueprint: dict) -> list[dict[str, str]]:
        user_messages = [item.get("content", "") for item in memory.get("messages", []) if item.get("role") == "user"]
        latest_user_message = user_messages[-1] if user_messages else ""
        assistant_reply = payload.get("assistant_reply", "")
        summary = payload.get("summary", {})
        genre = str(summary.get("genre") or memory.get("summary", {}).get("genre", "")).strip()
        current_focus = summary.get("focus", "")
        intent = blueprint.get("intent", memory.get("intent", {}))
        task_queue = blueprint.get("task_queue", memory.get("task_queue", []))
        next_action = str(intent.get("next_action", "")).strip()
        draft_content = payload.get("draft_content", memory.get("draft_content", ""))
        role_cards = payload.get("role_cards", memory.get("role_cards", []))
        structure_plan = blueprint.get("structure_plan", memory.get("structure_plan", []))
        output_outline = blueprint.get("output_outline", memory.get("output_outline", []))
        logic_checks = payload.get("logic_checks", memory.get("logic_checks", []))
        draft_mode = str(memory.get("draft_mode") or "complete").strip().lower()
        has_draft = bool((draft_content or "").strip())
        has_role_cards = bool(role_cards)
        role_names = [str(item.get("name", "")).strip() for item in role_cards if item.get("name")]
        stage_titles = [str(item.get("title", "")).strip() for item in structure_plan if item.get("title")]
        section_titles = self.llm_service._extract_draft_sections(draft_content) if has_draft else []
        output_titles = [str(item.get("title", "")).strip() for item in output_outline if item.get("title")]

        primary_text = "\n".join(
            fragment for fragment in (latest_user_message, assistant_reply, current_focus) if fragment
        )
        progress_text = "\n".join(
            fragment for fragment in (latest_user_message, assistant_reply, current_focus, next_action) if fragment
        )
        request_text = latest_user_message.strip()
        candidates: list[dict[str, str]] = []
        request_hint = self._short_text(latest_user_message or current_focus or next_action, 10) or "当前需求"
        role_hint = self._pick_relevant_hint(primary_text, role_names, role_names[0] if role_names else "核心角色")
        stage_hint = self._pick_relevant_hint(primary_text, stage_titles, stage_titles[0] if stage_titles else "当前关键阶段")
        short_stage_hint = self._short_text(stage_hint, 8)
        section_hint = self._pick_relevant_hint(primary_text, section_titles, section_titles[0] if section_titles else "当前重点章节")
        output_hint = self._pick_relevant_hint(primary_text, output_titles, output_titles[0] if output_titles else "")
        atmosphere_recipe = self._build_genre_atmosphere_recipe(genre, stage_hint, short_stage_hint)

        keyword_recipes = [
            {
                "keywords": ("秘密", "动机", "身份"),
                "label": f"细化{self._short_text(role_hint, 8) or '人物'}秘密",
                "prompt": f"请围绕{role_hint}继续细化隐藏秘密、公开身份冲突和行为动机，并补一个能让玩家触发秘密暴露的互动节点。",
                "category": "create",
            },
            {
                "keywords": ("长文本", "长稿", "正文", "剧情", "故事"),
                "label": f"补写{self._short_text(section_hint, 8) or '剧情'}长稿",
                "prompt": f"请把{section_hint}补成可直接使用的长文本剧情，重点补场景动作、角色试探、线索触发和后续影响，不要只列提纲。",
                "category": "create",
            },
            {
                "keywords": ("玩家手册", "玩家本", "角色手册", "叙事版"),
                "label": "生成玩家手册",
                "prompt": "请把当前剧本整理成玩家手册叙事版，用连续剧情写出角色视角、误解、秘密和每幕推进。",
                "category": "create",
            },
            {
                "keywords": ("角色剧本", "角色本", "个人本", "长剧情角色剧本"),
                "label": "生成角色剧本",
                "prompt": "请基于当前设定批量生成每个角色的长剧情角色剧本，保持与主持手册、真相页和线索链一致。",
                "category": "create",
            },
            {
                "keywords": ("线索", "搜证", "物证", "证据"),
                "label": f"补强{self._short_text(stage_hint, 8) or '关键'}线索",
                "prompt": f"请补强{stage_hint}的搜证线索、获取方式、误导线索和回收节点，并让它们能对应到当前真相链。",
                "category": "create",
            },
            atmosphere_recipe,
            {
                "keywords": ("结局", "反转", "真相"),
                "label": "重写结局回收",
                "prompt": "请在保留当前主线冲突的前提下，重写结局与真相回收，让前文伏笔、角色动机和关键物证都能对应落地。",
                "category": "create",
            },
            {
                "keywords": ("角色卡", "角色", "人物关系"),
                "label": "补全角色关系",
                "prompt": f"请继续补充角色关系链、站队变化和冲突触发点，重点看{role_hint}与其他角色的牵连。",
                "category": "create",
            },
            {
                "keywords": ("dm", "主持", "控场", "话术"),
                "label": f"补{self._short_text(stage_hint, 8) or '当前'}DM话术",
                "prompt": f"请补充{stage_hint}的 DM 开场、转场、应急圆场和控场提示，让主持人可以直接口播。",
                "category": "create",
            },
            {
                "keywords": ("复盘", "回顾", "行为记录"),
                "label": "整理复盘要点",
                "prompt": "请基于当前草案整理一版复盘要点清单，包含关键行为、真相揭示和未触发支线的说明。",
                "category": "workflow",
            },
            {
                "keywords": ("分幕", "结构", "节奏", "时间线"),
                "label": f"梳理{self._short_text(section_hint, 8) or '结构'}",
                "prompt": f"请把{section_hint}按分幕节奏、时间线和关键转折点重新梳理得更清晰，避免剧情堆在一起。",
                "category": "create",
            },
            {
                "keywords": ("分支", "选择", "后果"),
                "label": "补全分支后果",
                "prompt": f"请为{stage_hint}补写核心选择、次级分支和后续后果，让玩家的选择能真正影响后续走向。",
                "category": "create",
            },
        ]
        for recipe in keyword_recipes:
            if request_text and any(keyword in request_text for keyword in recipe["keywords"]):
                candidates.append(
                    {
                        "id": f"context_{len(candidates) + 1}",
                        "label": recipe["label"],
                        "prompt": recipe["prompt"],
                        "category": recipe["category"],
                    }
                )
        if len(candidates) < 2:
            for recipe in keyword_recipes:
                if any(keyword in primary_text for keyword in recipe["keywords"]):
                    candidates.append(
                        {
                            "id": f"context_{len(candidates) + 1}",
                            "label": recipe["label"],
                            "prompt": recipe["prompt"],
                            "category": recipe["category"],
                        }
                    )
        if len(candidates) < 2:
            for recipe in keyword_recipes:
                if any(keyword in progress_text for keyword in recipe["keywords"]):
                    candidates.append(
                        {
                            "id": f"context_{len(candidates) + 1}",
                            "label": recipe["label"],
                            "prompt": recipe["prompt"],
                            "category": recipe["category"],
                        }
                    )

        if has_draft:
            candidates.append(
                {
                    "id": "progress_complete_script_priority",
                    "label": "生成完整可用剧本" if draft_mode == "complete" else "生成完整成稿",
                    "prompt": "请直接基于当前设定和现有草案，生成一版完整可上桌的剧本杀成稿，正文里同时整合剧情长稿、分幕流程、角色关系、搜证线索、DM 关键话术、玩家手册视角和终局真相回收，尽量做到不需要我再去侧边交付包里二次编辑。",
                    "category": "create",
                }
            )
            candidates.append(
                {
                    "id": "progress_longform_priority",
                    "label": "补写剧情长稿",
                    "prompt": f"请基于当前草案，把{section_hint}扩写成更完整的长文本剧情，补足场景、对白、线索推进和结尾钩子。",
                    "category": "create",
                }
            )
            candidates.append(
                {
                    "id": "progress_player_book_priority",
                    "label": "整理玩家手册",
                    "prompt": "请把当前剧本整理成玩家手册叙事版，用角色视角写成连续故事剧情，并保留秘密、误导信息和每幕推进。",
                    "category": "workflow",
                }
            )
            candidates.append(
                {
                    "id": "progress_role_script_priority",
                    "label": "生成角色剧本",
                    "prompt": "请基于当前设定批量生成每个角色的长剧情角色剧本，保持与主持手册、真相页和线索链一致。",
                    "category": "workflow",
                }
            )

        if latest_user_message.strip():
            continue_label = "继续细化当前内容" if has_draft else "继续推进这个方向"
            candidates.append(
                {
                    "id": "request_followup",
                    "label": continue_label,
                    "prompt": f"请沿着我刚才这轮“{latest_user_message.strip()}”继续深化，这次优先补充还没展开的细节，不要重复已经完成的部分。",
                    "category": "create",
                }
            )

        if output_hint and "玩家手册" in output_hint:
            candidates.append(
                {
                    "id": "output_player_book_priority",
                    "label": "整理玩家手册",
                    "prompt": "请把当前剧本整理成玩家手册叙事版，用角色视角写成连续故事剧情，并保留秘密、误导信息和每幕推进。",
                    "category": "workflow",
                }
            )

        active_task = next((item for item in task_queue if item.get("status") == "active"), None)
        pending_task = next((item for item in task_queue if item.get("status") == "pending"), None)
        task_hint = active_task or pending_task
        if not has_draft:
            candidates.append(
                {
                    "id": "progress_outline",
                    "label": "先出完整草案",
                    "prompt": "请先基于当前设定输出一版完整草案，再继续细化局部内容。",
                    "category": "create",
                }
            )
        elif not has_role_cards:
            candidates.append(
                {
                    "id": "progress_roles",
                    "label": "补齐角色卡",
                    "prompt": "请优先补齐当前剧本的角色卡、人物关系和关键秘密。",
                    "category": "create",
                }
            )
        elif task_hint:
            task_title = self._short_text(str(task_hint.get("title", "")), 16) or "当前重点"
            candidates.append(
                {
                    "id": "progress_next",
                    "label": f"推进{task_title}",
                    "prompt": f"请根据当前完成进度，优先推进：{task_hint.get('title', '当前重点')}。{task_hint.get('detail', '')}",
                    "category": "create",
                }
            )
        elif next_action:
            candidates.append(
                {
                    "id": "progress_action",
                    "label": "按当前进度继续",
                    "prompt": f"请按当前完成进度继续推进，优先处理：{next_action}。",
                    "category": "create",
                }
            )

        open_logic_checks = [
            item for item in logic_checks
            if isinstance(item, str) and (item.startswith("提示：") or item.startswith("未通过："))
        ]
        if open_logic_checks:
            logic_hint = self._short_text(
                open_logic_checks[0].replace("提示：", "").replace("未通过：", ""),
                10,
            ) or "逻辑问题"
            candidates.append(
                {
                    "id": "logic_fix",
                    "label": f"修正{logic_hint}",
                    "prompt": f"请优先修正当前逻辑检查里提到的问题：{'；'.join(open_logic_checks[:2])}。修正时保持既有主线和角色设定不变。",
                    "category": "create",
                }
            )

        if current_focus:
            candidates.append(
                {
                    "id": "focus_followup",
                    "label": f"围绕{self._short_text(current_focus, 8) or '当前重点'}",
                    "prompt": f"请围绕“{current_focus}”继续深化，保留已经确认的设定不变。",
                    "category": "create",
                }
            )

        if output_hint:
            candidates.append(
                {
                    "id": "output_followup",
                    "label": f"整理{self._short_text(output_hint, 8)}",
                    "prompt": f"请把当前内容进一步整理成“{output_hint}”可直接交付的结构化内容。",
                    "category": "workflow",
                }
            )

        candidates.extend(payload.get("suggestions", []))
        return self._normalize_suggestions(candidates)

    def _build_onboarding_facts(self, answers: dict[str, str]) -> list[str]:
        labels = {
            "genre": "剧本类型",
            "era": "世界观",
            "players": "参与人数",
            "conflict": "核心冲突",
            "protagonist": "主控视角",
            "goal": "玩家目标",
        }
        return [f"{labels[key]}：{value}" for key, value in answers.items() if value and key in labels]

    def _build_onboarding_summary(self, title: str, onboarding: dict) -> dict:
        answers = onboarding.get("answers", {})
        resolved_title = self._resolve_session_title(title, answers)
        focus = onboarding.get("current_question") or "生成完整世界观草案"
        return {
            "title": resolved_title,
            "era": answers.get("era", "待确认"),
            "genre": answers.get("genre", "待确认"),
            "focus": focus,
            "revision_count": int(onboarding.get("progress", 0)),
        }

    def _resolve_session_title(self, current_title: str, answers: dict[str, str]) -> str:
        normalized_title = (current_title or "").strip()
        if normalized_title and normalized_title != "待命名剧本":
            return normalized_title

        conflict = answers.get("conflict", "")
        era = answers.get("era", "")
        genre = answers.get("genre", "")

        keyword_titles = [
            ("相亲", "良缘翻车局"),
            ("配对", "心动错位局"),
            ("密室", "密室回响"),
            ("旧案", "旧案回潮"),
            ("校园", "钟楼旧闻"),
            ("家族", "家宴暗潮"),
            ("争产", "继承人名单"),
            ("实验", "实验余烬"),
        ]
        for token, candidate in keyword_titles:
            if token in conflict:
                seed = candidate
                break
        else:
            era_seed_map = {
                "现代都市": "雾港夜宴",
                "现代校园": "钟楼旧闻",
                "民国小镇": "戏楼疑云",
                "古风武侠": "山门夜雪",
                "未来科幻": "零号回响",
            }
            seed = era_seed_map.get(era, "未命名剧本")

        suffix_map = {
            "欢乐": "欢乐局",
            "情感": "情感本",
            "硬核": "推理局",
            "恐怖": "惊夜本",
            "现代悬疑": "悬疑本",
        }
        suffix = suffix_map.get(genre, "")
        if not suffix or seed.endswith(suffix):
            return seed
        return f"{seed}·{suffix}"

    def _sync_onboarding_state(self, onboarding: dict, template_config: dict[str, Any] | None = None) -> None:
        resolved_template = template_config or self.template_config_service.get_template(
            str(onboarding.get("template_key") or DEFAULT_TEMPLATE_KEY)
        )
        question_flow = self._get_question_flow(resolved_template)
        onboarding["template_key"] = str(resolved_template.get("template_key") or DEFAULT_TEMPLATE_KEY)
        onboarding["total_questions"] = len(question_flow)
        answered = sum(1 for question in question_flow if onboarding["answers"].get(question["id"]))
        onboarding["progress"] = answered
        next_question = next((question for question in question_flow if not onboarding["answers"].get(question["id"])), None)
        if next_question is None:
            onboarding["completed"] = True
            onboarding["current_question_id"] = ""
            onboarding["current_question"] = ""
            onboarding["current_placeholder"] = ""
        else:
            onboarding["completed"] = False
            onboarding["current_question_id"] = next_question["id"]
            onboarding["current_question"] = next_question["question"]
            onboarding["current_placeholder"] = next_question["placeholder"]

    def _jump_to_question(self, onboarding: dict, target_id: str, template_config: dict[str, Any] | None = None) -> None:
        resolved_template = template_config or self.template_config_service.get_template(
            str(onboarding.get("template_key") or DEFAULT_TEMPLATE_KEY)
        )
        question_flow = self._get_question_flow(resolved_template)
        target_index = next((index for index, item in enumerate(question_flow) if item["id"] == target_id), None)
        if target_index is None:
            raise HTTPException(status_code=400, detail="未找到要回退的步骤。")

        for question in question_flow[target_index:]:
            onboarding["answers"].pop(question["id"], None)
        self._sync_onboarding_state(onboarding, resolved_template)

    def _skip_question(self, onboarding: dict, target_id: str, template_config: dict[str, Any] | None = None) -> None:
        resolved_template = template_config or self.template_config_service.get_template(
            str(onboarding.get("template_key") or DEFAULT_TEMPLATE_KEY)
        )
        target_question = self._find_question(target_id, resolved_template)
        if not target_question:
            raise HTTPException(status_code=400, detail="未找到要跳过的步骤。")
        onboarding["answers"][target_id] = "待补充"
        self._sync_onboarding_state(onboarding, resolved_template)

    def _infer_onboarding_answers(self, text: str) -> dict[str, str]:
        stripped = text.strip()
        answers: dict[str, str] = {}
        if not stripped:
            return answers

        genre_map = {
            "恐怖": "恐怖",
            "情感": "情感",
            "硬核": "硬核",
            "欢乐": "欢乐",
            "悬疑": "现代悬疑",
        }
        for token, value in genre_map.items():
            if token in stripped:
                answers["genre"] = value
                break

        era_map = {
            "现代都市": "现代都市",
            "现代校园": "现代校园",
            "民国": "民国小镇",
            "古风": "古风武侠",
            "武侠": "古风武侠",
            "未来": "未来科幻",
            "科幻": "未来科幻",
            "空间站": "未来科幻",
        }
        for token, value in era_map.items():
            if token in stripped:
                answers["era"] = value
                break

        player_match = re.search(r"(\d+)\s*人", stripped)
        if player_match:
            answers["players"] = f"{player_match.group(1)} 人"

        role_map = {
            "记者": "记者",
            "医生": "医生",
            "学生": "学生",
            "侦探": "侦探",
            "组织者": "组织者",
            "主持人": "组织者",
        }
        for token, value in role_map.items():
            if token in stripped:
                answers["protagonist"] = value
                break

        if "配对成功" in stripped or "完成配对" in stripped:
            answers["goal"] = "完成配对"
        elif "找出真凶" in stripped:
            answers["goal"] = "找出真凶"
        elif "揭开真相" in stripped or "查明真相" in stripped:
            answers["goal"] = "揭开真相"
        elif "站队" in stripped:
            answers["goal"] = "达成站队"

        if any(token in stripped for token in ("密室", "命案", "旧案", "相亲", "配对", "争产", "实验", "家族")):
            answers["conflict"] = stripped[:80]

        return answers

    def _normalize_onboarding_answer(self, question_id: str, answer: str) -> str:
        stripped = answer.strip()
        if not stripped:
            return ""

        if question_id == "genre":
            return self._infer_onboarding_answers(stripped).get("genre", stripped)
        if question_id == "era":
            return self._infer_onboarding_answers(stripped).get("era", stripped)
        if question_id == "players":
            range_match = re.search(r"(\d+)\s*[-~到至]\s*(\d+)", stripped)
            if range_match:
                return f"{range_match.group(1)}-{range_match.group(2)} 人"
            match = re.search(r"(\d+)", stripped)
            return f"{match.group(1)} 人" if match else stripped
        if question_id == "goal":
            return self._infer_onboarding_answers(stripped).get("goal", stripped)
        return stripped

    def _validate_onboarding_answer(self, question_id: str, answer: str) -> str | None:
        stripped = answer.strip()
        if not stripped:
            return "当前步骤不能为空，请按提示补充关键信息。"

        if question_id == "players":
            range_match = re.search(r"(\d+)\s*[-~到至]\s*(\d+)", stripped)
            if range_match:
                left = int(range_match.group(1))
                right = int(range_match.group(2))
                if left > right:
                    return "参与人数范围需要从小到大填写，例如“6-8 人”。"
                if left < 2 or right > 10:
                    return "参与人数请填写 2-10 人，可输入“6 人”或“6-8 人”。"
                return None

            match = re.search(r"(\d+)", stripped)
            if not match:
                return "参与人数请填写 2-10 人，可输入“6 人”或“6-8 人”。"
            count = int(match.group(1))
            if count < 2 or count > 10:
                return "参与人数请填写 2-10 人，可输入“6 人”或“6-8 人”。"

        if question_id in {"genre", "era", "conflict", "protagonist", "goal"} and len(stripped) < 2:
            return "这一项信息过短，请至少提供一个明确方向。"
        return None

    def _build_validation_payload(self, memory: dict, validation_error: str) -> dict:
        onboarding = memory.get("onboarding", self._empty_onboarding_state())
        current_question = onboarding.get("current_question", "当前步骤")
        assistant_reply = (
            f"{current_question} 还没有通过校验。{validation_error}"
            " 你可以直接修改后重新输入，我会在校验通过后继续推进下一阶段。"
        )
        self._append_memory_node(
            memory,
            "validation",
            "参数校验未通过",
            validation_error,
            tags=["validation", onboarding.get("current_question_id", "")],
        )
        self._refresh_agent_memory_layers(memory)
        return {
            "assistant_reply": assistant_reply,
            "draft_content": memory.get("draft_content", ""),
            "suggestions": self._build_question_suggestions(onboarding, self._get_template_config(memory)),
            "facts": self._build_onboarding_facts(onboarding.get("answers", {})),
            "summary": self._build_onboarding_summary(memory.get("summary", {}).get("title", "待命名剧本"), onboarding),
            "logic_checks": self._build_onboarding_logic_checks(onboarding, validation_error),
            "memory_summary": self.memory_service.build_agent_memory_summary(memory),
            "memory_stats": memory.get("memory_stats", {}),
            **self._build_workspace_blueprint(memory, memory.get("summary", {}).get("title", "待命名剧本")),
        }

    def _sanitize_inferred_answers(self, answers: dict[str, str]) -> dict[str, str]:
        sanitized: dict[str, str] = {}
        for key, value in answers.items():
            validation_error = self._validate_onboarding_answer(key, value)
            if validation_error:
                continue
            normalized = self._normalize_onboarding_answer(key, value)
            if normalized:
                sanitized[key] = normalized
        return sanitized

    def _append_memory_node(
        self,
        memory: dict,
        node_type: str,
        label: str,
        content: str,
        *,
        tags: list[str] | None = None,
    ) -> None:
        if not content.strip():
            return
        nodes = memory.setdefault("memory_nodes", [])
        nodes.append(
            {
                "id": str(uuid4()),
                "type": node_type,
                "label": label,
                "content": content.strip(),
                "tags": [tag for tag in (tags or []) if tag],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        if len(nodes) > 60:
            del nodes[:-60]

    def _refresh_agent_memory_layers(self, memory: dict) -> None:
        self._sanitize_memory_role_cards(memory)
        answers = memory.get("onboarding", {}).get("answers", {})
        role_cards = memory.get("role_cards", [])
        summary = memory.get("summary", {})
        structure_plan = memory.get("structure_plan", [])
        memory["generation_rules"] = self.llm_service.rule_engine.build_agent_generation_rules(
            answers,
            role_cards,
            str(summary.get("template_key") or memory.get("template_key") or DEFAULT_TEMPLATE_KEY),
        )
        memory["logic_checks"] = self.llm_service.rule_engine.build_agent_logic_checks(
            memory=memory,
            draft_content=memory.get("draft_content", ""),
            role_cards=role_cards,
            structure_plan=structure_plan,
            summary=summary,
        )
        memory["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
        memory["memory_stats"] = {
            "token_budget": memory.get("memory_stats", {}).get("token_budget", settings.token_budget),
            "estimated_tokens": self.memory_service.estimate_agent_tokens(memory),
            "compression_applied": bool(memory.get("compressed_context")),
            "message_entries": len(memory.get("messages", [])),
            "memory_nodes": len(memory.get("memory_nodes", [])),
        }

    def _build_onboarding_logic_checks(self, onboarding: dict, validation_error: str = "") -> list[str]:
        checks = [
            f"通过：已完成参数步骤 {onboarding.get('progress', 0)}/{onboarding.get('total_questions', 0)}",
            "通过：记忆节点库已为本次剧本会话初始化",
        ]
        if validation_error:
            checks.insert(0, f"提示：{validation_error}")
        else:
            checks.append("通过：当前步骤参数格式合法，可继续推进下一问")
        return checks

    def _is_clarification_request(self, memory: dict, message: str) -> bool:
        stripped = message.strip()
        if not stripped:
            return False

        clarification_patterns = (
            "什么是",
            "什么叫",
            "什么意思",
            "怎么理解",
            "举个例子",
            "举例",
            "你建议",
            "我不懂",
            "不太懂",
            "不知道怎么选",
        )
        if any(pattern in stripped for pattern in clarification_patterns):
            return True
        if stripped.endswith("?") or stripped.endswith("？"):
            return True
        return False

    @staticmethod
    def _is_workspace_help_request(message: str) -> bool:
        stripped = message.strip()
        if not stripped:
            return False

        question_tokens = ("如何", "怎么", "怎样", "在哪", "哪里", "什么是", "如何用", "怎么用")
        feature_tokens = ("玩家行为记录", "行为记录", "复盘报告", "复盘草案")
        if any(token in stripped for token in feature_tokens):
            if any(token in stripped for token in question_tokens):
                return True
            if stripped.endswith("?") or stripped.endswith("？"):
                return True
        return False

    def _build_workspace_help_payload(self, memory: dict, message: str) -> dict:
        stripped = message.strip()
        current_summary = memory.get("summary", {})
        current_title = current_summary.get("title", "当前剧本")
        current_era = current_summary.get("era", "待确认")
        current_genre = current_summary.get("genre", "待确认")
        current_revision_count = int(current_summary.get("revision_count", 0))

        if "玩家行为记录" in stripped or "行为记录" in stripped:
            assistant_reply = (
                "## 玩家行为记录\n"
                "玩家行为记录在右侧“交付包 -> 复盘检查”里维护。\n\n"
                "### 推荐写法\n"
                "1. 先写时间点或幕次\n"
                "2. 再写是谁做了什么\n"
                "3. 最后写触发了什么结果\n\n"
                "### 示例\n"
                "第二幕搜证，玩家A选择隐瞒门锁日志，触发林岚的对抗反应。\n\n"
                "记录保存后，你可以直接让系统基于这些记录生成复盘草案。"
            )
            suggestions = [
                {
                    "id": "help_player_record_1",
                    "label": "生成复盘",
                    "prompt": "请基于当前剧本草案和玩家行为记录，生成一版复盘报告草案。",
                    "category": "workflow",
                },
                {
                    "id": "help_player_record_2",
                    "label": "整理时间线",
                    "prompt": "请把当前玩家行为记录整理成按时间推进的行为时间线。",
                    "category": "workflow",
                },
                {
                    "id": "help_player_record_3",
                    "label": "补行为后果",
                    "prompt": "请根据当前玩家行为记录，补写每条行为对应的分支后果和复盘点。",
                    "category": "workflow",
                },
            ]
            focus = "玩家行为记录"
        else:
            assistant_reply = (
                "## 操作说明\n"
                "如果你告诉我你具体想做哪一步，我可以直接告诉你入口和推荐操作。\n\n"
                "### 你可以这样问\n"
                "1. 如何补玩家行为记录？\n"
                "2. 如何生成复盘报告？\n"
                "3. 如何导出主持手册 / 玩家手册 / 角色剧本？"
            )
            suggestions = [
                {
                    "id": "help_general_1",
                    "label": "补行为记录",
                    "prompt": "如何补充玩家行为记录？",
                    "category": "workflow",
                },
                {
                    "id": "help_general_2",
                    "label": "生成复盘",
                    "prompt": "如何基于当前草案生成复盘报告？",
                    "category": "workflow",
                },
                {
                    "id": "help_general_3",
                    "label": "怎么导出",
                    "prompt": "如何导出当前剧本和复盘内容？",
                    "category": "workflow",
                },
            ]
            focus = "操作说明"

        return {
            "assistant_reply": assistant_reply,
            "draft_content": memory.get("draft_content", ""),
            "facts": memory.get("facts", []),
            "role_cards": memory.get("role_cards", []),
            "suggestions": self._normalize_suggestions(suggestions, default_category="workflow"),
            "summary": {
                "title": current_title,
                "era": current_era,
                "genre": current_genre,
                "focus": focus,
                "revision_count": current_revision_count,
                "template_key": memory.get("template_key", DEFAULT_TEMPLATE_KEY),
                "template_label": memory.get("template_label", "通用剧本模板"),
                "template_family": memory.get("template_family", "general"),
            },
            "help_logic_note": "本轮未改动草案。",
        }

    @staticmethod
    def _is_plan_confirmation(message: str) -> bool:
        stripped = message.strip()
        confirmation_tokens = (
            "按这个计划执行",
            "按此计划执行",
            "按这个计划生成",
            "按此计划生成",
            "确认执行",
            "开始生成",
            "就按这个来",
            "照这个计划做",
        )
        return any(token in stripped for token in confirmation_tokens)

    @staticmethod
    def _is_plan_cancellation(message: str) -> bool:
        stripped = message.strip()
        cancel_tokens = (
            "取消这轮计划",
            "取消计划",
            "先别生成",
            "暂不执行",
            "算了",
        )
        return any(token in stripped for token in cancel_tokens)

    def _build_pending_plan_payload(self, memory: dict, message: str, route: dict[str, Any]) -> dict:
        summary = memory.get("summary", {})
        blueprint = self._build_workspace_blueprint(memory, summary.get("title", "当前剧本"))
        plan_memory = dict(memory)
        plan_memory["active_generation_request"] = message.strip()
        plan_memory["intent"] = blueprint.get("intent", memory.get("intent", {}))
        plan = self.llm_service._build_agent_execution_plan(plan_memory)
        pending_plan = {
            "request": plan.get("latest_request", message.strip()),
            "focus_sections": plan.get("focus_sections", [])[:3],
            "protected_sections": plan.get("protected_sections", [])[:3],
            "impacted_roles": plan.get("impacted_roles", [])[:3],
            "impacted_stages": plan.get("impacted_stages", [])[:3],
            "impacted_clues": plan.get("impacted_clues", [])[:3],
            "excluded_roles": plan.get("excluded_roles", [])[:3],
            "excluded_stages": plan.get("excluded_stages", [])[:3],
            "excluded_clues": plan.get("excluded_clues", [])[:3],
            "success_criteria": plan.get("success_criteria", [])[:3],
            "next_action": blueprint.get("intent", {}).get("next_action", route.get("label", "")),
            "awaiting_confirmation": True,
        }
        focus_preview = "、".join(pending_plan["focus_sections"]) or "当前重点章节"
        protected_preview = "、".join(pending_plan["protected_sections"]) or "既有设定"
        role_preview = "、".join(pending_plan["impacted_roles"]) or "暂无明确角色"
        stage_preview = "、".join(pending_plan["impacted_stages"]) or "暂无明确分幕"
        clue_preview = "、".join(pending_plan["impacted_clues"]) or "暂无明确线索"
        excluded_preview = "；".join(
            [
                *[f"角色 {item}" for item in pending_plan["excluded_roles"]],
                *[f"分幕 {item}" for item in pending_plan["excluded_stages"]],
                *[f"线索 {item}" for item in pending_plan["excluded_clues"]],
            ]
        )
        success_preview = "、".join(pending_plan["success_criteria"]) or "生成完整草案"
        excluded_line = f"- 明确排除：{excluded_preview}\n" if excluded_preview else ""
        assistant_reply = (
            "## 修改计划\n"
            "我先把这轮修改计划列出来，你确认后我再动稿。\n\n"
            f"- 这轮优先处理：{focus_preview}\n"
            f"- 尽量保持不动：{protected_preview}\n"
            f"- 预计会影响：角色 {role_preview} / 分幕 {stage_preview} / 线索 {clue_preview}\n"
            f"{excluded_line}"
            f"- 完成标准：{success_preview}\n\n"
            "如果这个方向没问题，直接点“按这个方向修改”；如果你想改范围，就继续补一句要求。"
        )
        return {
            "assistant_reply": assistant_reply,
            "draft_content": memory.get("draft_content", ""),
            "facts": memory.get("facts", []),
            "role_cards": memory.get("role_cards", []),
            "suggestions": self._normalize_suggestions([
                {
                    "id": "plan_confirm",
                    "label": "按这个方向修改",
                    "prompt": "按这个计划执行",
                    "category": "workflow",
                },
                {
                    "id": "plan_adjust",
                    "label": "我再补一点",
                    "prompt": f"请调整这轮计划：{message.strip()}",
                    "category": "workflow",
                },
                {
                    "id": "plan_cancel",
                    "label": "取消这轮计划",
                    "prompt": "取消这轮计划",
                    "category": "workflow",
                },
            ], default_category="workflow"),
            "summary": {
                "title": summary.get("title", "当前剧本"),
                "era": summary.get("era", "待确认"),
                "genre": summary.get("genre", "待确认"),
                "focus": "准备修改计划",
                "revision_count": int(summary.get("revision_count", 0)),
            },
            "pending_plan": pending_plan,
            "help_logic_note": "这轮先确认修改计划，草案暂时未改动。",
            **blueprint,
        }

    def _build_plan_cancel_payload(self, memory: dict) -> dict:
        summary = memory.get("summary", {})
        return {
            "assistant_reply": "## 已取消这轮修改\n当前草案保持原样。\n\n你可以重新说新的修改方向，或者先问我这个剧本现在还缺什么。",
            "draft_content": memory.get("draft_content", ""),
            "facts": memory.get("facts", []),
            "role_cards": memory.get("role_cards", []),
            "suggestions": self._normalize_suggestions([
                {
                    "id": "cancel_followup_1",
                    "label": "重新说需求",
                    "prompt": "我想重新描述这一轮修改目标。",
                    "category": "workflow",
                },
                {
                    "id": "cancel_followup_2",
                    "label": "看看还缺什么",
                    "prompt": "现在这个剧本还缺什么？",
                    "category": "analyze",
                },
                {
                    "id": "cancel_followup_3",
                    "label": "继续改当前稿",
                    "prompt": "继续优化当前草案，但先告诉我你准备怎么改。",
                    "category": "create",
                },
            ], default_category="workflow"),
            "summary": {
                "title": summary.get("title", "当前剧本"),
                "era": summary.get("era", "待确认"),
                "genre": summary.get("genre", "待确认"),
                "focus": "已取消这轮修改",
                "revision_count": int(summary.get("revision_count", 0)),
            },
            "pending_plan": None,
            "help_logic_note": "这轮计划已取消，草案未改动。",
        }

    @classmethod
    def _maybe_build_local_title_refresh_payload(cls, memory: dict[str, Any], message: str) -> dict[str, Any] | None:
        if not cls._is_generic_title_refresh_request(message):
            return None

        summary = memory.get("summary", {})
        current_title = str(summary.get("title") or "当前剧本").strip() or "当前剧本"
        candidates = cls._build_title_refresh_candidates(memory, current_title)
        if not candidates:
            return None

        suggestions = [
            {
                "id": f"title_refresh_{index}",
                "label": candidate,
                "prompt": f"把剧本标题改成《{candidate}》",
                "category": "create",
            }
            for index, candidate in enumerate(candidates, start=1)
        ]
        suggestions.append(
            {
                "id": "title_refresh_custom",
                "label": "我再补方向",
                "prompt": "我想要一个更贴合当前主题的新标题，风格更悬疑一点",
                "category": "create",
            }
        )

        suggestion_lines = "\n".join(
            f"{index}. 《{candidate}》"
            for index, candidate in enumerate(candidates, start=1)
        )

        return {
            "assistant_reply": (
                "可以。\n"
                "这句更像是在让我先给当前剧本补几版备选标题，而不是重写整份草案。\n\n"
                "我先给你几版基于当前设定整理出的名字：\n"
                f"{suggestion_lines}\n\n"
                "你可以直接点击下方某个标题，我会把当前会话名替换成它；"
                "如果你还想更校园、更悬疑、更情感或更商业化，也可以继续补一句方向。"
            ),
            "draft_content": memory.get("draft_content", ""),
            "facts": memory.get("facts", []),
            "role_cards": memory.get("role_cards", []),
            "suggestions": suggestions[: cls.MAX_SUGGESTIONS],
            "summary": {
                "title": current_title,
                "era": summary.get("era", "待确认"),
                "genre": summary.get("genre", "待确认"),
                "focus": "标题备选",
                "revision_count": int(summary.get("revision_count", 0)),
            },
            "help_logic_note": "已返回标题备选，正文草案未改动。",
        }

    @classmethod
    def _is_generic_title_refresh_request(cls, message: str) -> bool:
        normalized = re.sub(r"\s+", "", (message or "").strip())
        if not normalized:
            return False

        explicit_request = cls._parse_simple_rename_request(normalized, "")
        if explicit_request:
            return False

        generic_tokens = (
            "换一个名字",
            "换个名字",
            "换一个标题",
            "换个标题",
            "换一个其他的名字",
            "换个别的名字",
            "改个名字",
            "改个标题",
            "起个新名字",
            "重新起名",
            "重新起个名字",
            "重新取个名字",
        )
        title_markers = ("标题", "名字", "名称", "剧本", "剧本杀")
        return any(token in normalized for token in generic_tokens) and any(marker in normalized for marker in title_markers)

    @classmethod
    def _build_title_refresh_candidates(cls, memory: dict[str, Any], current_title: str) -> list[str]:
        summary = memory.get("summary", {})
        facts_text = " ".join(str(item) for item in memory.get("facts", []))
        title_seed = cls._extract_title_seed(current_title)
        era = str(summary.get("era", "")).strip()
        genre = str(summary.get("genre", "")).strip()

        seeds: list[str] = []
        for candidate in (
            title_seed,
            "校园旧闻" if any(token in f"{era} {facts_text}" for token in ("校园", "高中", "大学")) else "",
            "旧案回响" if "旧案" in f"{current_title} {facts_text}" else "",
            "真相回廊" if "真相" in facts_text else "",
            "疑影回潮" if any(token in f"{genre} {current_title} {facts_text}" for token in ("悬疑", "推理", "硬核")) else "",
        ):
            candidate = candidate.strip()
            if candidate and candidate not in seeds:
                seeds.append(candidate)

        suffixes = ("疑局", "回响录", "真相簿", "终局夜", "暗线局", "回潮录")
        titles: list[str] = []
        current_plain = current_title.strip("《》 ")
        for seed in seeds or ["当前谜案"]:
            for suffix in suffixes:
                candidate = f"{seed}·{suffix}"
                if candidate == current_plain or candidate in titles:
                    continue
                titles.append(candidate)
                if len(titles) == 4:
                    return titles
        return titles[:4]

    @staticmethod
    def _extract_title_seed(current_title: str) -> str:
        cleaned = re.sub(r"[《》\"“”‘’]", "", (current_title or "").strip())
        if not cleaned:
            return "当前谜案"
        parts = [item.strip() for item in re.split(r"[·•|｜:：/\\\\\\-—_]", cleaned) if item.strip()]
        seed = parts[0] if parts else cleaned
        return seed[:6] if len(seed) > 6 else seed

    @classmethod
    def _should_require_plan_confirmation(cls, memory: dict, message: str, route: dict[str, Any]) -> bool:
        if route.get("response_mode") != "generate":
            return False
        if not memory.get("draft_content", "").strip():
            return False
        return cls._is_explicit_plan_request(message)

    @classmethod
    def _is_explicit_plan_request(cls, message: str) -> bool:
        stripped = message.strip()
        return any(token in stripped for token in cls.PLAN_PREVIEW_TOKENS)

    def _resolve_runtime_route(self, memory: dict, message: str) -> dict[str, Any]:
        user_messages = [item.get("content", "") for item in memory.get("messages", []) if item.get("role") == "user"]
        onboarding_answers = memory.get("onboarding", {}).get("answers", {})
        previous_summary = memory.get("summary", {})
        return self.intent_service.classify_route(
            message=message,
            user_messages=user_messages,
            onboarding_answers=onboarding_answers,
            previous_summary=previous_summary,
            has_draft=bool(memory.get("draft_content", "").strip()),
            has_role_cards=bool(memory.get("role_cards", [])),
            onboarding_completed=memory.get("onboarding", {}).get("completed", False),
        )

    def _build_analysis_payload(self, memory: dict, message: str, route: dict[str, Any]) -> dict:
        stripped = message.strip()
        current_summary = memory.get("summary", {})
        current_title = current_summary.get("title", "当前剧本")
        current_era = current_summary.get("era", "待确认")
        current_genre = current_summary.get("genre", "待确认")
        current_revision_count = int(current_summary.get("revision_count", 0))
        facts = memory.get("facts", [])
        role_cards = memory.get("role_cards", [])
        blueprint = self._build_workspace_blueprint(memory, current_title)
        intent = blueprint.get("intent", memory.get("intent", {}))
        task_queue = blueprint.get("task_queue", memory.get("task_queue", []))
        active_tasks = [item for item in task_queue if item.get("status") == "active"]
        pending_tasks = [item for item in task_queue if item.get("status") == "pending"]
        done_tasks = [item for item in task_queue if item.get("status") == "done"]

        if any(token in stripped for token in ("完成度", "进度", "还缺什么", "还差什么", "下一步")):
            completed_titles = "、".join(item.get("title", "") for item in done_tasks[:3]) or "参数采集"
            pending_titles = "、".join(item.get("title", "") for item in (active_tasks + pending_tasks)[:3]) or "继续优化成稿"
            assistant_reply = (
                f"## 当前进度\n"
                f"当前《{current_title}》已经有一版能继续迭代的基础稿了。\n\n"
                f"- 已完成：{completed_titles}\n"
                f"- 还需要优先推进：{pending_titles}\n"
                f"- 当前已锁定：{len(facts)} 条稳定设定、{len(role_cards)} 张角色卡、{len(blueprint.get('structure_plan', []))} 个流程阶段\n"
                f"- 下一步最值得做：{intent.get('next_action', '继续补齐关键章节')}"
            )
            focus = "当前进度"
        elif any(token in stripped for token in ("为什么", "为何", "逻辑", "合理", "原因")):
            fact_preview = "；".join(facts[:4]) or "当前还缺少稳定设定事实"
            active_hint = active_tasks[0].get("title", "继续细化当前重点") if active_tasks else intent.get("next_action", "继续优化")
            assistant_reply = (
                f"## 设计逻辑\n"
                f"这版《{current_title}》的逻辑主线，主要建立在已经确认的事实和流程骨架上。\n\n"
                f"当前最关键的依据是：{fact_preview}。\n"
                f"所以你现在看到的角色秘密、线索安排和结构节奏，主要都在服务“{current_summary.get('focus', '当前重点')}”，"
                f"并且是为后面的“{active_hint}”做铺垫。\n\n"
                "如果你想让我继续讲清楚，我更建议你直接点名一块：角色逻辑、线索链、结局回收，或者 DM 手册。"
            )
            focus = "设计逻辑"
        else:
            fact_preview = "；".join(facts[:3]) or "当前还没有足够的稳定设定"
            assistant_reply = (
                f"## 当前说明\n"
                f"当前《{current_title}》已经有一版基础草案了。\n\n"
                f"目前最明确的重点包括：{fact_preview}。\n"
                "如果你想继续改，我可以直接动稿；如果你想先分析，我也可以继续帮你讲完成度、角色逻辑、线索结构或导出交付。"
            )
            focus = "分析说明"

        payload = {
            "assistant_reply": assistant_reply,
            "draft_content": memory.get("draft_content", ""),
            "facts": facts,
            "role_cards": role_cards,
            "summary": {
                "title": current_title,
                "era": current_era,
                "genre": current_genre,
                "focus": focus,
                "revision_count": current_revision_count,
            },
            "logic_checks": memory.get("logic_checks", []),
            "memory_summary": self.memory_service.build_agent_memory_summary(memory),
            "memory_stats": memory.get("memory_stats", {}),
            "help_logic_note": "本轮主要是在解释和分析，草案未改动。",
            **blueprint,
        }
        analysis_suggestions = [
            {
                "id": "analysis_followup_progress",
                "label": "看看还缺什么",
                "prompt": "现在这个剧本还缺什么？",
                "category": "analyze",
            },
            {
                "id": "analysis_followup_logic",
                "label": "讲角色逻辑",
                "prompt": "请解释当前角色逻辑和动机链，重点说清楚谁在隐瞒什么。",
                "category": "analyze",
            },
            {
                "id": "analysis_followup_create",
                "label": "直接开始改",
                "prompt": "请直接继续改稿，优先处理当前最影响体验的问题。",
                "category": "create",
            },
        ]
        payload["suggestions"] = self._normalize_suggestions(
            [*analysis_suggestions, *self._build_contextual_suggestions(memory, payload, blueprint)]
        )
        return payload

    def _find_question(self, question_id: str, template_config: dict[str, Any] | None = None) -> dict | None:
        resolved_template = template_config or self.template_config_service.get_template(DEFAULT_TEMPLATE_KEY)
        question_flow = self._get_question_flow(resolved_template)
        return next((question for question in question_flow if question["id"] == question_id), None)

    def _empty_onboarding_state(self, template_key: str = DEFAULT_TEMPLATE_KEY) -> dict:
        template_config = self.template_config_service.get_template(template_key)
        return {
            "enabled": True,
            "completed": False,
            "progress": 0,
            "total_questions": len(self._get_question_flow(template_config)),
            "template_key": str(template_config.get("template_key") or DEFAULT_TEMPLATE_KEY),
            "current_question_id": "",
            "current_question": "",
            "current_placeholder": "",
            "answers": {},
        }

    def _load_agent_memory(self, owner_id: int, session_id: str) -> tuple[str, dict]:
        db = SessionLocal()
        try:
            record = self._get_record(db, owner_id, session_id)
            memory = self.memory_service.load(record)
            if memory.get("mode") != "agent":
                raise HTTPException(status_code=400, detail="该会话不是智能体对话会话。")
            self._sanitize_memory_role_cards(memory)
            return record.title, memory
        finally:
            db.close()

    async def _yield_stream_chunks(self, payload: dict):
        for chunk in self._chunk_text(payload["assistant_reply"].strip(), 80):
            yield self._sse_event("assistant_delta", {"content": chunk})

        draft_content = payload["draft_content"].strip()
        if draft_content:
            for chunk in self._chunk_text(draft_content, 180):
                yield self._sse_event("draft_delta", {"content": chunk})

    def _apply_payload_to_memory(
        self,
        record_title: str,
        memory: dict,
        payload: dict,
        provider: str,
        *,
        append_assistant_message: bool = True,
    ) -> tuple[dict, str]:
        summary = payload.get("summary", {})
        if provider == "guide":
            summary["revision_count"] = int(memory.get("onboarding", {}).get("progress", 0))
        else:
            summary["revision_count"] = int(memory.get("summary", {}).get("revision_count", 0)) + 1

        assistant_message = None
        if append_assistant_message:
            assistant_message = self._append_message(memory, "assistant", payload["assistant_reply"].strip())
        memory["suggestions"] = self._ensure_three_suggestions(payload.get("suggestions", []))
        memory["facts"] = payload.get("facts", [])
        memory["role_cards"] = self.llm_service.sanitize_role_cards(
            payload.get("role_cards", memory.get("role_cards", [])),
            genre=self._memory_role_card_genre(memory, summary),
            conflict=self._memory_role_card_conflict(memory),
        )
        if payload.get("behavior_records") is not None:
            memory["behavior_records"] = payload.get("behavior_records", memory.get("behavior_records", []))
        memory["pending_plan"] = None
        memory.pop("active_generation_request", None)
        memory["draft_content"] = self.llm_service.sync_draft_role_card_section(
            payload["draft_content"].strip(),
            memory["role_cards"],
        )
        memory["summary"] = summary
        memory["provider"] = provider
        memory["logic_checks"] = payload.get("logic_checks", memory.get("logic_checks", []))
        memory["memory_summary"] = payload.get("memory_summary", memory.get("memory_summary", []))
        if payload.get("memory_stats"):
            memory["memory_stats"] = payload["memory_stats"]
        if payload.get("generated_role_scripts") is not None:
            memory["role_scripts"] = copy.deepcopy(payload.get("generated_role_scripts", []))
        if payload.get("generated_role_script_prompt_blueprints") is not None:
            memory["role_script_prompt_blueprints"] = copy.deepcopy(payload.get("generated_role_script_prompt_blueprints", []))

        resolved_title = summary.get("title") or record_title
        blueprint = self._build_workspace_blueprint(memory, resolved_title)
        memory["intent"] = payload.get("intent", blueprint["intent"])
        memory["task_queue"] = payload.get("task_queue", blueprint["task_queue"])
        memory["structure_plan"] = payload.get("structure_plan", blueprint["structure_plan"])
        memory["dm_script"] = payload.get("dm_script", blueprint["dm_script"])
        memory["output_outline"] = payload.get("output_outline", blueprint["output_outline"])
        if provider != "guide":
            memory["suggestions"] = self._build_contextual_suggestions(memory, payload, blueprint)
            self._append_memory_node(
                memory,
                "generation",
                summary.get("focus", "生成草案"),
                summary.get("title", resolved_title),
                tags=["draft", summary.get("genre", ""), summary.get("era", "")],
            )
            self._refresh_agent_memory_layers(memory)
        if assistant_message is not None:
            self._bind_snapshot_to_message(memory, resolved_title, assistant_message, provider)
        return memory, resolved_title

    def _memory_role_card_genre(self, memory: dict, summary: dict | None = None) -> str:
        answers = memory.get("onboarding", {}).get("answers", {})
        current_summary = summary or memory.get("summary", {})
        return str(answers.get("genre") or current_summary.get("genre") or "现代悬疑").strip()

    def _memory_role_card_conflict(self, memory: dict) -> str:
        answers = memory.get("onboarding", {}).get("answers", {})
        return str(answers.get("conflict") or "隐藏在关系中的真相").strip()

    def _sanitize_memory_role_cards(self, memory: dict) -> bool:
        original = memory.get("role_cards", [])
        genre = self._memory_role_card_genre(memory)
        conflict = self._memory_role_card_conflict(memory)
        sanitized = self.llm_service.sanitize_role_cards(
            original,
            genre=genre,
            conflict=conflict,
        )
        sanitized = self._recover_role_cards_from_revision_history(memory, sanitized, genre, conflict)
        if sanitized == original:
            return False
        memory["role_cards"] = sanitized
        memory["draft_content"] = self.llm_service.sync_draft_role_card_section(
            memory.get("draft_content", ""),
            sanitized,
        )
        return True

    def _recover_role_cards_from_revision_history(
        self,
        memory: dict,
        role_cards: list[dict[str, Any]],
        genre: str,
        conflict: str,
    ) -> list[dict[str, Any]]:
        history = memory.get("revision_history", [])
        if not isinstance(history, list):
            return role_cards
        current_count = len(role_cards)
        if current_count <= 0:
            return role_cards

        for snapshot in reversed(history):
            if not isinstance(snapshot, dict):
                continue
            candidate = self.llm_service.sanitize_role_cards(
                snapshot.get("role_cards", []),
                genre=genre,
                conflict=conflict,
            )
            if len(candidate) <= current_count:
                continue
            answers = memory.get("onboarding", {}).get("answers", {})
            rebuilt = self.llm_service._normalize_role_cards(
                role_cards,
                {"role_cards": candidate},
                answers.get("players") or f"{len(candidate)} 人",
                genre,
                answers.get("protagonist") or "关键角色",
                conflict,
            )
            if len(rebuilt) >= len(candidate):
                return rebuilt
        return role_cards

    def _refresh_workspace_blueprint(self, memory: dict, record_title: str) -> None:
        memory.update(self._build_workspace_blueprint(memory, record_title))

    @staticmethod
    def _can_fill_imported_answer(value: Any) -> bool:
        text = str(value or "").strip()
        return not text or text in {"待确认", "待补充"}

    @staticmethod
    def _can_fill_imported_title(title: str) -> bool:
        normalized = str(title or "").strip()
        return not normalized or normalized in {"待命名剧本", "未命名剧本"}

    @staticmethod
    def _merge_text_items(*collections: Any) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for collection in collections:
            if not isinstance(collection, list):
                continue
            for item in collection:
                text = str(item or "").strip()
                if not text or text in seen:
                    continue
                merged.append(text)
                seen.add(text)
        return merged[:16]

    def _merge_imported_role_cards(self, existing: list[dict], imported: list[dict], replace_names: set[str] | None = None) -> list[dict]:
        if not imported:
            return existing
        if not existing:
            return imported
        replace_names = replace_names or set()
        merged = copy.deepcopy(existing)
        existing_name_to_index = {
            str(item.get("name") or "").strip(): index
            for index, item in enumerate(merged)
            if str(item.get("name") or "").strip()
        }
        for item in imported:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            if name in existing_name_to_index:
                if name in replace_names:
                    merged[existing_name_to_index[name]] = item
                continue
            merged.append(item)
            existing_name_to_index[name] = len(merged) - 1
        return merged

    @staticmethod
    def _build_concept_import_reply(filename: str, analysis: dict[str, Any], recognized_fields: list[str]) -> str:
        lines = [f"已导入构想文件《{filename}》。"]
        summary = str(analysis.get("summary") or "").strip()
        if summary:
            lines.append(f"我从文件里提炼到的核心描述：{summary}")
        if recognized_fields:
            lines.append(f"本次已回填：{'；'.join(recognized_fields[:8])}")
        if analysis.get("role_cards"):
            lines.append(f"额外识别到 {len(analysis['role_cards'])} 个候选角色设定。")
        if analysis.get("facts"):
            lines.append("导入内容已写入事实清单，你可以继续补充或直接开始生成。")
        return "\n".join(lines)

    async def _generate_role_scripts_for_payload(
        self,
        record_title: str,
        memory: dict,
        payload: dict,
        provider: str,
    ) -> dict[str, list[dict[str, Any]]]:
        if provider == "guide":
            return {
                "scripts": copy.deepcopy(memory.get("role_scripts", [])),
                "prompt_blueprints": copy.deepcopy(memory.get("role_script_prompt_blueprints", [])),
            }
        preview_memory = copy.deepcopy(memory)
        preview_payload = copy.deepcopy(payload)
        preview_memory, resolved_title = self._apply_payload_to_memory(
            record_title,
            preview_memory,
            preview_payload,
            provider,
            append_assistant_message=False,
        )
        scripts = await self.role_script_service.generate_and_store(resolved_title, preview_memory, self.llm_service)
        return {
            "scripts": [item.model_dump() for item in scripts],
            "prompt_blueprints": copy.deepcopy(preview_memory.get("role_script_prompt_blueprints", [])),
        }

    async def _attach_generated_role_scripts_to_payload(
        self,
        record_title: str,
        memory: dict,
        payload: dict,
        provider: str,
    ) -> None:
        generated = await self._generate_role_scripts_for_payload(record_title, memory, payload, provider)
        payload["generated_role_scripts"] = generated["scripts"]
        payload["generated_role_script_prompt_blueprints"] = generated["prompt_blueprints"]

    def _maybe_build_local_edit_payload(self, memory: dict, message: str) -> dict[str, Any] | None:
        rename_request = self._parse_simple_rename_request(
            message,
            str(memory.get("summary", {}).get("title") or ""),
        )
        if not rename_request:
            return None

        old_name = rename_request["old"]
        new_name = rename_request["new"]
        scope = rename_request["scope"]
        if not old_name or not new_name or old_name == new_name:
            return None

        draft_content, draft_count = self._replace_text_in_value(memory.get("draft_content", ""), old_name, new_name)
        facts, fact_count = self._replace_text_in_value(memory.get("facts", []), old_name, new_name)
        role_cards, role_card_count = self._replace_text_in_value(memory.get("role_cards", []), old_name, new_name)
        role_scripts, role_script_count = self._replace_text_in_value(memory.get("role_scripts", []), old_name, new_name)
        behavior_records, behavior_count = self._replace_text_in_value(
            memory.get("behavior_records", []),
            old_name,
            new_name,
        )
        summary, summary_count = self._replace_text_in_value(memory.get("summary", {}), old_name, new_name)
        if scope == "title":
            summary["title"] = new_name

        replacement_count = draft_count + fact_count + role_card_count + role_script_count + behavior_count + summary_count
        if replacement_count <= 0 and scope != "title":
            return None

        focus = "本地重命名"
        summary["focus"] = focus
        logic_checks = [
            f"通过：本轮本地重命名已将“{old_name}”替换为“{new_name}”",
            *[item for item in memory.get("logic_checks", []) if f"“{old_name}”" not in item],
        ]
        return {
            "assistant_reply": (
                f"已直接在当前草案里把“{old_name}”改成“{new_name}”。"
                "这次走的是本地确定性修改，没有依赖大模型重写整份稿件。"
            ),
            "draft_content": draft_content,
            "facts": facts,
            "role_cards": role_cards,
            "generated_role_scripts": role_scripts,
            "behavior_records": behavior_records,
            "summary": {
                "title": summary.get("title", memory.get("summary", {}).get("title", "待命名剧本")),
                "era": summary.get("era", memory.get("summary", {}).get("era", "待确认")),
                "genre": summary.get("genre", memory.get("summary", {}).get("genre", "待确认")),
                "focus": focus,
            },
            "logic_checks": logic_checks,
            "memory_summary": memory.get("memory_summary", []),
        }

    @classmethod
    def _parse_simple_rename_request(cls, message: str, current_title: str) -> dict[str, str] | None:
        cleaned_message = re.sub(r"[。！？!?]+$", "", message.strip())
        heuristic_request = cls._parse_simple_rename_request_heuristic(cleaned_message, current_title)
        if heuristic_request:
            return heuristic_request
        for pattern in cls.TITLE_RENAME_PATTERNS:
            match = pattern.fullmatch(cleaned_message)
            if not match:
                continue
            new_name = cls._clean_rename_token(match.group("new"))
            old_name = cls._clean_rename_token(match.groupdict().get("old", "")) or current_title.strip()
            if old_name and new_name:
                return {"scope": "title", "old": old_name, "new": new_name}

        for pattern in cls.SIMPLE_RENAME_PATTERNS:
            match = pattern.fullmatch(cleaned_message)
            if not match:
                continue
            old_name = cls._clean_rename_token(match.group("old"))
            new_name = cls._clean_rename_token(match.group("new"))
            if old_name and new_name:
                return {"scope": "entity", "old": old_name, "new": new_name}
        return None

    @staticmethod
    def _clean_rename_token(token: str) -> str:
        cleaned = token.strip()
        cleaned = cleaned.strip("“”\"'《》【】[]()（）。！？!?")
        cleaned = re.sub(r"^(可以把|可以将|请把|请将|麻烦把|麻烦将|把|将)", "", cleaned)
        cleaned = re.sub(r"^(叫做|命名为)", "", cleaned)
        cleaned = re.sub(r"(吧|一下|试试|可以吗|可以么|好吗|行吗|吗|么)$", "", cleaned)
        cleaned = cleaned.strip("“”\"'《》【】[]()（）。！？!?")
        return cleaned.strip()

    @classmethod
    def _parse_simple_rename_request_heuristic(cls, message: str, current_title: str) -> dict[str, str] | None:
        normalized = message.strip()
        if not normalized:
            return None

        rename_keywords = (
            "的名字命名为",
            "名字命名为",
            "的名字改成",
            "名字改成",
            "改名为",
            "改名成",
            "改名叫",
            "改成",
            "改为",
            "换成",
            "换成叫",
            "叫做",
            "命名为",
        )
        title_markers = ("标题", "剧本", "名字", "名称", "题目")
        leading_tokens = ("可以把", "可以将", "请把", "请将", "麻烦把", "麻烦将", "把", "将", "把这部", "把这个")

        keyword = next((item for item in rename_keywords if item in normalized), "")
        if not keyword:
            return None

        before, after = normalized.split(keyword, 1)
        new_name = cls._clean_rename_token(after)
        if not new_name:
            return None

        subject = before.strip()
        for token in leading_tokens:
            if subject.startswith(token):
                subject = subject[len(token):].strip()
                break

        scope = "title" if any(marker in subject for marker in title_markers) else "entity"
        for marker in title_markers:
            subject = subject.replace(marker, " ")
        old_name = cls._clean_rename_token(subject) or current_title.strip()
        if not old_name or not new_name or old_name == new_name:
            return None
        if "《" in normalized and "》" in normalized and old_name == current_title.strip():
            scope = "title"
        return {"scope": scope, "old": old_name, "new": new_name}

    @classmethod
    def _replace_text_in_value(cls, value: Any, old_text: str, new_text: str) -> tuple[Any, int]:
        if isinstance(value, str):
            replacement_count = value.count(old_text)
            if replacement_count <= 0:
                return value, 0
            return value.replace(old_text, new_text), replacement_count
        if isinstance(value, list):
            total_count = 0
            replaced_items = []
            for item in value:
                replaced_item, item_count = cls._replace_text_in_value(item, old_text, new_text)
                replaced_items.append(replaced_item)
                total_count += item_count
            return replaced_items, total_count
        if isinstance(value, dict):
            total_count = 0
            replaced_items: dict[Any, Any] = {}
            for key, item in value.items():
                replaced_item, item_count = cls._replace_text_in_value(item, old_text, new_text)
                replaced_items[key] = replaced_item
                total_count += item_count
            return replaced_items, total_count
        return value, 0

    def _apply_help_payload_to_memory(
        self,
        record_title: str,
        memory: dict,
        payload: dict,
        provider: str,
    ) -> tuple[dict, str]:
        assistant_reply = payload.get("assistant_reply", "").strip()
        if assistant_reply:
            self._append_message(memory, "assistant", assistant_reply)

        if payload.get("suggestions") is not None:
            memory["suggestions"] = self._ensure_three_suggestions(payload.get("suggestions", []))
        if payload.get("facts") is not None:
            memory["facts"] = payload.get("facts", memory.get("facts", []))
        if payload.get("draft_content") is not None:
            memory["draft_content"] = self.llm_service.sync_draft_role_card_section(
                payload.get("draft_content", "").strip(),
                memory.get("role_cards", []),
            )
        if payload.get("role_cards") is not None:
            memory["role_cards"] = payload.get("role_cards", memory.get("role_cards", []))
        if "pending_plan" in payload:
            memory["pending_plan"] = payload.get("pending_plan")
        else:
            memory["pending_plan"] = memory.get("pending_plan")
        memory.pop("active_generation_request", None)

        current_summary = memory.get("summary", {})
        next_summary = payload.get("summary", {})
        resolved_title = next_summary.get("title") or current_summary.get("title") or record_title
        memory["summary"] = {
            "title": resolved_title,
            "era": next_summary.get("era", current_summary.get("era", "待确认")),
            "genre": next_summary.get("genre", current_summary.get("genre", "待确认")),
            "focus": next_summary.get("focus", current_summary.get("focus", "操作说明")),
            "revision_count": int(current_summary.get("revision_count", 0)),
        }
        memory["provider"] = provider
        self._append_memory_node(
            memory,
            "workspace_help",
            memory["summary"].get("focus", "操作说明"),
            assistant_reply[:160],
            tags=["help"],
        )
        self._refresh_workspace_blueprint(memory, resolved_title)
        self._refresh_agent_memory_layers(memory)

        if payload.get("help_logic_note"):
            existing_checks = memory.get("logic_checks", [])
            note = payload["help_logic_note"]
            if note not in existing_checks:
                memory["logic_checks"] = [note, *existing_checks]
        memory["memory_summary"] = self.memory_service.build_agent_memory_summary(memory)
        return memory, resolved_title

    def _build_workspace_blueprint(self, memory: dict, record_title: str) -> dict:
        messages = memory.get("messages", [])
        user_messages = [item["content"] for item in messages if item.get("role") == "user" and item.get("content")]
        latest_message = user_messages[-1] if user_messages else ""
        onboarding = memory.get("onboarding", self._empty_onboarding_state())
        onboarding_answers = onboarding.get("answers", {})
        summary = memory.get("summary", {})
        role_cards = memory.get("role_cards", [])
        has_draft = bool(memory.get("draft_content", "").strip())
        intent, task_queue = self.intent_service.analyze(
            message=latest_message,
            user_messages=user_messages,
            onboarding_answers=onboarding_answers,
            previous_summary=summary,
            has_draft=has_draft,
            has_role_cards=bool(role_cards),
            onboarding_completed=onboarding.get("completed", False),
        )
        params = intent.get("params", {})
        genre = params.get("type") or summary.get("genre", "现代悬疑")
        players = params.get("people") or onboarding_answers.get("players", "6 人")
        duration = params.get("time") or "120 分钟"
        conflict = onboarding_answers.get("conflict") or params.get("theme") or "关系冲突与真相揭示"
        goal = onboarding_answers.get("goal") or "揭开真相"
        structure_plan = self.structure_engine.build_flow(
            genre=genre,
            players=players,
            duration=duration,
            scene=params.get("scene") or "封闭主场景 + 多点互动空间",
            conflict=conflict,
            goal=goal,
            template_key=str(summary.get("template_key") or memory.get("template_key") or ""),
        )
        dm_script = self.dm_service.build_cues(
            stages=structure_plan,
            genre=genre,
            conflict=conflict,
            goal=goal,
            role_cards=role_cards,
            template_key=str(summary.get("template_key") or memory.get("template_key") or ""),
        )
        output_outline = self.render_service.build_output_outline(
            len(role_cards) if role_cards else self._resolve_expected_role_count(players),
        )
        return {
            "intent": intent,
            "task_queue": task_queue,
            "structure_plan": structure_plan,
            "dm_script": dm_script,
            "output_outline": output_outline,
        }

    @staticmethod
    def _resolve_expected_role_count(players: str) -> int:
        numbers = [int(item) for item in re.findall(r"\d+", players)]
        if not numbers:
            return 6
        return max(3, min(numbers[0], 10))

    def _persist_new_session(self, session_id: str, owner_id: int, record_title: str, memory: dict, payload: dict, provider: str) -> AgentStateResponse:
        db = SessionLocal()
        try:
            memory, resolved_title = self._apply_payload_to_memory(record_title, memory, payload, provider)
            summary = memory.get("summary", {})
            record = ScriptSession(
                id=session_id,
                owner_id=owner_id,
                title=resolved_title,
                world_config=summary,
                memory=memory,
                latest_segment=memory["draft_content"],
                latest_branches=memory["suggestions"],
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            self.memory_service.save(record, memory)
            return self._build_state(record, memory, provider)
        finally:
            db.close()

    def _persist_existing_session(self, session_id: str, owner_id: int, record_title: str, memory: dict, payload: dict, provider: str) -> AgentStateResponse:
        db = SessionLocal()
        try:
            record = self._get_record(db, owner_id, session_id)
            memory, resolved_title = self._apply_payload_to_memory(record_title, memory, payload, provider)
            summary = memory.get("summary", {})
            record.title = resolved_title
            record.latest_segment = memory["draft_content"]
            record.latest_branches = memory["suggestions"]
            record.world_config = summary
            record.memory = memory
            db.add(record)
            db.commit()
            db.refresh(record)
            self.memory_service.save(record, memory)
            return self._build_state(record, memory, provider)
        finally:
            db.close()

    def _persist_new_help_session(self, session_id: str, owner_id: int, record_title: str, memory: dict, payload: dict, provider: str) -> AgentStateResponse:
        db = SessionLocal()
        try:
            memory, resolved_title = self._apply_help_payload_to_memory(record_title, memory, payload, provider)
            record = ScriptSession(
                id=session_id,
                owner_id=owner_id,
                title=resolved_title,
                world_config=memory.get("summary", {}),
                memory=memory,
                latest_segment=memory.get("draft_content", ""),
                latest_branches=memory.get("suggestions", []),
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            self.memory_service.save(record, memory)
            return self._build_state(record, memory, provider)
        finally:
            db.close()

    def _persist_existing_help_session(self, session_id: str, owner_id: int, record_title: str, memory: dict, payload: dict, provider: str) -> AgentStateResponse:
        db = SessionLocal()
        try:
            record = self._get_record(db, owner_id, session_id)
            memory, resolved_title = self._apply_help_payload_to_memory(record_title, memory, payload, provider)
            record.title = resolved_title
            record.latest_segment = memory.get("draft_content", "")
            record.latest_branches = memory.get("suggestions", [])
            record.world_config = memory.get("summary", {})
            record.memory = memory
            db.add(record)
            db.commit()
            db.refresh(record)
            self.memory_service.save(record, memory)
            return self._build_state(record, memory, provider)
        finally:
            db.close()

    @staticmethod
    def _create_message(
        role: str,
        content: str,
        *,
        revision_id: str = "",
        can_rollback: bool = False,
        rollback_label: str = "",
    ) -> dict:
        return {
            "id": str(uuid4()),
            "role": role,
            "content": content,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "revision_id": revision_id,
            "can_rollback": can_rollback,
            "rollback_label": rollback_label,
        }

    def _append_message(self, memory: dict, role: str, content: str) -> dict:
        message = self._create_message(role, content)
        memory.setdefault("messages", []).append(message)
        return message

    def _bind_snapshot_to_message(self, memory: dict, record_title: str, message: dict, source: str) -> None:
        snapshot = self._capture_revision_snapshot(memory, record_title, source)
        if not snapshot:
            return
        message["revision_id"] = snapshot["revision_id"]
        message["can_rollback"] = True
        message["rollback_label"] = snapshot["label"]

    def _build_messages_for_state(self, memory: dict) -> list[dict]:
        messages = copy.deepcopy(memory.get("messages", []))
        last_user_message = ""
        for message in messages:
            if message.get("role") == "user" and message.get("content"):
                last_user_message = str(message.get("content", "")).strip()
                continue
            if message.get("role") != "assistant" or not message.get("can_rollback") or not message.get("revision_id"):
                continue

            snapshot = self._find_revision_snapshot(memory, message.get("revision_id", ""))
            current_label = str(message.get("rollback_label", "")).strip()
            source = str(snapshot.get("source", "")) if snapshot else ""
            preferred_label = self._build_revision_label(
                {
                    **memory,
                    "messages": [*memory.get("messages", []), {"role": "user", "content": last_user_message}],
                    "summary": snapshot.get("summary", memory.get("summary", {})) if snapshot else memory.get("summary", {}),
                },
                source or "history",
            )
            if not current_label or current_label in {"历史版本", "生成完整世界观草案"}:
                message["rollback_label"] = preferred_label
        return messages

    def _build_revision_label(self, memory: dict, source: str) -> str:
        source_map = {
            "manual_edit": "手动编辑稿件",
            "role_card_edit": "手动编辑角色卡",
            "facts_edit": "手动编辑设定",
            "behavior_record_edit": "记录复盘行为",
            "onboarding_control": "调整世界观步骤",
        }
        if source in source_map:
            return source_map[source]

        latest_user_message = next(
            (
                item.get("content", "").strip()
                for item in reversed(memory.get("messages", []))
                if item.get("role") == "user" and item.get("content")
            ),
            "",
        )
        if latest_user_message:
            return self._short_text(latest_user_message, 18)

        focus = str(memory.get("summary", {}).get("focus", "")).strip()
        if focus:
            return self._short_text(focus, 18)
        return "历史版本"

    def _capture_revision_snapshot(self, memory: dict, record_title: str, source: str) -> dict | None:
        if not memory.get("draft_content", "").strip() and not memory.get("role_cards") and not memory.get("behavior_records"):
            return None

        snapshot = {
            "revision_id": str(uuid4()),
            "title": record_title,
            "message_count": len(memory.get("messages", [])),
            "draft_content": memory.get("draft_content", ""),
            "suggestions": copy.deepcopy(memory.get("suggestions", [])),
            "facts": copy.deepcopy(memory.get("facts", [])),
            "role_cards": copy.deepcopy(memory.get("role_cards", [])),
            "role_scripts": copy.deepcopy(memory.get("role_scripts", [])),
            "role_script_prompt_blueprints": copy.deepcopy(memory.get("role_script_prompt_blueprints", [])),
            "behavior_records": copy.deepcopy(memory.get("behavior_records", [])),
            "onboarding": copy.deepcopy(memory.get("onboarding", self._empty_onboarding_state())),
            "summary": copy.deepcopy(memory.get("summary", {})),
            "intent": copy.deepcopy(memory.get("intent", {})),
            "task_queue": copy.deepcopy(memory.get("task_queue", [])),
            "structure_plan": copy.deepcopy(memory.get("structure_plan", [])),
            "dm_script": copy.deepcopy(memory.get("dm_script", [])),
            "output_outline": copy.deepcopy(memory.get("output_outline", [])),
            "logic_checks": copy.deepcopy(memory.get("logic_checks", [])),
            "memory_summary": copy.deepcopy(memory.get("memory_summary", [])),
            "memory_stats": copy.deepcopy(memory.get("memory_stats", {})),
            "memory_nodes": copy.deepcopy(memory.get("memory_nodes", [])),
            "generation_rules": copy.deepcopy(memory.get("generation_rules", [])),
            "pending_plan": copy.deepcopy(memory.get("pending_plan")),
            "label": self._build_revision_label(memory, source),
            "source": source,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        history = memory.setdefault("revision_history", [])
        history.append(snapshot)
        if len(history) > 30:
            del history[:-30]
        memory["active_revision_id"] = snapshot["revision_id"]
        return snapshot

    @staticmethod
    def _find_revision_snapshot(memory: dict, revision_id: str) -> dict | None:
        return next(
            (item for item in memory.get("revision_history", []) if item.get("revision_id") == revision_id),
            None,
        )

    def _restore_snapshot_to_memory(self, memory: dict, snapshot: dict) -> None:
        current_messages = memory.get("messages", [])
        message_count = min(snapshot.get("message_count", len(current_messages)), len(current_messages))
        memory["messages"] = copy.deepcopy(current_messages[:message_count])
        memory["draft_content"] = snapshot.get("draft_content", "")
        memory["suggestions"] = copy.deepcopy(snapshot.get("suggestions", []))
        memory["facts"] = copy.deepcopy(snapshot.get("facts", []))
        memory["role_cards"] = copy.deepcopy(snapshot.get("role_cards", []))
        memory["role_scripts"] = copy.deepcopy(snapshot.get("role_scripts", []))
        memory["role_script_prompt_blueprints"] = copy.deepcopy(snapshot.get("role_script_prompt_blueprints", []))
        memory["behavior_records"] = copy.deepcopy(snapshot.get("behavior_records", []))
        memory["onboarding"] = copy.deepcopy(snapshot.get("onboarding", self._empty_onboarding_state()))
        memory["summary"] = copy.deepcopy(snapshot.get("summary", {}))
        memory["intent"] = copy.deepcopy(snapshot.get("intent", {}))
        memory["task_queue"] = copy.deepcopy(snapshot.get("task_queue", []))
        memory["structure_plan"] = copy.deepcopy(snapshot.get("structure_plan", []))
        memory["dm_script"] = copy.deepcopy(snapshot.get("dm_script", []))
        memory["output_outline"] = copy.deepcopy(snapshot.get("output_outline", []))
        memory["logic_checks"] = copy.deepcopy(snapshot.get("logic_checks", []))
        memory["memory_summary"] = copy.deepcopy(snapshot.get("memory_summary", []))
        memory["memory_stats"] = copy.deepcopy(snapshot.get("memory_stats", {}))
        memory["memory_nodes"] = copy.deepcopy(snapshot.get("memory_nodes", []))
        memory["generation_rules"] = copy.deepcopy(snapshot.get("generation_rules", []))
        memory["pending_plan"] = copy.deepcopy(snapshot.get("pending_plan"))
        memory["active_revision_id"] = snapshot.get("revision_id", "")

    @staticmethod
    def _get_record(db: Session, owner_id: int, session_id: str) -> ScriptSession:
        record = db.get(ScriptSession, session_id)
        if not record or record.owner_id != owner_id:
            raise HTTPException(status_code=404, detail="未找到对应智能体会话。")
        return record

    @staticmethod
    def _sse_event(event: str, payload: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list[str]:
        if not text:
            return []
        return [text[index:index + chunk_size] for index in range(0, len(text), chunk_size)]
