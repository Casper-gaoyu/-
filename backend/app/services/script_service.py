from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.db_models import ScriptSession
from app.models.schemas import CreateSessionResponse, ReviewResponse, ScriptStateResponse, WorldConfig
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.rule_engine import RuleEngine

STAGES = ["发本导入", "破冰立人设", "第一轮搜证", "私聊交易", "集中公聊", "终局还原"]


class ScriptService:
    def __init__(self) -> None:
        self.rule_engine = RuleEngine()
        self.memory_service = MemoryService()
        self.llm_service = LLMService(self.rule_engine)

    async def create_session(self, db: Session, owner_id: int, payload: WorldConfig) -> CreateSessionResponse:
        record = ScriptSession(
            id=str(uuid4()),
            owner_id=owner_id,
            title=payload.title,
            world_config=payload.model_dump(),
            memory={},
            latest_branches=[],
        )
        db.add(record)
        db.flush()

        memory = self.memory_service.build_initial_memory(record.world_config)
        self.memory_service.save(record, memory)
        segment = await self._generate_next_segment(db, record, memory)
        db.commit()
        db.refresh(record)

        return CreateSessionResponse(session_id=record.id, title=record.title, segment=segment)

    def get_session_state(self, db: Session, owner_id: int, session_id: str) -> ScriptStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        stage_index = max(0, min(len(memory.get("history", [])) - 1, len(STAGES) - 1))
        stage = STAGES[stage_index]
        content = record.latest_segment or ""
        provider = self.llm_service.get_runtime_profile()["recommended_provider"]
        npc_dialogues = memory.get("last_npc_dialogues", [])
        memory_stats = memory.get("last_memory_stats") or {
            "token_budget": settings.token_budget,
            "estimated_tokens": self.memory_service.estimate_tokens(memory),
            "compression_applied": False,
            "history_entries": len(memory.get("history", [])),
        }
        return ScriptStateResponse(
            session_id=record.id,
            title=record.title,
            stage=stage,
            content=content,
            branches=record.latest_branches or [],
            npc_dialogues=npc_dialogues,
            memory_summary=self._build_memory_summary(memory),
            memory_stats=memory_stats,
            logic_checks=self.rule_engine.build_logic_checks(memory, content) if content else [],
            is_finished=len(memory.get("history", [])) >= len(STAGES),
            provider=provider,
        )

    async def advance_session(self, db: Session, owner_id: int, session_id: str, choice_index: int | None) -> ScriptStateResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)

        if len(memory.get("history", [])) >= len(STAGES):
            raise HTTPException(status_code=400, detail="剧情已经到达结局，请直接生成复盘。")

        branches = memory.get("last_branches") or record.latest_branches or []
        if branches:
            index = choice_index if choice_index is not None else 0
            if index < 0 or index >= len(branches):
                raise HTTPException(status_code=400, detail="分支索引超出范围。")
            chosen_branch = branches[index]
            choice_text = f"{chosen_branch['title']}：{chosen_branch['description']}"
            memory.setdefault("choices", []).append(choice_text)
            for fact in self.rule_engine.derive_facts_from_choice(choice_text):
                if fact not in memory["facts"]:
                    memory["facts"].append(fact)

        segment = await self._generate_next_segment(db, record, memory)
        db.commit()
        db.refresh(record)
        return segment

    async def generate_review(self, db: Session, owner_id: int, session_id: str) -> ReviewResponse:
        record = self._get_record(db, owner_id, session_id)
        memory = self.memory_service.load(record)
        review_content, provider = await self.llm_service.generate_review(record.world_config, memory)
        record.review_content = review_content
        db.add(record)
        db.commit()
        db.refresh(record)
        return ReviewResponse(
            session_id=record.id,
            title=record.title,
            review_content=review_content,
            provider=provider,
        )

    async def _generate_next_segment(self, db: Session, record: ScriptSession, memory: dict) -> ScriptStateResponse:
        stage_index = min(len(memory.get("history", [])), len(STAGES) - 1)
        stage = STAGES[stage_index]
        generation_memory, memory_stats = self.memory_service.prepare_generation_memory(memory)
        segment_payload, provider = await self.llm_service.generate_segment(record.world_config, generation_memory, stage)
        content = segment_payload["content"].strip()
        violations = self.rule_engine.check_logic_consistency(memory, content)
        npc_dialogues = segment_payload.get("npc_dialogues", [])

        if violations:
            content = (
                f"【{stage}】由于检测到潜在逻辑冲突，系统已切换到安全生成模式。"
                "当前剧情将保持世界观与既有事实一致，并优先延续上一轮玩家选择带来的影响。"
            )
            segment_payload["branches"] = []
            npc_dialogues = []

        memory.setdefault("history", []).append({"stage": stage, "content": content, "npc_dialogues": npc_dialogues})
        memory["last_branches"] = segment_payload.get("branches", [])
        memory["last_npc_dialogues"] = npc_dialogues
        memory["last_memory_stats"] = memory_stats
        self.memory_service.save(record, memory)

        record.latest_segment = content
        record.latest_branches = segment_payload.get("branches", [])
        db.add(record)

        return ScriptStateResponse(
            session_id=record.id,
            title=record.title,
            stage=stage,
            content=content,
            branches=record.latest_branches,
            npc_dialogues=npc_dialogues,
            memory_summary=self._build_memory_summary(memory),
            memory_stats=memory_stats,
            logic_checks=self.rule_engine.build_logic_checks(memory, content),
            is_finished=len(memory.get("history", [])) >= len(STAGES),
            provider=provider,
        )

    @staticmethod
    def _build_memory_summary(memory: dict) -> list[str]:
        summary: list[str] = []
        summary.extend(f"事实：{fact}" for fact in memory.get("facts", [])[-5:])
        summary.extend(f"选择：{choice}" for choice in memory.get("choices", [])[-3:])
        if memory.get("compressed_history"):
            summary.append(f"压缩摘要：{memory['compressed_history']}")
        if not summary:
            summary.append("当前尚无历史记忆。")
        return summary

    @staticmethod
    def _get_record(db: Session, owner_id: int, session_id: str) -> ScriptSession:
        record = db.get(ScriptSession, session_id)
        if not record or record.owner_id != owner_id:
            raise HTTPException(status_code=404, detail="未找到对应剧本会话。")
        return record
