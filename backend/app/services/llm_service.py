import copy
import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx

import json

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.services.dm_service import DMService
from app.services.longform_story_service import LongFormStoryService
from app.services.render_service import RenderService
from app.services.rule_engine import RuleEngine
from app.services.script_reference_service import ScriptReferenceService
from app.services.structure_engine import StructureEngine


class LLMService:
    ROLE_SECTION_TITLE = "【角色设定总览】"
    SUPPORTED_PROVIDERS = ("mock", "doubao", "openai", "dashscope", "siliconflow", "zhipu")
    PLAYER_HANDBOOK_TOKENS = ("玩家手册", "角色手册", "玩家本", "角色本")
    STORY_TOKENS = ("完整故事", "完整剧情", "故事剧情", "剧情故事", "叙事版", "写成故事", "写成剧情")
    CLUE_TOKENS = ("线索", "搜证", "证据", "物证", "线索卡")
    ENDING_TOKENS = ("结局", "反转", "真相", "收束", "回收")
    DM_TOKENS = ("dm", "主持", "控场", "主持人手册", "dm 手册")

    def __init__(self, rule_engine: RuleEngine) -> None:
        self.rule_engine = rule_engine
        self.structure_engine = StructureEngine()
        self.dm_service = DMService()
        self.render_service = RenderService()
        self.longform_story_service = LongFormStoryService()
        self.script_reference_service = ScriptReferenceService()
        self._provider_failure_until: dict[str, float] = {}

    def single_provider_mode_enabled(self) -> bool:
        return bool(settings.llm_single_provider_only)

    def mock_fallback_enabled(self) -> bool:
        return not settings.llm_disable_mock_fallback

    def locked_provider(self) -> str:
        if not self.single_provider_mode_enabled():
            return ""
        configured = (settings.llm_single_provider or settings.llm_provider or "mock").strip().lower()
        if configured in self.SUPPORTED_PROVIDERS:
            return configured
        return "mock"

    def locked_model(self) -> str:
        provider = self.locked_provider()
        if not provider:
            return ""
        return settings.llm_single_model or self.default_model_for_provider(provider)

    def default_model_for_provider(self, provider: str) -> str:
        normalized = self.normalize_provider(provider)
        if normalized == "doubao":
            return settings.doubao_model
        if normalized == "openai":
            return settings.openai_model
        if normalized == "dashscope":
            return settings.dashscope_model
        if normalized == "siliconflow":
            return settings.siliconflow_model
        if normalized == "zhipu":
            return settings.zhipu_model
        return "mock-story-model"

    def normalize_provider(self, provider: str | None) -> str:
        if self.single_provider_mode_enabled():
            return self.locked_provider()
        candidate = (provider or "").strip().lower()
        if candidate in self.SUPPORTED_PROVIDERS:
            return candidate
        fallback = (settings.llm_provider or "mock").strip().lower()
        if fallback in self.SUPPORTED_PROVIDERS:
            return fallback
        return "mock"

    def build_llm_profile(self, provider: str | None = None, model: str | None = None) -> dict[str, str]:
        requested_provider = (provider or "").strip().lower()
        normalized_provider = self.normalize_provider(provider)
        if self.single_provider_mode_enabled():
            return {"provider": normalized_provider, "model": self.locked_model()}
        if not requested_provider:
            first_enabled_real = next(
                (item for item in self._provider_items() if item["provider"] != "mock" and item["enabled"]),
                None,
            )
            if first_enabled_real:
                normalized_provider = str(first_enabled_real["provider"])
            else:
                configured_provider = self.normalize_provider(settings.llm_provider)
                configured_item = next(
                    (item for item in self._provider_items() if item["provider"] == configured_provider),
                    None,
                )
                if configured_item and (configured_item["provider"] == "mock" or configured_item["enabled"]):
                    normalized_provider = str(configured_item["provider"])
        normalized_model = (model or "").strip() or self.default_model_for_provider(normalized_provider)
        return {"provider": normalized_provider, "model": normalized_model}

    def _all_provider_items(self) -> list[dict[str, Any]]:
        return [
            {
                "provider": "doubao",
                "enabled": bool(settings.doubao_api_key),
                "model": settings.doubao_model,
                "reason": "已配置 Doubao / Ark API Key" if settings.doubao_api_key else "未配置 Doubao / Ark API Key",
            },
            {
                "provider": "openai",
                "enabled": bool(settings.openai_api_key),
                "model": settings.openai_model,
                "reason": "已配置 OpenAI API Key" if settings.openai_api_key else "未配置 OpenAI API Key",
            },
            {
                "provider": "dashscope",
                "enabled": bool(settings.dashscope_api_key),
                "model": settings.dashscope_model,
                "reason": "已配置 DashScope API Key" if settings.dashscope_api_key else "未配置 DashScope API Key",
            },
            {
                "provider": "siliconflow",
                "enabled": bool(settings.siliconflow_api_key),
                "model": settings.siliconflow_model,
                "reason": "已配置 SiliconFlow API Key" if settings.siliconflow_api_key else "未配置 SiliconFlow API Key",
            },
            {
                "provider": "zhipu",
                "enabled": bool(settings.zhipu_api_key),
                "model": settings.zhipu_model,
                "reason": "已配置智谱 API Key" if settings.zhipu_api_key else "未配置智谱 API Key",
            },
            {
                "provider": "mock",
                "enabled": True,
                "model": "mock-story-model",
                "reason": "本地兜底模式可用",
            },
        ]

    def _provider_items(self) -> list[dict[str, Any]]:
        items = self._all_provider_items()
        if not self.single_provider_mode_enabled():
            return items

        locked_provider = self.locked_provider()
        locked_model = self.locked_model()
        filtered: list[dict[str, Any]] = []
        for item in items:
            if item["provider"] == locked_provider:
                filtered.append({**item, "model": locked_model or item["model"]})
            elif item["provider"] == "mock" and self.mock_fallback_enabled():
                filtered.append(item)
        return filtered

    def get_runtime_profile(self) -> dict[str, Any]:
        providers = self._provider_items()
        if self.single_provider_mode_enabled():
            recommended = next((item for item in providers if item["provider"] == self.locked_provider()), providers[0])
        else:
            recommended = next((item for item in providers if item["provider"] != "mock" and item["enabled"]), providers[-1])
        configured = self.normalize_provider(settings.llm_provider)
        configured_item = next((item for item in providers if item["provider"] == configured), None)
        if not self.single_provider_mode_enabled() and configured_item and configured_item["provider"] != "mock" and configured_item["enabled"]:
            recommended = configured_item
        return {
            "recommended_provider": recommended["provider"],
            "recommended_model": recommended["model"],
            "providers": providers,
            "single_provider_mode": self.single_provider_mode_enabled(),
            "mock_fallback_enabled": self.mock_fallback_enabled(),
        }

    async def get_runtime_diagnostics(self) -> dict[str, Any]:
        providers = []
        for item in self._provider_items():
            if item["provider"] == "mock":
                providers.append({**item, "status": "warning", "detail": item["reason"], "latency_ms": None})
            elif item["enabled"]:
                started = time.monotonic()
                try:
                    await self._probe_provider_connectivity(str(item["provider"]), str(item["model"]))
                    providers.append(
                        {
                            **item,
                            "status": "ok",
                            "detail": "配置已就绪",
                            "latency_ms": max(1, round((time.monotonic() - started) * 1000)),
                        }
                    )
                except Exception as exc:
                    self._mark_provider_failure(str(item["provider"]))
                    providers.append(
                        {
                            **item,
                            "status": "error",
                            "detail": str(exc) or "连通性探测失败",
                            "latency_ms": max(1, round((time.monotonic() - started) * 1000)),
                        }
                    )
            else:
                providers.append({**item, "status": "skipped", "detail": item["reason"], "latency_ms": None})
        profile = self.get_runtime_profile()
        return {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "recommended_provider": profile["recommended_provider"],
            "recommended_model": profile["recommended_model"],
            "providers": providers,
            "single_provider_mode": self.single_provider_mode_enabled(),
            "mock_fallback_enabled": self.mock_fallback_enabled(),
        }

    def get_model_strategy(self) -> dict[str, Any]:
        profile = self.get_runtime_profile()
        return {
            "document_title": "剧本杀智能生成模型选型建议",
            "quickstart_summary": "优先跑通智能体创作、结构化输出和交付流程；未配置真实模型时可先用 Mock 演示完整链路。",
            "recommended_primary_provider": profile["recommended_provider"],
            "recommended_primary_model": profile["recommended_model"],
            "engineering_focus": [
                "优先保证角色设定、流程结构、DM 话术和导出结果的一致性。",
                "将大模型输出限制为结构化字段，避免整段自由发挥导致内容漂移。",
                "对长会话保留摘要和关键事实，避免上下文失控。",
            ],
            "cards": [
                {
                    "id": "mock",
                    "title": "本地演示",
                    "provider": "mock",
                    "model": "mock-story-model",
                    "recommendation": "适合功能演示、联调和论文截图，不依赖外部模型。",
                    "strengths": ["零成本", "稳定", "便于本地开发"],
                    "tradeoffs": ["创作质量有限"],
                    "use_cases": ["流程演示", "前后端联调"],
                },
                {
                    "id": "compatible",
                    "title": "兼容接口模型",
                    "provider": profile["recommended_provider"],
                    "model": profile["recommended_model"],
                    "recommendation": "适合接入真实大模型，提升生成质量和多轮改写表现。",
                    "strengths": ["接入成本低", "便于切换供应商"],
                    "tradeoffs": ["依赖 API Key 和网络"],
                    "use_cases": ["正式演示", "高质量生成"],
                },
            ],
        }

    def _resolve_provider_from_memory(self, memory: dict[str, Any]) -> str:
        if self.single_provider_mode_enabled():
            return self.locked_provider()
        profile = memory.get("llm_profile", {})
        requested = self.normalize_provider(profile.get("provider"))
        enabled_map = {item["provider"]: item["enabled"] for item in self._provider_items()}
        if requested != "mock" and enabled_map.get(requested):
            return requested
        recommended = self.get_runtime_profile()["recommended_provider"]
        return self.normalize_provider(recommended)

    def _mark_provider_failure(self, provider: str) -> None:
        self._provider_failure_until[self.normalize_provider(provider)] = time.monotonic() + settings.provider_failure_cooldown_seconds

    def _provider_is_temporarily_failed(self, provider: str) -> bool:
        normalized = self.normalize_provider(provider)
        expires_at = self._provider_failure_until.get(normalized, 0.0)
        if not expires_at:
            return False
        if expires_at <= time.monotonic():
            self._provider_failure_until.pop(normalized, None)
            return False
        return True

    def _build_real_provider_attempts(self, preferred_provider: str, preferred_model: str) -> list[tuple[str, str]]:
        attempts: list[tuple[str, str]] = []
        preferred_normalized = self.normalize_provider(preferred_provider)
        enabled_items = {str(item["provider"]): item for item in self._provider_items()}

        preferred_item = enabled_items.get(preferred_normalized)
        if (
            preferred_normalized != "mock"
            and preferred_item
            and preferred_item.get("enabled")
            and not self._provider_is_temporarily_failed(preferred_normalized)
        ):
            attempts.append((preferred_normalized, preferred_model or self.default_model_for_provider(preferred_normalized)))

        for item in self._provider_items():
            provider = str(item["provider"])
            if provider == "mock" or provider == preferred_normalized or not item["enabled"]:
                continue
            if self._provider_is_temporarily_failed(provider):
                continue
            attempts.append((provider, str(item["model"])))
        return attempts

    async def _probe_provider_connectivity(self, provider: str, model: str) -> None:
        api_key = self._provider_api_key(provider)
        base_url = self._provider_base_url(provider)
        if not api_key or not base_url:
            raise RuntimeError("缺少 API Key 或 Base URL")
        content = await self._request_compatible_chat_completion_text(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt="你是连通性探测器，只返回 ok。",
            user_prompt="请只输出 ok",
            timeout_seconds=settings.llm_diagnostic_timeout_seconds,
            temperature=0,
        )
        if not content or "ok" not in content.lower():
            raise RuntimeError("模型未返回预期探测结果")

    async def generate_segment(self, world_config: dict[str, Any], memory: dict[str, Any], stage: str) -> tuple[dict[str, Any], str]:
        provider = self._resolve_provider_from_memory(memory)
        if provider != "mock":
            profile = memory.get("llm_profile", {})
            model = self.locked_model() if self.single_provider_mode_enabled() else (profile.get("model") or "").strip() or self.default_model_for_provider(provider)
            for candidate_provider, candidate_model in self._build_real_provider_attempts(provider, model):
                generator = getattr(self, f"_generate_segment_with_{candidate_provider}", None)
                if not callable(generator):
                    continue
                try:
                    payload = await generator(world_config, memory, stage, candidate_model)
                except Exception:
                    self._mark_provider_failure(candidate_provider)
                    continue
                if payload and str(payload.get("content") or "").strip():
                    return payload, candidate_provider
                self._mark_provider_failure(candidate_provider)
            if not self.mock_fallback_enabled():
                raise HTTPException(status_code=503, detail="真实模型生成剧情片段失败，且系统已关闭 Mock 自动回退。")
        return self._build_mock_segment_payload(world_config, memory, stage), "mock"

    async def generate_review(self, world_config: dict[str, Any], memory: dict[str, Any]) -> tuple[str, str]:
        provider = self._resolve_provider_from_memory(memory)
        if provider != "mock":
            profile = memory.get("llm_profile", {})
            model = self.locked_model() if self.single_provider_mode_enabled() else (profile.get("model") or "").strip() or self.default_model_for_provider(provider)
            for candidate_provider, candidate_model in self._build_real_provider_attempts(provider, model):
                generator = getattr(self, f"_generate_review_with_{candidate_provider}", None)
                if not callable(generator):
                    continue
                try:
                    review = await generator(world_config, memory, candidate_model)
                except Exception:
                    self._mark_provider_failure(candidate_provider)
                    continue
                if review and str(review).strip():
                    return str(review).strip(), candidate_provider
                self._mark_provider_failure(candidate_provider)
            if not self.mock_fallback_enabled():
                raise HTTPException(status_code=503, detail="真实模型生成复盘失败，且系统已关闭 Mock 自动回退。")
        return self._build_mock_review(world_config, memory), "mock"

    def _build_mock_segment_payload(self, world_config: dict[str, Any], memory: dict[str, Any], stage: str) -> dict[str, Any]:
        conflict = world_config.get("core_conflict", "一场失控的局中局")
        role_name = world_config.get("role_name", "关键角色")
        content = self.longform_story_service.build_session_segment_content(world_config, memory, stage)
        branches = [
            {"id": f"{stage}-1", "title": "公开追问", "description": "把当前疑点摆到台面上，迫使关键人物回应。"},
            {"id": f"{stage}-2", "title": "私下试探", "description": "保留信息优势，先观察场上反应再行动。"},
            {"id": f"{stage}-3", "title": "转移焦点", "description": "暂时制造新的讨论中心，为后续布局争取空间。"},
        ]
        npc_dialogues = [
            {"speaker": role_name, "intent": "试探", "line": f"如果你们继续忽略‘{conflict}’，真正的答案只会被埋得更深。"},
            {"speaker": "主持人", "intent": "推进", "line": "现在请各位根据新线索表态，并准备进入下一轮互动。"},
        ]
        return {"content": content, "branches": branches, "npc_dialogues": npc_dialogues}

    def _build_mock_review(self, world_config: dict[str, Any], memory: dict[str, Any]) -> str:
        history = memory.get("history", [])
        choices = memory.get("choices", [])
        lines = [f"《{world_config.get('title', '未命名剧本')}》复盘摘要"]
        for index, item in enumerate(history, start=1):
            lines.append(f"{index}. {item.get('stage', '阶段')}：{item.get('content', '')}")
        if choices:
            lines.append("玩家关键选择：")
            for index, choice in enumerate(choices, start=1):
                lines.append(f"{index}. {choice}")
        if not history:
            lines.append("当前会话尚未形成可复盘的完整剧情推进。")
        return "\n".join(lines).strip()

    async def _generate_segment_with_openai(self, world_config: dict[str, Any], memory: dict[str, Any], stage: str, model: str) -> dict[str, Any]:
        return await self._generate_segment_with_compatible_api("openai", world_config, memory, stage, model)

    async def _generate_segment_with_doubao(self, world_config: dict[str, Any], memory: dict[str, Any], stage: str, model: str) -> dict[str, Any]:
        return await self._generate_segment_with_compatible_api("doubao", world_config, memory, stage, model)

    async def _generate_segment_with_dashscope(self, world_config: dict[str, Any], memory: dict[str, Any], stage: str, model: str) -> dict[str, Any]:
        return await self._generate_segment_with_compatible_api("dashscope", world_config, memory, stage, model)

    async def _generate_segment_with_siliconflow(self, world_config: dict[str, Any], memory: dict[str, Any], stage: str, model: str) -> dict[str, Any]:
        return await self._generate_segment_with_compatible_api("siliconflow", world_config, memory, stage, model)

    async def _generate_segment_with_zhipu(self, world_config: dict[str, Any], memory: dict[str, Any], stage: str, model: str) -> dict[str, Any]:
        return await self._generate_segment_with_compatible_api("zhipu", world_config, memory, stage, model)

    async def _generate_review_with_openai(self, world_config: dict[str, Any], memory: dict[str, Any], model: str) -> str:
        return await self._generate_review_with_compatible_api("openai", world_config, memory, model)

    async def _generate_review_with_doubao(self, world_config: dict[str, Any], memory: dict[str, Any], model: str) -> str:
        return await self._generate_review_with_compatible_api("doubao", world_config, memory, model)

    async def _generate_review_with_dashscope(self, world_config: dict[str, Any], memory: dict[str, Any], model: str) -> str:
        return await self._generate_review_with_compatible_api("dashscope", world_config, memory, model)

    async def _generate_review_with_siliconflow(self, world_config: dict[str, Any], memory: dict[str, Any], model: str) -> str:
        return await self._generate_review_with_compatible_api("siliconflow", world_config, memory, model)

    async def _generate_review_with_zhipu(self, world_config: dict[str, Any], memory: dict[str, Any], model: str) -> str:
        return await self._generate_review_with_compatible_api("zhipu", world_config, memory, model)

    async def generate_agent_turn(self, memory: dict[str, Any]) -> tuple[dict[str, Any], str]:
        provider = self._resolve_provider_from_memory(memory)
        if provider != "mock":
            profile = memory.get("llm_profile", {})
            model = self.locked_model() if self.single_provider_mode_enabled() else (profile.get("model") or "").strip() or self.default_model_for_provider(provider)
            attempts = self._build_real_provider_attempts(provider, model) or [(provider, model)]
            for candidate_provider, candidate_model in attempts:
                payload = await self._generate_agent_turn_with_provider(candidate_provider, memory, candidate_model)
                if payload:
                    return payload, candidate_provider
                self._mark_provider_failure(candidate_provider)
            if not self.mock_fallback_enabled():
                raise HTTPException(
                    status_code=503,
                    detail=f"当前已锁定仅使用 {provider} / {model}，但该模型请求失败，且系统已关闭 Mock 自动回退。",
                )
        payload = self._build_mock_agent_turn_payload(memory)
        return payload, "mock"

    def _build_mock_agent_turn_payload(self, memory: dict[str, Any]) -> dict[str, Any]:
        onboarding = memory.get("onboarding", {})
        answers = onboarding.get("answers", {})
        summary = memory.get("summary", {})
        title = summary.get("title") or self._build_title_from_answers(answers)
        era = answers.get("era") or summary.get("era", "现代都市")
        genre = answers.get("genre") or summary.get("genre", "现代悬疑")
        players = answers.get("players", "6 人")
        conflict = answers.get("conflict", "隐藏在关系中的真相")
        protagonist = answers.get("protagonist", "关键角色")
        goal = answers.get("goal", "揭开真相")
        role_cards = copy.deepcopy(memory.get("role_cards") or self._build_role_cards(players, genre, protagonist, conflict))
        draft_content = self.sync_draft_role_card_section(
            self._build_agent_draft(title, era, genre, conflict, protagonist, goal, role_cards, memory),
            role_cards,
        )
        structure_plan = self.structure_engine.build_flow(
            genre=genre,
            players=players,
            duration="120 分钟",
            scene=answers.get("scene", "封闭主场景 + 多点互动空间"),
            conflict=conflict,
            goal=goal,
            template_key=str(memory.get("template_key") or ""),
        )
        dm_script = self.dm_service.build_cues(
            stages=structure_plan,
            genre=genre,
            conflict=conflict,
            goal=goal,
            role_cards=role_cards,
            template_key=str(memory.get("template_key") or ""),
        )
        output_outline = self.render_service.build_output_outline(len(role_cards) if role_cards else 6)
        latest_request = self._extract_latest_generation_request(memory)
        generated_summary = {
            "title": title,
            "era": era,
            "genre": genre,
            "focus": self._build_focus_label(latest_request),
        }
        logic_checks = self.rule_engine.build_agent_logic_checks(
            memory=memory,
            draft_content=draft_content,
            role_cards=role_cards,
            structure_plan=structure_plan,
            summary=generated_summary,
            template_key=str(memory.get("template_key") or ""),
        )
        payload = {
            "assistant_reply": self._build_agent_reply(title, conflict, goal, latest_request),
            "draft_content": draft_content,
            "suggestions": self._build_suggestions(conflict, role_cards),
            "facts": self._merge_facts(memory.get("facts", []), answers, conflict, goal),
            "role_cards": role_cards,
            "summary": generated_summary,
            "logic_checks": logic_checks,
            "task_queue": [
                {"id": "plan-1", "module": "structure", "title": "完善流程分幕", "detail": "根据当前设定继续细化每一幕目标", "status": "active"},
                {"id": "plan-2", "module": "roles", "title": "补充角色秘密", "detail": "让角色关系链更清晰", "status": "pending"},
                {"id": "plan-3", "module": "export", "title": "准备交付内容", "detail": "同步 DM 手册和复盘说明", "status": "pending"},
            ],
            "structure_plan": structure_plan,
            "dm_script": dm_script,
            "output_outline": output_outline,
        }
        return payload

    async def _generate_agent_turn_with_provider(
        self,
        provider: str,
        memory: dict[str, Any],
        model: str,
        *,
        compact: bool = False,
        timeout_seconds: int = 0,
    ) -> dict[str, Any] | None:
        normalized = self.normalize_provider(provider)
        if normalized == "openai":
            return await self._generate_agent_turn_with_openai(memory, model, compact=compact, timeout_seconds=timeout_seconds)
        if normalized == "doubao":
            return await self._generate_agent_turn_with_doubao(memory, model, compact=compact, timeout_seconds=timeout_seconds)
        if normalized == "dashscope":
            return await self._generate_agent_turn_with_dashscope(memory, model, compact=compact, timeout_seconds=timeout_seconds)
        if normalized == "siliconflow":
            return await self._generate_agent_turn_with_siliconflow(memory, model, compact=compact, timeout_seconds=timeout_seconds)
        if normalized == "zhipu":
            return await self._generate_agent_turn_with_zhipu(memory, model, compact=compact, timeout_seconds=timeout_seconds)
        return None

    async def _generate_agent_turn_with_openai(self, memory: dict[str, Any], model: str, *, compact: bool = False, timeout_seconds: int = 0) -> dict[str, Any] | None:
        return await self._generate_agent_turn_with_compatible_api("openai", memory, model, compact=compact, timeout_seconds=timeout_seconds)

    async def _generate_agent_turn_with_doubao(self, memory: dict[str, Any], model: str, *, compact: bool = False, timeout_seconds: int = 0) -> dict[str, Any] | None:
        return await self._generate_agent_turn_with_compatible_api("doubao", memory, model, compact=compact, timeout_seconds=timeout_seconds)

    async def _generate_agent_turn_with_dashscope(self, memory: dict[str, Any], model: str, *, compact: bool = False, timeout_seconds: int = 0) -> dict[str, Any] | None:
        return await self._generate_agent_turn_with_compatible_api("dashscope", memory, model, compact=compact, timeout_seconds=timeout_seconds)

    async def _generate_agent_turn_with_siliconflow(self, memory: dict[str, Any], model: str, *, compact: bool = False, timeout_seconds: int = 0) -> dict[str, Any] | None:
        return await self._generate_agent_turn_with_compatible_api("siliconflow", memory, model, compact=compact, timeout_seconds=timeout_seconds)

    async def _generate_agent_turn_with_zhipu(self, memory: dict[str, Any], model: str, *, compact: bool = False, timeout_seconds: int = 0) -> dict[str, Any] | None:
        return await self._generate_agent_turn_with_compatible_api("zhipu", memory, model, compact=compact, timeout_seconds=timeout_seconds)

    async def _generate_agent_turn_with_compatible_api(
        self,
        provider: str,
        memory: dict[str, Any],
        model: str,
        *,
        compact: bool = False,
        timeout_seconds: int = 0,
    ) -> dict[str, Any] | None:
        api_key = self._provider_api_key(provider)
        base_url = self._provider_base_url(provider)
        if not api_key or not base_url:
            return None
        prompt = self._build_agent_turn_prompt(memory, compact=compact)
        response_json = await self._post_compatible_chat_completion(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=(
                "你是剧本杀创作智能体。你当前通过 OpenAI 兼容接口被调用。"
                " 你必须只返回一个合法 JSON 对象，不能输出 markdown、代码块、项目符号、解释或任何前后缀文本。"
                " 顶层字段必须包含 assistant_reply, draft_content, suggestions, facts, role_cards, summary。"
                " summary 必须包含 title, era, genre, focus。"
                " role_cards 必须是数组；每个角色至少包含 id, name, archetype, public_identity, hidden_secret, motivation, key_prop, tags, behavior_rules, relationships。"
                " 如果信息不足，也必须补空字符串或空数组，不能省略这些字段。"
            ),
            user_prompt=prompt,
            timeout_seconds=timeout_seconds or settings.agent_primary_timeout_seconds,
        )
        if not response_json:
            return None
        return self._normalize_real_provider_agent_payload(response_json, memory)

    async def _generate_segment_with_compatible_api(
        self,
        provider: str,
        world_config: dict[str, Any],
        memory: dict[str, Any],
        stage: str,
        model: str,
    ) -> dict[str, Any]:
        api_key = self._provider_api_key(provider)
        base_url = self._provider_base_url(provider)
        if not api_key or not base_url:
            raise RuntimeError("缺少 API Key 或 Base URL")
        payload = await self._request_json_payload(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=(
                "你是线下剧本杀编剧与 DM 联合创作助手。"
                " 你必须只返回一个合法 JSON 对象，不能输出 markdown、解释或代码块。"
                " 顶层字段必须包含 content, branches, npc_dialogues。"
                " content 必须是连续剧情正文，不是提纲。"
            ),
            user_prompt=self._build_segment_prompt(world_config, memory, stage),
            required_fields=("content", "branches", "npc_dialogues"),
        )
        if not payload:
            raise RuntimeError("模型未返回可用剧情片段")
        content = str(payload.get("content") or "").strip()
        if not content:
            raise RuntimeError("剧情片段为空")
        return {
            "content": content,
            "branches": self._normalize_branch_options(payload.get("branches"), stage),
            "npc_dialogues": self._normalize_npc_dialogues(payload.get("npc_dialogues"), world_config),
        }

    async def _generate_review_with_compatible_api(
        self,
        provider: str,
        world_config: dict[str, Any],
        memory: dict[str, Any],
        model: str,
    ) -> str:
        api_key = self._provider_api_key(provider)
        base_url = self._provider_base_url(provider)
        if not api_key or not base_url:
            raise RuntimeError("缺少 API Key 或 Base URL")
        content = await self._request_compatible_chat_completion_text(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt="你是剧本杀复盘作者，请直接输出复盘正文，不要 JSON，不要 markdown。",
            user_prompt=self._build_review_prompt(world_config, memory),
            timeout_seconds=settings.agent_primary_timeout_seconds,
            temperature=0.5,
        )
        cleaned = str(content or "").strip()
        if not cleaned:
            raise RuntimeError("复盘内容为空")
        return cleaned

    async def _post_compatible_chat_completion(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: int,
    ) -> dict[str, Any] | None:
        strict_user_prompt = (
            f"{user_prompt}\n\n"
            "输出模板示例（字段名必须保持一致，可以自行补全内容）：\n"
            "{\n"
            '  "assistant_reply": "本轮响应摘要",\n'
            '  "draft_content": "完整草案正文",\n'
            '  "suggestions": [{"label": "建议标题", "prompt": "建议指令", "category": "create"}],\n'
            '  "facts": ["已确认事实"],\n'
            '  "role_cards": [\n'
            "    {\n"
            '      "id": "role-1",\n'
            '      "name": "角色名",\n'
            '      "archetype": "人物类型",\n'
            '      "public_identity": "公开身份",\n'
            '      "hidden_secret": "隐藏秘密",\n'
            '      "motivation": "核心动机",\n'
            '      "key_prop": "关键道具",\n'
            '      "tags": ["标签"],\n'
            '      "behavior_rules": ["行为规则"],\n'
            '      "relationships": ["人物关系"]\n'
            "    }\n"
            "  ],\n"
            '  "summary": {"title": "标题", "era": "时代", "genre": "类型", "focus": "本轮重点"}\n'
            "}\n"
            "再次强调：你的回复必须从 { 开始并以 } 结束，且只能包含 JSON。"
        )

        first_content = await self._request_compatible_chat_completion_text(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            user_prompt=strict_user_prompt,
            timeout_seconds=timeout_seconds,
            temperature=0.4,
        )
        parsed = self._parse_json_text(first_content or "")
        if parsed:
            return parsed

        retry_content = await self._request_compatible_chat_completion_text(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=(
                f"{system_prompt} 你上一次没有按要求输出。"
                " 这一次必须只输出一个合法 JSON 对象，禁止任何解释、标题、markdown、代码块或额外文本。"
            ),
            user_prompt=strict_user_prompt,
            timeout_seconds=timeout_seconds,
            temperature=0,
        )
        parsed = self._parse_json_text(retry_content or "")
        if parsed:
            return parsed

        repaired_content = await self._request_compatible_chat_completion_text(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=(
                "你是 JSON 修复器。"
                " 你只能输出一个合法 JSON 对象，不能输出解释、markdown、代码块或额外文本。"
            ),
            user_prompt=(
                "请把下面这段模型回复整理成一个合法 JSON 对象。"
                " 保留 assistant_reply, draft_content, suggestions, facts, role_cards, summary 这些顶层字段；"
                " 如果原文缺失某个字段，请补空字符串、空数组或空对象。"
                f"\n\n原始回复：\n{retry_content or first_content or ''}"
            ),
            timeout_seconds=timeout_seconds,
            temperature=0,
        )
        return self._parse_json_text(repaired_content or "")

    async def _request_json_payload(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        system_prompt: str,
        user_prompt: str,
        required_fields: tuple[str, ...],
    ) -> dict[str, Any] | None:
        strict_user_prompt = (
            f"{user_prompt}\n\n"
            "输出必须是一个合法 JSON 对象，且只能包含 JSON。"
            f" 顶层字段至少包含：{', '.join(required_fields)}。"
        )
        first_content = await self._request_compatible_chat_completion_text(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            user_prompt=strict_user_prompt,
            timeout_seconds=settings.agent_primary_timeout_seconds,
            temperature=0.4,
        )
        parsed = self._parse_json_text(first_content or "")
        if parsed and all(field in parsed for field in required_fields):
            return parsed
        retry_content = await self._request_compatible_chat_completion_text(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=f"{system_prompt} 你上一次没有按要求输出，这一次必须只输出合法 JSON。",
            user_prompt=strict_user_prompt,
            timeout_seconds=settings.agent_primary_timeout_seconds,
            temperature=0,
        )
        parsed = self._parse_json_text(retry_content or "")
        if parsed and all(field in parsed for field in required_fields):
            return parsed
        return None

    async def _request_compatible_chat_completion_text(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        system_prompt: str,
        user_prompt: str,
        timeout_seconds: int,
        temperature: float,
        max_tokens: int = 0,
    ) -> str | None:
        try:
            request_body = {
                "model": model,
                "temperature": temperature,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if max_tokens > 0:
                request_body["max_tokens"] = max_tokens
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_body,
                )
                response.raise_for_status()
                data = response.json()
        except Exception:
            return None
        try:
            return str(data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return None

    @staticmethod
    def _parse_json_text(content: str) -> dict[str, Any] | None:
        text = (content or "").strip()
        if not text:
            return None
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        for block in fenced_blocks:
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                parsed, _ = decoder.raw_decode(text[match.start():])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    def _normalize_real_provider_agent_payload(self, payload: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
        onboarding = memory.get("onboarding", {})
        answers = onboarding.get("answers", {})
        current_summary = memory.get("summary", {})
        latest_request = self._extract_latest_generation_request(memory)
        raw_summary = payload.get("summary")
        raw_draft = payload.get("draft_content")
        summary = raw_summary if isinstance(raw_summary, dict) else {}
        draft_mapping = raw_draft if isinstance(raw_draft, dict) else {}
        title = self._pick_meaningful_text(
            summary.get("title"),
            draft_mapping.get("title"),
            current_summary.get("title"),
            self._build_title_from_answers(answers),
        )
        era = self._pick_meaningful_text(
            summary.get("era"),
            draft_mapping.get("era"),
            answers.get("era"),
            current_summary.get("era"),
            "现代都市",
        )
        genre = self._pick_meaningful_text(
            summary.get("genre"),
            draft_mapping.get("genre"),
            draft_mapping.get("type"),
            answers.get("genre"),
            current_summary.get("genre"),
            "现代悬疑",
        )
        conflict = answers.get("conflict", "隐藏在关系中的真相")
        protagonist = answers.get("protagonist", "关键角色")
        goal = answers.get("goal", "揭开真相")
        players = answers.get("players", "6 人")
        role_cards = self._normalize_role_cards(payload.get("role_cards"), memory, players, genre, protagonist, conflict)
        draft_content = self._normalize_provider_draft_content(
            raw_draft,
            memory=memory,
            title=title,
            era=era,
            genre=genre,
            conflict=conflict,
            protagonist=protagonist,
            goal=goal,
            role_cards=role_cards,
            latest_request=latest_request,
        )
        if not draft_content:
            draft_content = self.sync_draft_role_card_section(
                self._build_agent_draft(title, era, genre, conflict, protagonist, goal, role_cards, memory),
                role_cards,
            )
        else:
            draft_content = self._ensure_request_section(
                draft_content,
                latest_request=latest_request,
                title=title,
                conflict=conflict,
                protagonist=protagonist,
                goal=goal,
                role_cards=role_cards,
                memory=memory,
            )
            draft_content = self.sync_draft_role_card_section(draft_content, role_cards)
        structure_plan = self.structure_engine.build_flow(
            genre=genre,
            players=players,
            duration="120 分钟",
            scene=answers.get("scene", "封闭主场景 + 多点互动空间"),
            conflict=conflict,
            goal=goal,
            template_key=str(memory.get("template_key") or ""),
        )
        dm_script = self.dm_service.build_cues(
            stages=structure_plan,
            genre=genre,
            conflict=conflict,
            goal=goal,
            role_cards=role_cards,
            template_key=str(memory.get("template_key") or ""),
        )
        output_outline = self.render_service.build_output_outline(len(role_cards) if role_cards else 6)
        normalized_summary = {
            "title": title,
            "era": era,
            "genre": genre,
            "focus": self._pick_meaningful_text(
                summary.get("focus"),
                raw_summary if isinstance(raw_summary, str) else "",
                self._build_focus_label(latest_request),
            ),
        }
        facts = self._normalize_facts(payload.get("facts"), memory, answers, conflict, goal)
        suggestions = self._normalize_provider_suggestions(payload.get("suggestions"), conflict, role_cards)
        logic_checks = self.rule_engine.build_agent_logic_checks(
            memory=memory,
            draft_content=draft_content,
            role_cards=role_cards,
            structure_plan=structure_plan,
            summary=normalized_summary,
            template_key=str(memory.get("template_key") or ""),
        )
        return {
            "assistant_reply": str(payload.get("assistant_reply") or self._build_agent_reply(title, conflict, goal, latest_request)).strip(),
            "draft_content": draft_content,
            "suggestions": suggestions,
            "facts": facts,
            "role_cards": role_cards,
            "summary": normalized_summary,
            "logic_checks": logic_checks,
            "task_queue": [
                {"id": "plan-1", "module": "structure", "title": "完善流程分幕", "detail": "根据当前设定继续细化每一幕目标", "status": "active"},
                {"id": "plan-2", "module": "roles", "title": "补充角色秘密", "detail": "让角色关系链更清晰", "status": "pending"},
                {"id": "plan-3", "module": "export", "title": "准备交付内容", "detail": "同步 DM 手册和复盘说明", "status": "pending"},
            ],
            "structure_plan": structure_plan,
            "dm_script": dm_script,
            "output_outline": output_outline,
        }

    def _normalize_provider_draft_content(
        self,
        raw: Any,
        *,
        memory: dict[str, Any],
        title: str,
        era: str,
        genre: str,
        conflict: str,
        protagonist: str,
        goal: str,
        role_cards: list[dict[str, Any]],
        latest_request: str,
    ) -> str:
        if isinstance(raw, str):
            return raw.strip()

        if isinstance(raw, dict):
            structured_conflict = self._pick_meaningful_text(raw.get("core_conflict"), conflict)
            structured_protagonist = self._pick_meaningful_text(raw.get("main_character_viewpoint"), protagonist)
            structured_goal = self._pick_meaningful_text(raw.get("player_objective"), goal)
            structured_players = self._pick_meaningful_text(
                raw.get("number_of_characters"),
                raw.get("players"),
                memory.get("onboarding", {}).get("answers", {}).get("players"),
                "6 人",
            )

            synthetic_memory = copy.deepcopy(memory)
            synthetic_memory.setdefault("facts", [])
            for item in (
                f"剧本类型：{genre}",
                f"时代背景：{era}",
                f"参与人数：{structured_players}",
                f"核心冲突：{structured_conflict}",
                f"玩家目标：{structured_goal}",
            ):
                if item not in synthetic_memory["facts"]:
                    synthetic_memory["facts"].append(item)

            structured_draft = self._build_agent_draft(
                title=title,
                era=era,
                genre=genre,
                conflict=structured_conflict,
                protagonist=structured_protagonist,
                goal=structured_goal,
                role_cards=role_cards,
                memory=synthetic_memory,
            )
            return self._ensure_request_section(
                structured_draft,
                latest_request=latest_request,
                title=title,
                conflict=structured_conflict,
                protagonist=structured_protagonist,
                goal=structured_goal,
                role_cards=role_cards,
                memory=synthetic_memory,
            ).strip()

        return ""

    def _normalize_provider_suggestions(self, raw: Any, conflict: str, role_cards: list[dict[str, Any]]) -> list[dict[str, str]]:
        if isinstance(raw, list):
            suggestions: list[dict[str, str]] = []
            for index, item in enumerate(raw[:3], start=1):
                if isinstance(item, str):
                    prompt = item.strip()
                    if prompt:
                        suggestions.append(
                            {
                                "id": f"suggestion-{index}",
                                "label": f"建议 {index}",
                                "prompt": prompt,
                                "category": "create",
                            }
                        )
                    continue
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "").strip()
                prompt = str(item.get("prompt") or "").strip()
                category = str(item.get("category") or "create").strip() or "create"
                if label and prompt:
                    suggestions.append({"id": f"suggestion-{index}", "label": label, "prompt": prompt, "category": category})
            if suggestions:
                return suggestions
        return self._build_suggestions(conflict, role_cards)

    def _build_segment_prompt(self, world_config: dict[str, Any], memory: dict[str, Any], stage: str) -> str:
        base_content = self.longform_story_service.build_session_segment_content(world_config, memory, stage)
        facts = "\n".join(f"- {str(item).strip()}" for item in memory.get("facts", [])[-6:] if str(item).strip()) or "- 暂无额外事实"
        choices = "\n".join(f"- {str(item).strip()}" for item in memory.get("choices", [])[-3:] if str(item).strip()) or "- 暂无关键选择"
        history = "\n".join(
            f"- {item.get('stage', '阶段')}：{self._short_text(str(item.get('content', '')), 120)}"
            for item in memory.get("history", [])[-3:]
        ) or "- 暂无前序阶段"
        return (
            f"当前阶段：{stage}\n"
            f"剧本标题：{world_config.get('title', '未命名剧本')}\n"
            f"核心冲突：{world_config.get('core_conflict', '一场失控的局中局')}\n"
            f"主控角色：{world_config.get('role_name', '关键角色')}\n"
            f"阶段长文本基底：\n{base_content}\n\n"
            f"已确认事实：\n{facts}\n\n"
            f"玩家最近选择：\n{choices}\n\n"
            f"最近阶段历史：\n{history}\n\n"
            "请生成：\n"
            '1. "content"：800-1500 字左右的连续剧情阶段正文，要有场景、动作、对白试探、信息差和阶段收束。\n'
            '2. "branches"：3 个分支选项，每个选项包含 id, title, description。\n'
            '3. "npc_dialogues"：2-4 条 NPC/DM 可直接使用的话术，每条包含 speaker, intent, line。\n'
            "要求更贴近当前线下剧本杀：有搜证误导、关系撕扯、公聊私聊转换和终局回收意识。"
        )

    def _build_review_prompt(self, world_config: dict[str, Any], memory: dict[str, Any]) -> str:
        history = "\n".join(
            f"{index}. {item.get('stage', '阶段')}：{item.get('content', '')}"
            for index, item in enumerate(memory.get("history", []), start=1)
        ) or "暂无完整历史"
        choices = "\n".join(
            f"{index}. {choice}"
            for index, choice in enumerate(memory.get("choices", []), start=1)
        ) or "暂无关键选择"
        return (
            f"请为《{world_config.get('title', '未命名剧本')}》生成一份可宣读的复盘。\n\n"
            f"核心冲突：{world_config.get('core_conflict', '一场失控的局中局')}\n"
            f"剧情历史：\n{history}\n\n"
            f"玩家选择：\n{choices}\n\n"
            "复盘要求：先还原真相，再回收误导线索与角色动机，最后总结玩家抉择如何改变了局势。"
        )

    def _normalize_branch_options(self, raw: Any, stage: str) -> list[dict[str, str]]:
        if isinstance(raw, list):
            normalized: list[dict[str, str]] = []
            for index, item in enumerate(raw[:3], start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                description = str(item.get("description") or "").strip()
                if title and description:
                    normalized.append(
                        {
                            "id": str(item.get("id") or f"{stage}-{index}").strip(),
                            "title": title,
                            "description": description,
                        }
                    )
            if normalized:
                return normalized
        return self._build_mock_segment_payload({}, {}, stage)["branches"]

    def _normalize_npc_dialogues(self, raw: Any, world_config: dict[str, Any]) -> list[dict[str, str]]:
        if isinstance(raw, list):
            normalized: list[dict[str, str]] = []
            for item in raw[:4]:
                if not isinstance(item, dict):
                    continue
                speaker = str(item.get("speaker") or "").strip()
                intent = str(item.get("intent") or "").strip()
                line = str(item.get("line") or "").strip()
                if speaker and intent and line:
                    normalized.append({"speaker": speaker, "intent": intent, "line": line})
            if normalized:
                return normalized
        return self._build_mock_segment_payload(world_config, {}, "当前阶段")["npc_dialogues"]

    def _normalize_facts(self, raw: Any, memory: dict[str, Any], answers: dict[str, str], conflict: str, goal: str) -> list[str]:
        facts: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                text = str(item or "").strip()
                if text:
                    facts.append(text)
        return self._merge_facts(facts or memory.get("facts", []), answers, conflict, goal)

    def _normalize_role_cards(
        self,
        raw: Any,
        memory: dict[str, Any],
        players: str,
        genre: str,
        protagonist: str,
        conflict: str,
    ) -> list[dict[str, Any]]:
        defaults = self.sanitize_role_cards(
            copy.deepcopy(memory.get("role_cards") or self._build_role_cards(players, genre, protagonist, conflict)),
            genre=genre,
            conflict=conflict,
        )
        if not isinstance(raw, list) or not raw:
            return defaults

        raw_items = [item for item in raw if isinstance(item, dict)]
        if not raw_items:
            return defaults

        cards: list[dict[str, Any]] = []
        matched_raw_indexes: set[int] = set()
        fallback_cards = defaults or self._build_role_cards(players, genre, protagonist, conflict)
        fallback_base = copy.deepcopy(fallback_cards[0]) if fallback_cards else {}

        for index, base in enumerate(fallback_cards):
            raw_index = self._find_matching_role_card_index(base, raw_items, matched_raw_indexes)
            if raw_index is not None:
                matched_raw_indexes.add(raw_index)
                item = raw_items[raw_index]
            else:
                item = {}
            cards.append(self._build_role_card(item, base, index, genre, conflict))

        for raw_index, item in enumerate(raw_items):
            if raw_index in matched_raw_indexes:
                continue
            cards.append(self._build_role_card(item, fallback_base, len(cards), genre, conflict))

        return self.sanitize_role_cards(cards, genre=genre, conflict=conflict) or defaults

    def sanitize_role_cards(
        self,
        raw: Any,
        *,
        genre: str = "现代悬疑",
        conflict: str = "隐藏在关系中的真相",
    ) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []

        cards: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            card = self._build_role_card(item, item, index, genre, conflict)
            card_id = self._role_card_lookup_key(card.get("id"))
            card_name = self._role_card_lookup_key(card.get("name"))
            if card_id and card_id in seen_ids:
                continue
            if card_name and card_name in seen_names:
                continue
            if card_id:
                seen_ids.add(card_id)
            if card_name:
                seen_names.add(card_name)
            cards.append(card)
        return cards

    @staticmethod
    def _role_card_lookup_key(value: Any) -> str:
        return str(value or "").strip().casefold()

    def _find_matching_role_card_index(
        self,
        base: dict[str, Any],
        raw_items: list[dict[str, Any]],
        used_indexes: set[int],
    ) -> int | None:
        base_id = self._role_card_lookup_key(base.get("id"))
        base_name = self._role_card_lookup_key(base.get("name"))
        for index, item in enumerate(raw_items):
            if index in used_indexes:
                continue
            if base_id and self._role_card_lookup_key(item.get("id")) == base_id:
                return index
        for index, item in enumerate(raw_items):
            if index in used_indexes:
                continue
            if base_name and self._role_card_lookup_key(item.get("name")) == base_name:
                return index
        return None

    @staticmethod
    def _normalize_string_list(values: Any, fallback: list[str]) -> list[str]:
        if isinstance(values, list):
            normalized = [str(item).strip() for item in values if str(item).strip()]
            if normalized:
                return normalized
        return [str(item).strip() for item in fallback if str(item).strip()]

    def _build_role_card(
        self,
        item: dict[str, Any],
        base: dict[str, Any],
        index: int,
        genre: str,
        conflict: str,
    ) -> dict[str, Any]:
        return {
            "id": str(item.get("id") or base.get("id") or f"role-{index + 1}").strip(),
            "name": str(item.get("name") or base.get("name") or f"角色{index + 1}").strip(),
            "archetype": str(item.get("archetype") or base.get("archetype") or "关键关系人").strip(),
            "public_identity": str(item.get("public_identity") or base.get("public_identity") or "公开身份待补充").strip(),
            "hidden_secret": str(item.get("hidden_secret") or base.get("hidden_secret") or f"与“{conflict}”相关的隐藏秘密待逐步揭示").strip(),
            "motivation": str(item.get("motivation") or base.get("motivation") or "在局中保护自身立场并争取主动权").strip(),
            "key_prop": str(item.get("key_prop") or base.get("key_prop") or f"关键道具{index + 1}").strip(),
            "tags": self._normalize_string_list(item.get("tags"), base.get("tags") or [genre, "信息拼图"]),
            "behavior_rules": self._normalize_string_list(item.get("behavior_rules"), base.get("behavior_rules") or ["优先维护人设"]),
            "relationships": self._normalize_string_list(item.get("relationships"), base.get("relationships") or ["与其他角色存在待揭示的关系链"]),
        }

    @staticmethod
    def _is_placeholder_like(value: Any) -> bool:
        text = str(value or "").strip()
        if not text:
            return True
        normalized = text.replace("？", "?").replace(" ", "")
        if normalized and set(normalized) == {"?"}:
            return True
        return normalized in {"待补充", "待确认", "未知", "n/a", "none", "null"}

    def _pick_meaningful_text(self, *values: Any) -> str:
        for value in values:
            if self._is_placeholder_like(value):
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _provider_api_key(self, provider: str) -> str:
        normalized = self.normalize_provider(provider)
        if normalized == "doubao":
            return settings.doubao_api_key or ""
        if normalized == "openai":
            return settings.openai_api_key or ""
        if normalized == "dashscope":
            return settings.dashscope_api_key or ""
        if normalized == "siliconflow":
            return settings.siliconflow_api_key or ""
        if normalized == "zhipu":
            return settings.zhipu_api_key or ""
        return ""

    def _provider_base_url(self, provider: str) -> str:
        normalized = self.normalize_provider(provider)
        if normalized == "doubao":
            return settings.doubao_base_url
        if normalized == "openai":
            return settings.openai_base_url
        if normalized == "dashscope":
            return settings.dashscope_base_url
        if normalized == "siliconflow":
            return settings.siliconflow_base_url
        if normalized == "zhipu":
            return settings.zhipu_base_url
        return ""

    def _build_agent_turn_prompt(self, memory: dict[str, Any], *, compact: bool = False) -> str:
        latest_request = self._extract_latest_generation_request(memory) or "继续完善当前剧本"
        summary = memory.get("summary", {})
        onboarding = memory.get("onboarding", {})
        answers = onboarding.get("answers", {})
        facts = "\n".join(f"- {item}" for item in memory.get("facts", [])[:12]) or "- 暂无已确认事实"
        role_lines = "\n".join(
            f"- {item.get('name', '角色')} / 公开身份：{item.get('public_identity', '待补充')} / 隐藏秘密：{item.get('hidden_secret', '待补充')}"
            for item in memory.get("role_cards", [])[:8]
        ) or "- 暂无角色卡"
        recent_messages = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}"
            for item in memory.get("messages", [])[-6:]
        ) or "- 暂无历史消息"
        task_queue = "\n".join(
            f"- {item.get('title', '任务')} / {item.get('status', 'pending')} / {item.get('detail', '')}"
            for item in memory.get("task_queue", [])[:6]
        ) or "- 暂无任务"
        structure_plan = "\n".join(
            f"- {item.get('title', '阶段')}：{item.get('objective', '')}"
            for item in memory.get("structure_plan", [])[:6]
        ) or "- 暂无结构分幕"
        memory_nodes = "\n".join(
            f"- {item.get('label', '记忆节点')}：{self._short_text(str(item.get('content', '')), 70)}"
            for item in memory.get("memory_nodes", [])[-5:]
        ) or "- 暂无记忆节点"
        output_outline = "\n".join(
            f"- {item.get('title', '交付项')}：{item.get('description', '')}"
            for item in memory.get("output_outline", [])[:6]
        ) or "- 暂无交付结构"
        compressed_context = str(memory.get("compressed_context") or "").strip() or "暂无压缩上下文"
        draft = str(memory.get("draft_content") or "").strip() or "暂无草案，请基于上下文生成。"
        mode_hint = "请尽量压缩但保留结构完整性。" if compact else "请优先生成可继续迭代的完整结构化草案。"
        execution_plan = self._build_agent_execution_plan(memory)
        longform_context = self.longform_story_service.build_agent_longform_prompt_context(memory, compact=compact)
        genre_constraints = self._build_genre_prompt_directives(answers, summary, latest_request)
        reference_context = self._build_agent_reference_context(memory, latest_request)
        reference_section = f"{reference_context}\n\n" if reference_context else ""
        co_creation = memory.get("co_creation", {})
        locked_fields = "、".join(co_creation.get("locked_fields", []) or []) or "无"
        locked_values = "\n".join(
            f"- {key}: {str(value)[:160]}"
            for key, value in (co_creation.get("locked_values", {}) or {}).items()
            if str(value).strip()
        ) or "- 无"
        co_creation_section = ""
        if co_creation.get("enabled", False):
            co_creation_section = (
                "共创模式约束：\n"
                f"- 当前阶段：{co_creation.get('current_stage_title') or '未命名阶段'}\n"
                f"- 已锁定字段：{locked_fields}\n"
                f"- 已锁定内容：\n{locked_values}\n"
                "- 绝不擅自推翻已锁定的世界观、角色核心、剧情核心。\n"
                "- 如果用户只要求局部修改，只能精准修改被点名部分。\n"
                "- 如果用户卡壳或请求兜底，可以补当前阶段内容，但仍不能越权改动锁定内容。\n\n"
            )
        return (
            f"当前请求：{latest_request}\n"
            f"标题：{summary.get('title') or self._build_title_from_answers(answers)}\n"
            f"类型：{answers.get('genre') or summary.get('genre') or '待确认'}\n"
            f"时代：{answers.get('era') or summary.get('era') or '待确认'}\n"
            f"人数：{answers.get('players') or '待确认'}\n"
            f"主控视角：{answers.get('protagonist') or '待确认'}\n"
            f"核心冲突：{answers.get('conflict') or '待确认'}\n"
            f"玩家目标：{answers.get('goal') or '待确认'}\n\n"
            f"最近消息：\n{recent_messages}\n\n"
            f"已确认事实：\n{facts}\n\n"
            f"当前草案：\n{draft}\n\n"
            f"角色卡：\n{role_lines}\n\n"
            f"任务队列：\n{task_queue}\n\n"
            f"分幕结构：\n{structure_plan}\n\n"
            f"交付结构：\n{output_outline}\n\n"
            f"记忆节点：\n{memory_nodes}\n\n"
            f"压缩上下文：\n- {compressed_context}\n\n"
            f"{co_creation_section}"
            f"{reference_section}"
            "本轮执行计划：\n"
            f"- 聚焦章节：{'、'.join(execution_plan.get('focus_sections', []) or ['基础设定'])}\n"
            f"- 保护章节：{'、'.join(execution_plan.get('protected_sections', []) or ['既有正文'])}\n"
            f"- 影响角色：{'、'.join(execution_plan.get('impacted_roles', []) or ['待补充'])}\n"
            f"- 影响分幕：{'、'.join(execution_plan.get('impacted_stages', []) or ['待补充'])}\n"
            f"- 影响线索：{'、'.join(execution_plan.get('impacted_clues', []) or ['待补充'])}\n"
            f"- 排除角色：{'、'.join(execution_plan.get('excluded_roles', []) or ['无'])}\n"
            f"- 排除分幕：{'、'.join(execution_plan.get('excluded_stages', []) or ['无'])}\n"
            f"- 排除线索：{'、'.join(execution_plan.get('excluded_clues', []) or ['无'])}\n\n"
            "局部修改策略：\n"
            "- 如果用户只要求补某一部分，就在对应章节深挖，不要整份重写成另一个剧本。\n"
            "- 未被点名修改的既有正文、角色设定和已经成立的因果链，默认视为保护区。\n"
            "- 如果要补长文本剧情，优先补具体场景、角色动作、试探对白和线索触发，不要只拉长提纲。\n\n"
            f"题材专项约束：\n{genre_constraints}\n\n"
            f"{longform_context}\n\n"
            "自检要求：\n"
            "- 输出前确认本轮修改有没有真正落到用户点名的章节、角色、分幕或线索上。\n"
            "- 输出前确认长文本段落里出现了场景、动作、信息差和后果，而不是纯概述。\n"
            "- 输出前确认没有推翻已确认事实，也没有把受保护章节改坏。\n\n"
            "要求：\n"
            "1. 必须真正响应当前请求，不要只复述需求。\n"
            "2. 保留已经确认的设定，避免与现有草案冲突。\n"
            "3. 如果请求指向玩家手册、线索、DM 手册或结局，就重点改写对应部分。\n"
            "4. 输出必须是合法 JSON 对象。\n"
            f"5. {mode_hint}\n"
            "6. 你的回复不能包含 markdown、```json、解释文字或项目符号，只能输出 JSON。\n"
            "7. assistant_reply、draft_content、suggestions、facts、role_cards、summary 这 6 个顶层字段不能缺失。"
        )

    def _build_agent_reference_context(self, memory: dict[str, Any], latest_request: str) -> str:
        query = self._build_agent_reference_query(memory, latest_request)
        if not query:
            return ""
        category = self._resolve_agent_reference_category(latest_request)
        db: Session = SessionLocal()
        try:
            fragments = self.script_reference_service.search_fragments(db, query=query, limit=3, category=category)
            if len(fragments) < 2 and category:
                supplements = self.script_reference_service.search_fragments(db, query=query, limit=3)
                existing_keys = {f"{item['source_path']}::{item['content'][:80]}" for item in fragments}
                for item in supplements:
                    key = f"{item['source_path']}::{item['content'][:80]}"
                    if key in existing_keys:
                        continue
                    fragments.append(item)
                    existing_keys.add(key)
                    if len(fragments) >= 3:
                        break
        finally:
            db.close()
        return self.script_reference_service.build_reference_context(fragments)

    def _build_agent_reference_query(self, memory: dict[str, Any], latest_request: str) -> str:
        summary = memory.get("summary", {})
        facts = " ".join(str(fact) for fact in memory.get("facts", [])[:6])
        role_names = " ".join(str(item.get("name", "")) for item in memory.get("role_cards", [])[:4] if item.get("name"))
        parts = [
            summary.get("title", ""),
            summary.get("genre", ""),
            summary.get("era", ""),
            summary.get("focus", ""),
            latest_request,
            facts,
            role_names,
        ]
        return " ".join(str(item or "").strip() for item in parts if str(item or "").strip())

    def _resolve_agent_reference_category(self, latest_request: str) -> str:
        if self._contains_request_token(latest_request, self.DM_TOKENS):
            return "dm_manual"
        if self._contains_request_token(latest_request, self.CLUE_TOKENS):
            return "clue"
        if self._contains_request_token(latest_request, self.PLAYER_HANDBOOK_TOKENS) or "角色剧本" in latest_request or "角色本" in latest_request:
            return "character"
        return "plot"

    def _ensure_request_section(
        self,
        draft_content: str,
        *,
        latest_request: str,
        title: str,
        conflict: str,
        protagonist: str,
        goal: str,
        role_cards: list[dict[str, Any]],
        memory: dict[str, Any],
    ) -> str:
        cleaned = (draft_content or "").strip()
        if not cleaned:
            return cleaned
        if self._request_targets_player_handbook_story(latest_request) and "玩家手册叙事版" not in cleaned:
            section = self._build_player_handbook_story_section(title, conflict, protagonist, goal, role_cards)
            return f"{cleaned}\n\n{section}".strip()
        if self.longform_story_service.request_targets_longform_story(latest_request) and self.longform_story_service.STORYLINE_SECTION_TITLE not in cleaned:
            section = self.longform_story_service.build_agent_storyline_section(
                title=title,
                conflict=conflict,
                protagonist=protagonist,
                goal=goal,
                role_cards=role_cards,
                memory=memory,
            )
            return f"{cleaned}\n\n{section}".strip()
        if self._contains_request_token(latest_request, self.CLUE_TOKENS) and "线索与搜证补强" not in cleaned:
            section = self._build_clue_expansion_section(conflict, role_cards)
            return f"{cleaned}\n\n{section}".strip()
        if self._contains_request_token(latest_request, self.ENDING_TOKENS) and "结局与反转回收" not in cleaned:
            section = self._build_ending_section(conflict, goal, role_cards)
            return f"{cleaned}\n\n{section}".strip()
        if self._contains_request_token(latest_request, self.DM_TOKENS) and "DM 手册补充" not in cleaned:
            section = self._build_dm_section(conflict, goal)
            return f"{cleaned}\n\n{section}".strip()
        return cleaned

    def _build_title_from_answers(self, answers: dict[str, str]) -> str:
        conflict = answers.get("conflict", "谜局")
        genre = answers.get("genre", "剧本")
        return f"{genre}·{conflict}"[:40]

    def _build_agent_reply(self, title: str, conflict: str, goal: str, latest_request: str = "") -> str:
        request = self._short_text(latest_request, 22)
        if request:
            if self._request_targets_player_handbook_story(latest_request):
                return (
                    f"我已按“{request}”重写《{title}》的玩家手册表达方式，"
                    f"这次把角色视角、分幕推进和真相回收整理成更完整的故事剧情。"
                )
            if self.longform_story_service.request_targets_longform_story(latest_request):
                return f"我已按“{request}”把《{title}》补成更完整的长文本剧情，并强化了场景、互动和线索推进。"
            if self._contains_request_token(latest_request, self.CLUE_TOKENS):
                return f"我已按“{request}”补强《{title}》的线索与搜证结构，让关键证据更能服务“{conflict}”的推进。"
            if self._contains_request_token(latest_request, self.ENDING_TOKENS):
                return f"我已按“{request}”收束《{title}》的结局回收，让最终真相更服务于“{goal}”。"
            if self._contains_request_token(latest_request, self.DM_TOKENS):
                return f"我已按“{request}”补充《{title}》的主持话术和控场节点，方便后续直接拆成 DM 手册。"
            return f"我已按“{request}”继续改写《{title}》，并让后续内容更集中服务于“{goal}”和“{conflict}”。"
        return f"我已围绕《{title}》继续推进草案，重点收束到“{conflict}”这条主线，并让后续内容更服务于“{goal}”。"

    def _merge_facts(self, existing_facts: list[str], answers: dict[str, str], conflict: str, goal: str) -> list[str]:
        merged = [*existing_facts]
        for item in (
            f"剧本类型：{answers.get('genre', '待确认')}",
            f"时代背景：{answers.get('era', '待确认')}",
            f"参与人数：{answers.get('players', '待确认')}",
            f"核心冲突：{conflict}",
            f"玩家目标：{goal}",
        ):
            if item not in merged:
                merged.append(item)
        return merged[:12]

    def _build_role_cards(self, players: str, genre: str, protagonist: str, conflict: str) -> list[dict[str, Any]]:
        count = self._resolve_role_count(players)
        cards: list[dict[str, Any]] = []
        suspense_names = ["沈砚", "林岚", "周野", "许澈", "苏曼", "程澈", "顾宁", "韩舟", "宋妍", "梁川"]
        comedy_names = ["周野", "许澈", "林桃", "顾南枝", "沈悦", "陆一鸣", "苏小满", "何知夏", "程见秋", "叶轻舟"]
        identities = (
            ["记者", "档案管理员", "律师", "医生", "摄影师", "程序员", "主持人", "策展人", "老板", "教师"]
            if genre != "欢乐"
            else ["相亲局主持", "互联网产品经理", "情感博主", "宠物医生", "连锁餐饮店长", "摄影师", "金融顾问", "高中教师", "创业合伙人", "品牌策划"]
        )
        personalities = (
            ["克制冷静", "敏感防御", "嘴硬心细", "表面体面", "强势掌控", "温和疏离", "观察欲强", "情绪压抑", "习惯伪装", "擅长控场"]
            if genre != "欢乐"
            else ["嘴硬心虚", "社牛外壳", "佛系嘴贫", "体面要强", "爱拆台", "极度好胜", "擅长圆场", "自带笑点", "表面温柔", "情绪外放"]
        )
        for index in range(count):
            name_pool = comedy_names if "欢乐" in genre or "相亲" in conflict else suspense_names
            character_name = protagonist if index == 0 and protagonist and protagonist not in {"关键角色", "主控角色"} else name_pool[index % len(name_pool)]
            public_identity = identities[index % len(identities)]
            temperament = personalities[index % len(personalities)]
            cards.append(
                {
                    "id": f"role-{index + 1}",
                    "name": character_name,
                    "archetype": temperament,
                    "public_identity": public_identity,
                    "hidden_secret": f"{character_name}与“{conflict}”存在直接牵连，并且一直在隐瞒会让全场翻盘的一段旧事。",
                    "motivation": f"{character_name}想在局中保护自己最在乎的人，同时逼近“{conflict}”背后的真正真相。",
                    "key_prop": f"{character_name}的关键物证{index + 1}",
                    "tags": [genre or "剧本杀", temperament],
                    "behavior_rules": ["先用公开身份站稳位置", "关键节点再抛出能改变场面的信息"],
                    "relationships": [f"与其他角色之间至少存在一段未公开的旧关系，最终会回收到“{conflict}”。"],
                }
            )
        return cards

    def _build_agent_draft(
        self,
        title: str,
        era: str,
        genre: str,
        conflict: str,
        protagonist: str,
        goal: str,
        role_cards: list[dict[str, Any]],
        memory: dict[str, Any],
    ) -> str:
        facts = "\n".join(f"- {fact}" for fact in memory.get("facts", [])[:6]) or "- 待补充关键事实"
        role_lines = "\n".join(f"- {card['name']}：{card['hidden_secret']}" for card in role_cards[:6])
        latest_request = self._extract_latest_generation_request(memory)
        request_section = self._build_request_specific_section(
            latest_request=latest_request,
            title=title,
            conflict=conflict,
            protagonist=protagonist,
            goal=goal,
            role_cards=role_cards,
            memory=memory,
        )
        return (
            f"# {title}\n\n"
            f"## 本轮执行重点\n"
            f"{self._build_request_summary(latest_request, conflict, goal)}\n\n"
            f"## 基础设定\n"
            f"- 类型：{genre}\n"
            f"- 时代：{era}\n"
            f"- 主控视角：{protagonist}\n"
            f"- 核心冲突：{conflict}\n"
            f"- 玩家目标：{goal}\n\n"
            f"## 世界观设定\n"
            f"故事发生在{era}，表面上这是一场围绕“{conflict}”展开的剧本局，实际上每个角色都带着旧关系、旧债和不敢公开的个人立场进入现场。"
            f" 玩家开场会先看到一层体面关系和公开叙事，随着分幕推进，误导线索、关系裂口和关键物证会一层层把真相扒开。\n\n"
            f"## 核心剧情总览\n"
            f"故事将围绕“{conflict}”展开，通过多角色信息拼图逐步逼近真相。主控角色 {protagonist} 会在每一幕承担推进视角，"
            f"其他角色分别掌握局部事实、误导线索与关系压力。前半段先让玩家围绕公开关系与第一轮怀疑展开互动，"
            f"中段通过搜证、互咬和站队变化抬高冲突，后段再把隐藏动机与关键证据统一回收到“{goal}”这条主线上。\n\n"
            f"## 三幕剧结构\n"
            f"第一幕：抛出公开矛盾与第一轮怀疑，让所有人先围绕表层版本站队。\n"
            f"第二幕：通过线索、私聊和场上对质，把关系裂口和证据矛盾逐步放大。\n"
            f"第三幕：集中回收误导信息、隐藏动机和关键物证，完成真相反转与结局裁定。\n\n"
            f"{request_section}\n\n"
            f"## 已确认事实\n{facts}\n\n"
            f"## 角色摘要\n{role_lines}\n"
        ).strip()

    def _extract_latest_generation_request(self, memory: dict[str, Any]) -> str:
        request = str(memory.get("active_generation_request") or "").strip()
        if request:
            return request
        for message in reversed(memory.get("messages", [])):
            if message.get("role") == "user":
                content = str(message.get("content") or "").strip()
                if content:
                    return content
        return ""

    @staticmethod
    def _short_text(text: str, limit: int = 24) -> str:
        stripped = (text or "").strip()
        if len(stripped) <= limit:
            return stripped
        return stripped[:limit].rstrip() + "..."

    @staticmethod
    def _contains_request_token(text: str, tokens: tuple[str, ...]) -> bool:
        normalized = (text or "").strip().lower()
        return any(token.lower() in normalized for token in tokens)

    def _request_targets_player_handbook_story(self, latest_request: str) -> bool:
        return self._contains_request_token(latest_request, self.PLAYER_HANDBOOK_TOKENS) and self._contains_request_token(latest_request, self.STORY_TOKENS)

    def _build_focus_label(self, latest_request: str) -> str:
        short_request = self._short_text(latest_request, 14)
        if not short_request:
            return "继续完善剧本草案"
        return f"执行：{short_request}"

    def _build_request_summary(self, latest_request: str, conflict: str, goal: str) -> str:
        if not latest_request:
            return f"- 默认继续围绕“{conflict}”推进，并让结构、角色和真相回收共同服务“{goal}”。"
        return (
            f"- 用户最新要求：{latest_request}\n"
            f"- 本轮改写原则：不推翻既有设定，优先把新增内容落到“{conflict}”与“{goal}”的主线上。"
        )

    def _build_request_specific_section(
        self,
        *,
        latest_request: str,
        title: str,
        conflict: str,
        protagonist: str,
        goal: str,
        role_cards: list[dict[str, Any]],
        memory: dict[str, Any],
    ) -> str:
        if self._request_targets_player_handbook_story(latest_request):
            return self._build_player_handbook_story_section(title, conflict, protagonist, goal, role_cards)
        if self.longform_story_service.request_targets_longform_story(latest_request):
            return self.longform_story_service.build_agent_storyline_section(
                title=title,
                conflict=conflict,
                protagonist=protagonist,
                goal=goal,
                role_cards=role_cards,
                memory=memory,
            )
        if self._contains_request_token(latest_request, self.CLUE_TOKENS):
            return self._build_clue_expansion_section(conflict, role_cards)
        if self._contains_request_token(latest_request, self.ENDING_TOKENS):
            return self._build_ending_section(conflict, goal, role_cards)
        if self._contains_request_token(latest_request, self.DM_TOKENS):
            return self._build_dm_section(conflict, goal)
        return self.longform_story_service.build_agent_storyline_section(
            title=title,
            conflict=conflict,
            protagonist=protagonist,
            goal=goal,
            role_cards=role_cards,
            memory=memory,
        )

    def _build_standard_progress_section(
        self,
        conflict: str,
        protagonist: str,
        goal: str,
        role_cards: list[dict[str, Any]],
    ) -> str:
        side_roles = "、".join(card["name"] for card in role_cards[1:4]) if len(role_cards) > 1 else "其余角色"
        return (
            "## 当前推进稿\n"
            f"第一阶段由 {protagonist} 带出表面冲突与场上关系，先让 {side_roles} 分别站到不同立场，制造公开对抗。\n"
            f"第二阶段通过搜证与私聊交换，把每个人与“{conflict}”的真实牵连逐步抬出来，逼迫玩家重新站队。\n"
            f"第三阶段集中回收隐藏动机、证据链和口供矛盾，把所有信息统一推向“{goal}”的终局揭示。"
        )

    def _build_genre_prompt_directives(
        self,
        answers: dict[str, Any],
        summary: dict[str, Any],
        latest_request: str,
    ) -> str:
        genre = str(answers.get("genre") or summary.get("genre") or "").strip()
        conflict = str(answers.get("conflict") or "").strip()
        normalized = f"{genre} {conflict} {latest_request}".lower()
        template_key = str(summary.get("template_key") or "").strip()
        if template_key == "liangyuan-chaos" or any(token in f"{summary.get('title', '')} {genre} {conflict} {latest_request}" for token in ("良缘翻车局", "相亲", "配对", "翻车")):
            return (
                "- 保留轻喜剧节奏，但冲突推进仍要真实成立，不能只堆梗。\n"
                "- 允许出现佛系相亲、嘴硬心虚、社死翻车等本土化互动表达。\n"
                "- 如果用户点名补话术或玩家剧情，要把新手发言模板、破冰梗和关系误会写进正文。 "
            ).strip()
        if "恐怖" in genre:
            return (
                "- 长文本要优先制造压迫感、未知感和迟到的信息回收。\n"
                "- 恐怖氛围应通过场景细节、感官描写和角色反应推进，而不是只靠主持人口播。\n"
                "- 关键线索不要一次给全，优先让玩家先感知异常，再逐步确认异常的来源和代价。"
            ).strip()
        if "情感" in genre:
            return (
                "- 长文本要优先处理关系拉扯、旧事回收和情绪递进，不能只写推理结论。\n"
                "- 每一幕都要让玩家明确：自己在乎谁、误会谁、又为什么迟迟不愿把话说满。\n"
                "- 情感爆点应建立在前文关系链和牺牲感上，而不是只靠临时煽情。"
            ).strip()
        if "悬疑" in genre or "硬核" in genre or "推理" in normalized:
            return (
                "- 维持剧本杀的信息拼图结构，保证线索、口供和角色立场能相互印证。\n"
                "- 至少保留一层公开线索、一层误导线索和一层反转线索，让推理链条能逐步收束。\n"
                "- 长文本剧情要服务互动节奏、推理链条和终局回收。"
            ).strip()
        return (
            "- 维持剧本杀的信息拼图结构，保证线索、口供和角色立场能相互印证。\n"
            "- 长文本剧情要服务互动节奏、推理链条和终局回收。"
        ).strip()

    def _build_player_handbook_story_section(
        self,
        title: str,
        conflict: str,
        protagonist: str,
        goal: str,
        role_cards: list[dict[str, Any]],
    ) -> str:
        role_names = [card["name"] for card in role_cards[:4]]
        partner_preview = "、".join(role_names[1:4]) if len(role_names) > 1 else "其他玩家"
        return (
            "## 玩家手册叙事版\n"
            f"你所拿到的玩家手册，不再只是设定条目，而是一段从 {protagonist} 视角切入的完整故事剧情。\n\n"
            "### 开场处境\n"
            f"故事开始时，你和 {partner_preview} 都被卷入同一场围绕“{conflict}”的局中局。表面上大家还在维持体面关系，"
            "但每个人都带着不能公开的历史和立场，局面从第一轮发言开始就已经失衡。\n\n"
            "### 第一幕：关系破冰\n"
            f"{protagonist} 会先接触最容易产生冲突的角色，让玩家在闲聊、试探和公开抬杠里逐步摸清谁在撒谎，谁在故意转移视线。"
            "这一幕的玩家阅读体验要像在进入一场已经暗流涌动的人际修罗场，重点是先看懂自己为什么不得不下场。\n\n"
            "### 第二幕：秘密升级\n"
            f"随着线索发放和私聊推进，玩家会发现自己知道的版本并不完整：有人隐瞒了和“{conflict}”直接相关的旧事，"
            "有人把个人动机伪装成集体立场，还有人一直在等最合适的时间把局面彻底掀翻。此处应把玩家手册写成连续剧情，"
            "让读者清楚自己此刻该怀疑谁、维护谁、又为什么不能把全部信息一次性说完。\n\n"
            "### 第三幕：真相收束\n"
            f"当所有人的公开叙事开始互相冲撞时，玩家手册需要把前面埋下的个人关系、误导信息和关键证据统一回收到“{goal}”上，"
            "让玩家读完就能理解自己的最终抉择、站队理由和情绪爆点，而不是只看到一堆散碎设定。\n\n"
            "### 写法要求\n"
            f"- 每个玩家手册都要保留角色口吻和视角偏见。\n"
            "- 要让读者一页页读下去像在追自己的故事线，而不是在看设定清单。\n"
            "- 每一幕都要明确：你知道什么、你误会了什么、你必须隐瞒什么、你准备在场上争什么。"
        )

    def _build_clue_expansion_section(self, conflict: str, role_cards: list[dict[str, Any]]) -> str:
        focus_roles = "、".join(card["name"] for card in role_cards[:3]) or "核心角色"
        return (
            "## 线索与搜证补强\n"
            f"- 围绕“{conflict}”设置三层证据：公开证据、误导证据、反转证据。\n"
            f"- 第一轮优先放出能指向 {focus_roles} 的表层线索，让玩家先形成错误怀疑链。\n"
            "- 第二轮再给出能推翻前述判断的细节证据，逼玩家回头重读口供和关系。\n"
            "- 终局证据必须同时解释动机、时序和行为后果，不能只起到补丁作用。"
        )

    def _build_ending_section(self, conflict: str, goal: str, role_cards: list[dict[str, Any]]) -> str:
        pivot_role = role_cards[0]["name"] if role_cards else "主控角色"
        return (
            "## 结局与反转回收\n"
            f"- 终局揭示时先让 {pivot_role} 面对自己此前最相信的一条叙事被推翻。\n"
            f"- 再用关键证据证明“{conflict}”从一开始就不是表面看到的样子，把所有角色动机回收到“{goal}”。\n"
            "- 结尾必须同时交代：谁在利用局面、谁在掩盖真相、谁真正承受了代价。"
        )

    def _build_dm_section(self, conflict: str, goal: str) -> str:
        return (
            "## DM 手册补充\n"
            f"- 开场提示：先用一句能快速点燃“{conflict}”的引导语把玩家拽进局中。\n"
            "- 转场提示：每次发放新信息时，要明确告诉玩家这一轮应该围绕什么冲突继续打。\n"
            f"- 收束提示：在终局前帮玩家把讨论焦点收回“{goal}”，避免场上只剩无效争吵。"
        )

    def _build_suggestions(self, conflict: str, role_cards: list[dict[str, Any]]) -> list[dict[str, str]]:
        lead_role = role_cards[0]["name"] if role_cards else "主控角色"
        return [
            {
                "id": "suggestion-1",
                "label": "生成完整可用剧本",
                "prompt": "请直接基于当前设定生成一版完整可上桌的剧本杀成稿，正文里要同时覆盖剧情长稿、分幕流程、角色关系、搜证线索、DM 关键话术、玩家手册视角和终局真相回收，尽量做到不需要我再去交付包里二次编辑。",
                "category": "create",
            },
            {"id": "suggestion-2", "label": "补全角色关系", "prompt": f"请继续补全围绕“{conflict}”的角色关系链，重点加强 {lead_role} 与其他人的冲突。", "category": "create"},
            {"id": "suggestion-3", "label": "生成剧情长稿", "prompt": "请把当前剧本的核心剧情写成长文本正文，补足场景、人物动作、试探对白和线索推进。", "category": "create"},
            {"id": "suggestion-4", "label": "生成角色剧本", "prompt": "请基于当前设定批量生成每个角色的长剧情角色剧本，保持与主持手册、真相页和线索链一致。", "category": "workflow"},
            {"id": "suggestion-5", "label": "补充 DM 话术", "prompt": "请基于当前草案补充 DM 开场、转场和应急控场话术。", "category": "workflow"},
        ]

    @staticmethod
    def _resolve_role_count(players: str) -> int:
        numbers = [int(item) for item in re.findall(r"\d+", players)]
        if not numbers:
            return 6
        return max(3, min(numbers[0], 10))

    @staticmethod
    def _extract_draft_sections(draft_content: str) -> list[str]:
        sections: list[str] = []
        for line in draft_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                sections.append(stripped.lstrip("# "))
            elif stripped.startswith("【") and stripped.endswith("】") and len(stripped) > 2:
                sections.append(stripped[1:-1].strip())
        return sections

    def _build_agent_execution_plan(self, memory: dict[str, Any]) -> dict[str, Any]:
        latest_request = (memory.get("active_generation_request") or "").strip()
        if not latest_request:
            latest_request = next(
                (str(item.get("content") or "").strip() for item in reversed(memory.get("messages", [])) if item.get("role") == "user" and item.get("content")),
                "",
            )
        role_cards = memory.get("role_cards", [])
        role_names = [card.get("name", "") for card in role_cards if card.get("name")]
        sections = self._extract_draft_sections(memory.get("draft_content", ""))
        request_text = latest_request.replace("：", ":")
        focus_sections = [section for section in sections if section and any(token in request_text for token in section.split())]
        if not focus_sections:
            if "线索" in request_text or "搜证" in request_text or "证据" in request_text:
                focus_sections = [section for section in sections if any(token in section for token in ("线索", "互动", "搜证"))]
            elif "人物" in request_text or "秘密" in request_text or "身份" in request_text:
                focus_sections = [section for section in sections if any(token in section for token in ("人物", "角色", "关系"))]
        protected_sections = [section for section in sections if any(token in request_text for token in ("保留", "不动", "不要改动")) and section not in focus_sections]
        if "结局" in request_text and "不动" in request_text:
            protected_sections.extend([section for section in sections if "结局" in section])
        clue_candidates = [str(fact) for fact in memory.get("facts", [])[:3]]
        clue_candidates.extend(str(card.get("key_prop") or "").strip() for card in role_cards if str(card.get("key_prop") or "").strip())
        impacted_roles = [name for name in role_names if name and name in request_text][:3] or role_names[:3]
        impacted_stages = [str(item.get("title") or "").strip() for item in memory.get("structure_plan", []) if str(item.get("title") or "").strip() and str(item.get("title") or "").strip() in request_text]
        if not impacted_stages:
            impacted_stages = [item.get("title", "") for item in memory.get("structure_plan", [])[:3]]
        impacted_clues = [item for item in clue_candidates if item and item in request_text][:3] or [item for item in clue_candidates if item][:3]
        excluded_roles = re.findall(r"不要改动角色[:：]([^\n]+)", request_text)
        excluded_stages = re.findall(r"不要改动分幕[:：]([^\n]+)", request_text)
        excluded_clues = re.findall(r"不要改动线索[:：]([^\n]+)", request_text)
        impacted_roles = [item for item in impacted_roles if item not in excluded_roles]
        impacted_stages = [item for item in impacted_stages if item not in excluded_stages]
        impacted_clues = [item for item in impacted_clues if item not in excluded_clues]
        return {
            "latest_request": latest_request,
            "focus_sections": focus_sections[:3] or sections[:3] or ["基础设定", "当前草案"],
            "protected_sections": protected_sections[:3] or sections[3:6] or ["已确认事实"],
            "impacted_roles": impacted_roles,
            "impacted_stages": impacted_stages,
            "impacted_clues": impacted_clues,
            "excluded_roles": [item.strip() for item in excluded_roles if item.strip()],
            "excluded_stages": [item.strip() for item in excluded_stages if item.strip()],
            "excluded_clues": [item.strip() for item in excluded_clues if item.strip()],
            "success_criteria": [
                "保持既有设定不冲突",
                "补全关键人物关系",
                "让后续内容可继续迭代",
            ],
        }

    def _post_process_agent_turn_payload(self, payload: dict[str, Any], memory: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
        next_payload = copy.deepcopy(payload)
        logic_checks = list(next_payload.get("logic_checks", []))
        if plan.get("focus_sections"):
            logic_checks.append(f"执行计划聚焦：{'、'.join(plan['focus_sections'])}")
        next_payload["logic_checks"] = logic_checks
        next_payload["execution_plan"] = plan
        next_payload["self_check"] = {"passed": True, "must_keep_facts": plan.get("must_keep_facts", [])}
        return next_payload

    def _build_agent_followup_line(self, genre: str) -> str:
        if "欢乐" in genre:
            return "下一轮优先补配对玩法、互动节奏和翻车笑点。"
        if "恐怖" in genre:
            return "下一轮优先补恐怖氛围、异常细节和压迫推进。"
        return "下一轮优先补推理链条、互动节奏和终局回收。"

    def sync_draft_role_card_section(self, draft_content: str, role_cards: list[dict[str, Any]]) -> str:
        cleaned = draft_content.strip()
        role_section = self._build_role_card_section(role_cards)
        if not cleaned:
            return role_section
        if self.ROLE_SECTION_TITLE in cleaned:
            cleaned = cleaned.split(self.ROLE_SECTION_TITLE, 1)[0].strip()
        return f"{cleaned}\n\n{role_section}".strip()

    def _build_role_card_section(self, role_cards: list[dict[str, Any]]) -> str:
        if not role_cards:
            return self.ROLE_SECTION_TITLE
        lines = [self.ROLE_SECTION_TITLE]
        for index, card in enumerate(role_cards, start=1):
            lines.extend(
                [
                    f"{index}. {card.get('name', '角色')}（{card.get('archetype', '人物')}）",
                    f"   公开身份：{card.get('public_identity', '')}",
                    f"   隐藏秘密：{card.get('hidden_secret', '')}",
                    f"   核心动机：{card.get('motivation', '')}",
                    f"   关键道具：{card.get('key_prop', '')}",
                ]
            )
        return "\n".join(lines).strip()

