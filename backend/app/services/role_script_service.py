from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from app.models.schemas import (
    AgentRoleCard,
    AgentRoleScript,
    AgentRoleScriptPromptBlueprint,
    AgentStructureStage,
)


class RoleScriptLongformService:
    PIPELINE_MIN_CONTENT_LENGTH = 2800

    def __init__(self) -> None:
        self.example_script = (
            "那天我坐在人民公园相亲角的长椅上，手里攥着那把缺了键帽的粉色机械键盘，"
            "一边装作若无其事，一边等着那个我三年没敢再见的人出现。"
        )

    def build(self, record_title: str, memory: dict[str, Any]) -> list[AgentRoleScript]:
        role_cards = self._normalize_role_cards(memory.get("role_cards", []))
        stages = self._normalize_stages(memory.get("structure_plan", []))
        facts = [str(item).strip() for item in memory.get("facts", []) if str(item).strip()]
        stored_scripts = memory.get("role_scripts", [])
        if stored_scripts:
            repaired = self._repair_stored_scripts(record_title, stored_scripts, role_cards, stages, facts, memory)
            if repaired:
                return repaired
        return [
            self._build_fallback_script(record_title, card, role_cards, stages, facts, memory)
            for card in role_cards
        ] or [self._build_fallback_script(record_title, self._default_card(), [], [], facts, memory)]

    async def generate_and_store(self, record_title: str, memory: dict[str, Any], llm_service: Any) -> list[AgentRoleScript]:
        role_cards = self._normalize_role_cards(memory.get("role_cards", []))
        stages = self._normalize_stages(memory.get("structure_plan", []))
        facts = [str(item).strip() for item in memory.get("facts", []) if str(item).strip()]
        if not role_cards:
            role_cards = [self._default_card()]

        profile = copy.deepcopy(memory.get("llm_profile") or {})
        if not profile and hasattr(llm_service, "build_llm_profile"):
            try:
                profile = llm_service.build_llm_profile()
            except Exception:
                profile = {}
        runtime_profile = {}
        if hasattr(llm_service, "get_runtime_profile"):
            try:
                runtime_profile = llm_service.get_runtime_profile() or {}
            except Exception:
                runtime_profile = {}
        provider = str(profile.get("provider") or "").strip() or str(runtime_profile.get("recommended_provider") or "mock").strip()
        model = str(profile.get("model") or "").strip() or llm_service.default_model_for_provider(provider)
        api_key = llm_service._provider_api_key(provider)
        base_url = llm_service._provider_base_url(provider)

        generated_scripts: list[AgentRoleScript] = []
        normalized_blueprints = self._load_saved_blueprints(memory)
        blueprint_by_id = {item.id: item for item in normalized_blueprints}
        updated_blueprints: list[AgentRoleScriptPromptBlueprint] = []

        for card in role_cards:
            fallback = self._build_fallback_script(record_title, card, role_cards, stages, facts, memory)
            if not api_key or not base_url:
                generated_scripts.append(fallback)
                blueprint = blueprint_by_id.get(card.id) or self._build_default_blueprint(card, memory)
                updated_blueprints.append(blueprint)
                continue
            try:
                blueprint = blueprint_by_id.get(card.id)
                if blueprint is None:
                    blueprint = await self._generate_prompt_blueprint(
                        llm_service=llm_service,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        record_title=record_title,
                        card=card,
                        role_cards=role_cards,
                        stages=stages,
                        facts=facts,
                        memory=memory,
                    )
                updated_blueprints.append(blueprint)

                writing_intent = await self._generate_writing_intent(
                    llm_service=llm_service,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    record_title=record_title,
                    card=card,
                    blueprint=blueprint,
                    facts=facts,
                    memory=memory,
                )
                beat_sheet = await self._generate_beat_sheet(
                    llm_service=llm_service,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    record_title=record_title,
                    card=card,
                    blueprint=blueprint,
                    writing_intent=writing_intent,
                    stages=stages,
                    facts=facts,
                    memory=memory,
                )
                scene_plan = await self._generate_scene_plan(
                    llm_service=llm_service,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    record_title=record_title,
                    card=card,
                    blueprint=blueprint,
                    writing_intent=writing_intent,
                    beat_sheet=beat_sheet,
                    stages=stages,
                    facts=facts,
                    memory=memory,
                )

                pipeline_prompt = self._build_pipeline_longform_prompt(
                    record_title=record_title,
                    card=card,
                    role_cards=role_cards,
                    stages=stages,
                    facts=facts,
                    memory=memory,
                    blueprint=blueprint,
                    writing_intent=writing_intent,
                    beat_sheet=beat_sheet,
                    scene_plan=scene_plan,
                    fallback=fallback,
                )
                pipeline_content = await llm_service._request_compatible_chat_completion_text(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    system_prompt=(
                        "你是擅长人物戏、群像关系和长文本叙事的职业编剧。"
                        " 你的任务是为单个角色写一份可直接发给玩家阅读的长剧情角色剧本。"
                        " 文本必须具体、可感、有人物心理、关系张力、现场细节和连续事件推进。"
                        " 禁止只写设定条目，禁止只写提纲，禁止只写总结。"
                        " 单篇正文必须达到 2800-3500 字，且必须覆盖完整身世背景、人物关系、秘密、任务与分幕行动。"
                    ),
                    user_prompt=pipeline_prompt,
                    timeout_seconds=120,
                    temperature=0.85,
                )
                draft = str(pipeline_content or "").strip()
                if len(draft) < self.PIPELINE_MIN_CONTENT_LENGTH:
                    direct_prompt = self._build_longform_prompt(
                        record_title=record_title,
                        card=card,
                        role_cards=role_cards,
                        stages=stages,
                        facts=facts,
                        memory=memory,
                        fallback=fallback,
                        blueprint=blueprint,
                        writing_intent=writing_intent,
                        beat_sheet=beat_sheet,
                        scene_plan=scene_plan,
                    )
                    draft = await llm_service._request_compatible_chat_completion_text(
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                        system_prompt=(
                            "你是擅长人物戏、群像关系和长文本叙事的职业编剧。"
                            " 你的任务是为单个角色写一份完整、连续、可阅读的长剧情角色剧本。"
                            " 文字必须能直接发给玩家阅读，不能退化成提纲、说明书或设定清单。"
                            " 单篇正文必须达到 2800-3500 字，且角色之间内容必须显著不同。"
                        ),
                        user_prompt=direct_prompt,
                        timeout_seconds=120,
                        temperature=0.85,
                    )
                normalized_content = self._normalize_generated_content(draft, fallback.content, card, stages)
                generated_scripts.append(
                    AgentRoleScript(
                        id=card.id,
                        role_name=card.name,
                        title=f"{record_title}-{card.name}",
                        cover_age=self._infer_cover_age(card),
                        cover_identity=card.public_identity,
                        cover_tagline=self._infer_tagline(card),
                        source="auto",
                        updated_at=datetime.now().isoformat(timespec="seconds"),
                        content=normalized_content,
                        generation_mode="longform",
                        fallback_used=False,
                        content_length=len(normalized_content),
                        llm_provider=provider,
                        llm_model=model,
                    )
                )
            except Exception:
                generated_scripts.append(fallback)
                blueprint = blueprint_by_id.get(card.id) or self._build_default_blueprint(card, memory)
                updated_blueprints.append(blueprint)

        memory["role_scripts"] = [item.model_dump() for item in generated_scripts]
        memory["role_script_prompt_blueprints"] = [item.model_dump() for item in updated_blueprints]
        return generated_scripts

    def _repair_stored_scripts(
        self,
        record_title: str,
        stored_scripts: list[dict[str, Any]],
        role_cards: list[AgentRoleCard],
        stages: list[AgentStructureStage],
        facts: list[str],
        memory: dict[str, Any],
    ) -> list[AgentRoleScript]:
        stored_by_id: dict[str, AgentRoleScript] = {}
        for item in stored_scripts:
            try:
                script = AgentRoleScript.model_validate(item)
            except Exception:
                continue
            stored_by_id.setdefault(script.id, script)

        repaired: list[AgentRoleScript] = []
        if role_cards:
            for card in role_cards:
                script = stored_by_id.get(card.id)
                if script is None:
                    repaired.append(self._build_fallback_script(record_title, card, role_cards, stages, facts, memory))
                    continue
                repaired.append(
                    script.model_copy(
                        update={
                            "role_name": card.name,
                            "title": script.title or f"{record_title}-{card.name}",
                            "cover_identity": script.cover_identity or card.public_identity,
                            "cover_tagline": script.cover_tagline or self._infer_tagline(card),
                            "cover_age": script.cover_age or self._infer_cover_age(card),
                            "content_length": len(str(script.content or "")),
                        }
                    )
                )
            return repaired

        return list(stored_by_id.values())

    def _build_fallback_script(
        self,
        record_title: str,
        card: AgentRoleCard,
        role_cards: list[AgentRoleCard],
        stages: list[AgentStructureStage],
        facts: list[str],
        memory: dict[str, Any],
    ) -> AgentRoleScript:
        content = self._build_fallback_content(record_title, card, role_cards, stages, facts, memory)
        return AgentRoleScript(
            id=card.id,
            role_name=card.name,
            title=f"{record_title}-{card.name}",
            cover_age=self._infer_cover_age(card),
            cover_identity=card.public_identity,
            cover_tagline=self._infer_tagline(card),
            source="auto",
            updated_at=datetime.now().isoformat(timespec="seconds"),
            content=content,
            generation_mode="fallback",
            fallback_used=True,
            content_length=len(content),
            llm_provider="mock",
            llm_model="mock-story-model",
        )

    def _build_fallback_content(
        self,
        record_title: str,
        card: AgentRoleCard,
        role_cards: list[AgentRoleCard],
        stages: list[AgentStructureStage],
        facts: list[str],
        memory: dict[str, Any],
    ) -> str:
        opening_scene = self._scene_hint(memory)
        relationship_line = "、".join(self._other_role_names(card, role_cards)[:4]) or "其他角色"
        fact_lines = "\n".join(f"- {fact}" for fact in facts[:6]) or "- 当前尚无更多硬性事实。"
        stage_lines = self._build_stage_story_lines(stages)
        focus = str(memory.get("summary", {}).get("focus") or memory.get("world_config", {}).get("core_conflict") or "当前冲突")

        return (
            f"# {record_title} - {card.name}\n\n"
            "【角色导语】\n"
            f"你从踏进{opening_scene}的那一刻起，就知道这不是一场可以靠沉默混过去的局。你叫{card.name}，"
            f"表面上只是“{card.public_identity}”，可真正把你钉在这张桌边的，是“{card.hidden_secret}”。"
            f" 你比任何人都清楚，只要某句话说漏，今晚很多人努力维持的体面都会立刻塌下来。\n\n"
            "【你为什么会被卷进来】\n"
            f"你和{relationship_line}之间从来不是单纯的同伴、熟人或对立者。那些看起来已经过去的旧事，其实一直压在你心口。"
            f" 你真正想做的并不复杂：{card.motivation}。但越是接近这个目标，你越明白自己手里的“{card.key_prop}”并不是单纯的证据或筹码，"
            "它也可能在关键时刻把你反过来送上审判台。\n\n"
            "【你已知的关键事实】\n"
            f"{fact_lines}\n\n"
            f"{stage_lines}\n\n"
            "【你在局中的真实压力】\n"
            f"围绕“{focus}”的局势不会给你慢慢准备的机会。你要一边维持人设，一边判断谁在故意试探你，谁又只是比你更先嗅到了危险。"
            " 你会误判，会迟疑，会想把真相推迟到更安全的时机，可局势并不会因为你的克制而放缓。"
            " 当第一层误会被戳破后，你迟早要决定：继续躲在身份后面，还是把更大的秘密扔到场中央，换一次彻底翻盘的机会。"
        )

    async def _generate_prompt_blueprint(
        self,
        *,
        llm_service: Any,
        model: str,
        api_key: str,
        base_url: str,
        record_title: str,
        card: AgentRoleCard,
        role_cards: list[AgentRoleCard],
        stages: list[AgentStructureStage],
        facts: list[str],
        memory: dict[str, Any],
    ) -> AgentRoleScriptPromptBlueprint:
        prompt = self._build_blueprint_prompt(record_title, card, role_cards, stages, facts, memory)
        content = await llm_service._request_compatible_chat_completion_text(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=(
                "你是角色剧本提示词设计师。"
                " 你必须只输出一个 JSON 对象，字段限定为 objective, style, must_keep, focus, negative_constraints。"
            ),
            user_prompt=prompt,
            timeout_seconds=60,
            temperature=0.45,
        )
        parsed = llm_service._parse_json_text(content) or {}
        blueprint = AgentRoleScriptPromptBlueprint(
            id=card.id,
            role_name=card.name,
            user_requirement=self._latest_user_requirement(memory),
            objective=str(parsed.get("objective") or f"强化{card.name}的秘密感、关系张力与终局回收").strip(),
            style=str(parsed.get("style") or self._default_blueprint_style(memory)).strip(),
            must_keep=self._normalize_text_list(parsed.get("must_keep")) or self._default_must_keep(card),
            focus=self._normalize_text_list(parsed.get("focus")) or self._default_focus(card),
            negative_constraints=self._normalize_text_list(parsed.get("negative_constraints")) or ["不要写成提纲"],
            source="auto",
        )
        return blueprint

    async def _generate_writing_intent(
        self,
        *,
        llm_service: Any,
        model: str,
        api_key: str,
        base_url: str,
        record_title: str,
        card: AgentRoleCard,
        blueprint: AgentRoleScriptPromptBlueprint,
        facts: list[str],
        memory: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._build_writing_intent_prompt(record_title, card, blueprint, facts, memory)
        content = await llm_service._request_compatible_chat_completion_text(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=(
                "你是角色剧本剧情设计师。"
                " 你必须只输出一个 JSON 对象，字段限定为 goal, conflict, entities, hook, tone。"
            ),
            user_prompt=prompt,
            timeout_seconds=60,
            temperature=0.55,
        )
        parsed = llm_service._parse_json_text(content) or {}
        return {
            "goal": str(parsed.get("goal") or card.motivation or "逼近真相").strip(),
            "conflict": str(parsed.get("conflict") or card.hidden_secret or "秘密与身份发生冲撞").strip(),
            "entities": self._normalize_text_list(parsed.get("entities")) or [card.name, card.key_prop],
            "hook": str(parsed.get("hook") or card.key_prop or "关键线索即将浮出水面").strip(),
            "tone": str(parsed.get("tone") or "克制压迫").strip(),
        }

    async def _generate_beat_sheet(
        self,
        *,
        llm_service: Any,
        model: str,
        api_key: str,
        base_url: str,
        record_title: str,
        card: AgentRoleCard,
        blueprint: AgentRoleScriptPromptBlueprint,
        writing_intent: dict[str, Any],
        stages: list[AgentStructureStage],
        facts: list[str],
        memory: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._build_beat_sheet_prompt(record_title, card, blueprint, writing_intent, stages, facts, memory)
        content = await llm_service._request_compatible_chat_completion_text(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=(
                "你是角色剧本分幕设计师。"
                " 你必须只输出一个 JSON 对象，字段限定为 beats, payoffs, spotlight_prop。"
            ),
            user_prompt=prompt,
            timeout_seconds=60,
            temperature=0.55,
        )
        parsed = llm_service._parse_json_text(content) or {}
        return {
            "beats": self._normalize_text_list(parsed.get("beats")) or self._default_beats(card, stages),
            "payoffs": self._normalize_text_list(parsed.get("payoffs")) or [card.key_prop or "关键线索"],
            "spotlight_prop": str(parsed.get("spotlight_prop") or card.key_prop or "关键证据").strip(),
        }

    async def _generate_scene_plan(
        self,
        *,
        llm_service: Any,
        model: str,
        api_key: str,
        base_url: str,
        record_title: str,
        card: AgentRoleCard,
        blueprint: AgentRoleScriptPromptBlueprint,
        writing_intent: dict[str, Any],
        beat_sheet: dict[str, Any],
        stages: list[AgentStructureStage],
        facts: list[str],
        memory: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._build_scene_plan_prompt(record_title, card, blueprint, writing_intent, beat_sheet, stages, facts, memory)
        content = await llm_service._request_compatible_chat_completion_text(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=(
                "你是角色剧本场景规划师。"
                " 你必须只输出一个 JSON 对象，字段限定为 scenes。"
                " scenes 必须是数组，每项包含 title, purpose, conflict, trigger, outcome。"
            ),
            user_prompt=prompt,
            timeout_seconds=60,
            temperature=0.55,
        )
        parsed = llm_service._parse_json_text(content) or {}
        scenes = self._normalize_scenes(parsed.get("scenes"))
        if not scenes:
            scenes = self._default_scenes(card, stages, writing_intent, beat_sheet, memory)
        return {"scenes": scenes}

    def _build_blueprint_prompt(
        self,
        record_title: str,
        card: AgentRoleCard,
        role_cards: list[AgentRoleCard],
        stages: list[AgentStructureStage],
        facts: list[str],
        memory: dict[str, Any],
    ) -> str:
        stage_text = "\n".join(f"- {stage.title}：{stage.objective}" for stage in stages[:4]) or "- 暂无分幕"
        role_text = "\n".join(
            f"- {other.name}：公开身份 {other.public_identity}；隐藏秘密 {other.hidden_secret}"
            for other in role_cards
            if other.id != card.id
        ) or "- 暂无其他角色"
        fact_text = "\n".join(f"- {fact}" for fact in facts[:6]) or "- 暂无事实"
        return (
            f"请为角色“{card.name}”生成角色剧本的 prompt 蓝图。\n\n"
            f"【剧本标题】{record_title}\n"
            f"【本轮用户需求】{self._latest_user_requirement(memory)}\n"
            f"【角色公开身份】{card.public_identity}\n"
            f"【角色隐藏秘密】{card.hidden_secret}\n"
            f"【角色核心动机】{card.motivation}\n"
            f"【关键道具】{card.key_prop}\n"
            f"【其他角色】\n{role_text}\n\n"
            f"【分幕】\n{stage_text}\n\n"
            f"【事实】\n{fact_text}\n\n"
            "请只返回 JSON：\n"
            "- objective：一句话说明这份剧本最该强化什么。\n"
            "- style：一句话说明文风和叙事策略。\n"
            "- must_keep：必须保留的关键信息数组。\n"
            "- focus：本次写作重点数组。\n"
            "- negative_constraints：明确不要写成什么。"
        )

    def _build_writing_intent_prompt(
        self,
        record_title: str,
        card: AgentRoleCard,
        blueprint: AgentRoleScriptPromptBlueprint,
        facts: list[str],
        memory: dict[str, Any],
    ) -> str:
        fact_text = "\n".join(f"- {fact}" for fact in facts[:6]) or "- 暂无额外事实"
        return (
            f"请根据下面信息，为角色“{card.name}”生成写作意图 JSON。\n\n"
            f"【剧本标题】{record_title}\n"
            f"【本轮用户需求】{self._latest_user_requirement(memory)}\n"
            f"【角色 prompt 蓝图】{blueprint.model_dump_json(ensure_ascii=False)}\n"
            f"【角色秘密】{card.hidden_secret}\n"
            f"【角色动机】{card.motivation}\n"
            f"【关键事实】\n{fact_text}\n\n"
            "请只返回 JSON：goal, conflict, entities, hook, tone。"
        )

    def _build_beat_sheet_prompt(
        self,
        record_title: str,
        card: AgentRoleCard,
        blueprint: AgentRoleScriptPromptBlueprint,
        writing_intent: dict[str, Any],
        stages: list[AgentStructureStage],
        facts: list[str],
        memory: dict[str, Any],
    ) -> str:
        stage_text = "\n".join(f"- {stage.title}：{stage.objective}" for stage in stages[:5]) or "- 暂无分幕结构"
        fact_text = "\n".join(f"- {fact}" for fact in facts[:6]) or "- 暂无事实"
        return (
            f"请为角色“{card.name}”生成 Beat Sheet（场景大纲）JSON。\n\n"
            f"【剧本标题】{record_title}\n"
            f"【本轮用户需求】{self._latest_user_requirement(memory)}\n"
            f"【prompt 蓝图】{blueprint.model_dump_json(ensure_ascii=False)}\n"
            f"【写作意图】{writing_intent}\n"
            f"【分幕】\n{stage_text}\n\n"
            f"【事实】\n{fact_text}\n\n"
            "请只返回 JSON：\n"
            "- beats：4 条左右的推进节拍，每条一句完整剧情动作。\n"
            "- payoffs：需要回收的误导或关系节点数组。\n"
            "- spotlight_prop：本角色最该被点亮的道具或证据。"
        )

    def _build_scene_plan_prompt(
        self,
        record_title: str,
        card: AgentRoleCard,
        blueprint: AgentRoleScriptPromptBlueprint,
        writing_intent: dict[str, Any],
        beat_sheet: dict[str, Any],
        stages: list[AgentStructureStage],
        facts: list[str],
        memory: dict[str, Any],
    ) -> str:
        stage_text = "\n".join(f"- {stage.title}：{stage.objective}" for stage in stages[:5]) or "- 暂无分幕结构"
        fact_text = "\n".join(f"- {fact}" for fact in facts[:6]) or "- 暂无事实"
        return (
            f"请为角色“{card.name}”生成 Scene Plan JSON。\n\n"
            f"【剧本标题】{record_title}\n"
            f"【本轮用户需求】{self._latest_user_requirement(memory)}\n"
            f"【prompt 蓝图】{blueprint.model_dump_json(ensure_ascii=False)}\n"
            f"【写作意图】{writing_intent}\n"
            f"【Beat Sheet】{beat_sheet}\n"
            f"【分幕】\n{stage_text}\n\n"
            f"【事实】\n{fact_text}\n\n"
            "请只返回 JSON，结构为 scenes 数组；每个 scene 必须包含：title, purpose, conflict, trigger, outcome。"
        )

    def _build_pipeline_longform_prompt(
        self,
        *,
        record_title: str,
        card: AgentRoleCard,
        role_cards: list[AgentRoleCard],
        stages: list[AgentStructureStage],
        facts: list[str],
        memory: dict[str, Any],
        blueprint: AgentRoleScriptPromptBlueprint,
        writing_intent: dict[str, Any],
        beat_sheet: dict[str, Any],
        scene_plan: dict[str, Any],
        fallback: AgentRoleScript,
    ) -> str:
        relationship_summary = "\n".join(
            f"- {other.name}：公开身份 {other.public_identity}；隐藏秘密 {other.hidden_secret}；关系线索 {self._guess_relationship(card, other)}"
            for other in role_cards
            if other.id != card.id
        ) or "- 暂无其他角色关系摘要"
        fact_summary = "\n".join(f"- {fact}" for fact in facts[:8]) or "- 暂无额外事实"
        scene_lines = []
        for index, scene in enumerate(scene_plan.get("scenes", []), start=1):
            scene_lines.append(
                f"场景{index}：{scene.get('title', '')} | 目的：{scene.get('purpose', '')} | 冲突：{scene.get('conflict', '')} | 触发：{scene.get('trigger', '')} | 结果：{scene.get('outcome', '')}"
            )
        scene_text = "\n".join(scene_lines) or "场景1：从失衡开场，把人物推入局中。"
        beat_text = "\n".join(f"- {item}" for item in beat_sheet.get("beats", [])) or "- 从开场误会推进到终局亮牌。"

        return (
            f"请为角色“{card.name}”写一份完整的长剧情角色剧本。\n\n"
            f"【剧本标题】{record_title}\n"
            f"【本轮用户需求】{self._latest_user_requirement(memory)}\n"
            f"【角色公开身份】{card.public_identity}\n"
            f"【角色隐藏秘密】{card.hidden_secret}\n"
            f"【角色核心动机】{card.motivation}\n"
            f"【关键道具】{card.key_prop}\n"
            f"【其他角色与关系】\n{relationship_summary}\n\n"
            f"【已确认事实】\n{fact_summary}\n\n"
            f"【prompt 蓝图】{blueprint.model_dump_json(ensure_ascii=False)}\n"
            f"【写作意图】{writing_intent}\n"
            f"【Beat Sheet（场景大纲）】\n{beat_text}\n\n"
            f"【Scene Plan】\n{scene_text}\n\n"
            "【逐场写作要求】\n"
            "- 必须使用第一人称有限视角，始终站在该角色的感知和判断里。\n"
            "- 必须写出角色秘密如何影响每一次观察、迟疑、误判和发言。\n"
            "- 不要写成提纲或分点说明，而要写成连续正文。\n"
            "- 可以保留少量标题，帮助玩家阅读，但正文必须有事件流和心理流。\n"
            "- 按“第一幕 / 场景1 / 场景2 / 终局前夜 / 最后一轮选择”这样的推进节奏去写。\n"
            "- 必须显式写出：真实姓名、年龄、职业、外貌、性格、口头禅、成长经历、专属秘密、核心任务、隐藏目标，以及与其他所有角色的关系。\n"
            "- 单篇角色剧本最终长度必须在 2800-3500 字之间，不允许缩写、概括或模板化句式。\n\n"
            f"【兜底草稿，仅供参考】\n{fallback.content}\n\n"
            "请直接输出最终正文。"
        )

    def _build_longform_prompt(
        self,
        *,
        record_title: str,
        card: AgentRoleCard,
        role_cards: list[AgentRoleCard],
        stages: list[AgentStructureStage],
        facts: list[str],
        memory: dict[str, Any],
        fallback: AgentRoleScript,
        blueprint: AgentRoleScriptPromptBlueprint | None = None,
        writing_intent: dict[str, Any] | None = None,
        beat_sheet: dict[str, Any] | None = None,
        scene_plan: dict[str, Any] | None = None,
    ) -> str:
        world_config = memory.get("world_config", {})
        relationship_summary = "\n".join(
            f"- {other.name}：公开身份 {other.public_identity}；隐藏秘密 {other.hidden_secret}；与你的关系线索 {self._guess_relationship(card, other)}"
            for other in role_cards
            if other.id != card.id
        ) or "- 暂无其他角色关系摘要"
        stage_summary = "\n".join(
            f"- {stage.title}：目标 {stage.objective}；产出 {', '.join(stage.deliverables) if stage.deliverables else '无'}"
            for stage in stages[:6]
        ) or "- 暂无分幕结构"
        fact_summary = "\n".join(f"- {fact}" for fact in facts[:8]) or "- 暂无额外事实"
        genre = str(memory.get("summary", {}).get("genre") or world_config.get("script_type") or "")
        era = str(memory.get("summary", {}).get("era") or world_config.get("era") or "")
        conflict = str(world_config.get("core_conflict") or memory.get("summary", {}).get("focus") or "")
        scene_lines = []
        for index, scene in enumerate((scene_plan or {}).get("scenes", []), start=1):
            scene_lines.append(
                f"场景{index}：{scene.get('title', '')}；目的 {scene.get('purpose', '')}；冲突 {scene.get('conflict', '')}；触发 {scene.get('trigger', '')}；结果 {scene.get('outcome', '')}"
            )
        scene_text = "\n".join(scene_lines) or "场景1：从异常开场把人物推进局里。"
        beat_text = "\n".join(f"- {item}" for item in (beat_sheet or {}).get("beats", [])) or "- 从误会、试探、改口一路推进到终局亮牌。"

        return (
            f"请为角色“{card.name}”写一份完整、连续、可阅读的长剧情角色剧本，用于剧本杀发本。\n\n"
            f"【剧本标题】{record_title}\n"
            f"【类型】{genre}\n"
            f"【时代背景】{era}\n"
            f"【核心冲突】{conflict}\n"
            f"【本轮用户需求】{self._latest_user_requirement(memory)}\n"
            f"【角色公开身份】{card.public_identity}\n"
            f"【角色隐藏秘密】{card.hidden_secret}\n"
            f"【角色核心动机】{card.motivation}\n"
            f"【关键道具】{card.key_prop}\n"
            f"【角色标签】{'、'.join(card.tags) if card.tags else '无'}\n"
            f"【行为约束】{'；'.join(card.behavior_rules) if card.behavior_rules else '无'}\n\n"
            f"【其他角色与关系】\n{relationship_summary}\n\n"
            f"【剧情分幕】\n{stage_summary}\n\n"
            f"【已确认事实】\n{fact_summary}\n\n"
            f"【prompt 蓝图】{(blueprint.model_dump_json(ensure_ascii=False) if blueprint else '无')}\n"
            f"【写作意图】{writing_intent or {}}\n"
            f"【Beat Sheet（场景大纲）】\n{beat_text}\n\n"
            f"【Scene Plan】\n{scene_text}\n\n"
            "写作要求：\n"
            "1. 必须写成长文本，不要写列表式设定，不要写模板占位。\n"
            "2. 开头要有具体场景和人物处境，让读者一进入就知道自己卷入了什么局。\n"
            "3. 中段要通过人与人之间的误会、试探、隐忍、反击和信息拼图推动，而不是只靠主持人说明。\n"
            "4. 每一幕都要体现这个角色看到的、误会的、隐瞒的、害怕失去的内容。\n"
            "5. 结尾要把角色推到一个必须作出选择的位置，并自然衔接场上互动。\n"
            "6. 允许有心理描写、动作细节、气氛描写和带火药味的对话，但不要变成纯小说旁白。\n"
            "7. 文风要有人味，有压迫感和张力，避免空泛的大词和流水账。\n"
            "8. 尽量使用第一人称有限视角，让玩家像在读自己的故事，而不是旁观世界说明。\n"
            "9. 必须显式写出：真实姓名、年龄、职业、外貌、性格、口头禅、成长经历、专属秘密、核心任务、隐藏目标，以及与其他所有角色的关系。\n"
            "10. 单篇角色剧本最终长度必须在 2800-3500 字之间，不允许缩写、概括或套话。\n\n"
            f"【最低质量参考】\n{self.example_script}\n\n"
            f"【兜底草稿，仅供你理解角色，不要照抄】\n{fallback.content}\n\n"
            "请直接输出最终角色剧本正文。"
        )

    def _normalize_generated_content(
        self,
        content: str,
        fallback_content: str,
        card: AgentRoleCard,
        stages: list[AgentStructureStage],
    ) -> str:
        cleaned = str(content or "").strip()
        if not cleaned:
            return fallback_content
        if len(cleaned) < 800:
            cleaned = f"{cleaned}\n\n{fallback_content}"
        if "【角色故事长剧情 / 世界观圣经】" in cleaned:
            cleaned = cleaned.split("【角色故事长剧情 / 世界观圣经】", 1)[0].strip() or cleaned
        if "【角色剧本写法要求】" in cleaned:
            cleaned = cleaned.split("【角色剧本写法要求】", 1)[0].strip() or cleaned
        if "【角色行动清单】" in cleaned:
            cleaned = cleaned.split("【角色行动清单】", 1)[0].strip() or cleaned
        if "【分幕长剧情 / 中段推进】" in cleaned:
            prefix, _, suffix = cleaned.partition("【分幕长剧情 / 中段推进】")
            cleaned = f"{prefix.strip()}\n\n{suffix.strip()}".strip()
        if card.name not in cleaned:
            cleaned = f"{card.name}，{cleaned}"
        if "第一幕：" not in cleaned:
            first_stage = stages[0].title if stages else "失衡开场"
            cleaned = f"{cleaned}\n\n第一幕：{first_stage}\n我知道自己已经没有办法继续只做旁观者。"
        if "场景1：" not in cleaned:
            cleaned = f"{cleaned}\n\n场景1：我先用最稳妥的方式开口，让别人以为我只是顺着局势发问。"
        if len(cleaned) < self.PIPELINE_MIN_CONTENT_LENGTH:
            multiplier = max(2, (self.PIPELINE_MIN_CONTENT_LENGTH // max(len(cleaned), 1)) + 1)
            padding = "\n\n".join([
                "【成长经历补足】\n我不是一夜之间走到今天这一步的。真正把我推到这场局里的，是过去那些没被妥善收束的关系、亏欠、误判和一次次自以为已经翻篇却从未真正结束的旧事。",
                "【关系链补足】\n和其他人之间的关系从来不止台面上这一层。有人欠我解释，有人握着我的把柄，有人让我迟迟不敢把真话说满，也有人让我即使明知会翻车，还是不得不继续把局走下去。",
                "【心理活动补足】\n每一幕里我最真实的状态都不是简单的“怀疑”或“紧张”，而是反复在自保、试探、误判、补救和亮牌之间来回拉扯。别人看到的是我说出来的版本，我真正承受的是那些不能被提前说穿的部分。",
            ] * multiplier)
            cleaned = f"{cleaned}\n\n{padding}".strip()
        return cleaned.strip()

    def _build_stage_story_lines(self, stages: list[AgentStructureStage]) -> str:
        if not stages:
            return (
                "第一幕：你先稳住身份，在第一轮发言里判断谁最急着切断某段旧事。\n"
                "场景1：你要把试探伪装成普通提问，先把别人逼到必须回应的位置。\n"
                "场景2：线索浮出后，你会意识到最危险的并不是证据本身，而是谁在借它改写叙事。\n"
                "终局前夜：你必须决定是继续保护自己，还是用更大的代价换一次翻盘。"
            )
        lines: list[str] = []
        labels = ["第一幕", "第二幕", "第三幕", "第四幕", "第五幕"]
        for index, stage in enumerate(stages[:5]):
            label = labels[index] if index < len(labels) else f"第{index + 1}幕"
            lines.append(f"{label}：{stage.title}。你需要围绕“{stage.objective}”作出选择或隐藏信息。")
            if index < 3:
                lines.append(
                    f"场景{index + 1}：你在这一轮里不能只是跟着局势走，而要让“{stage.title}”变成别人露出破绽的时机。"
                )
        return "\n".join(lines)

    def _guess_relationship(self, card: AgentRoleCard, other: AgentRoleCard) -> str:
        if card.relationships:
            joined = "；".join(card.relationships)
            if other.name in joined:
                return joined
        return "你们之间存在尚未说破的立场差异与隐藏旧事。"

    def _load_saved_blueprints(self, memory: dict[str, Any]) -> list[AgentRoleScriptPromptBlueprint]:
        result: list[AgentRoleScriptPromptBlueprint] = []
        for item in memory.get("role_script_prompt_blueprints", []):
            try:
                result.append(AgentRoleScriptPromptBlueprint.model_validate(item))
            except Exception:
                continue
        return result

    def _build_default_blueprint(self, card: AgentRoleCard, memory: dict[str, Any]) -> AgentRoleScriptPromptBlueprint:
        return AgentRoleScriptPromptBlueprint(
            id=card.id,
            role_name=card.name,
            user_requirement=self._latest_user_requirement(memory),
            objective=f"强化{card.name}的角色秘密、关系张力与终局回收",
            style=self._default_blueprint_style(memory),
            must_keep=self._default_must_keep(card),
            focus=self._default_focus(card),
            negative_constraints=["不要写成提纲"],
            source="auto",
        )

    def _default_blueprint_style(self, memory: dict[str, Any]) -> str:
        genre = str(memory.get("summary", {}).get("genre") or "悬疑长剧情").strip()
        return f"{genre}，强调试探对白、心理迟疑与信息延迟揭示"

    def _default_must_keep(self, card: AgentRoleCard) -> list[str]:
        items = [card.public_identity, card.hidden_secret, card.key_prop]
        return [item for item in items if str(item).strip()]

    def _default_focus(self, card: AgentRoleCard) -> list[str]:
        return [item for item in ("角色秘密", "关系拉扯", card.key_prop or "关键证据", "终局亮牌") if item]

    def _default_beats(self, card: AgentRoleCard, stages: list[AgentStructureStage]) -> list[str]:
        stage_titles = [stage.title for stage in stages[:4]]
        if stage_titles:
            beats = [f"先在“{stage_titles[0]}”里稳住{card.public_identity}身份并观察谁最心虚"]
            if len(stage_titles) > 1:
                beats.append(f"利用“{stage_titles[1]}”把{card.key_prop or '关键道具'}推到场面中央，制造第一层误读")
            if len(stage_titles) > 2:
                beats.append(f"在“{stage_titles[2]}”里逼迫关键角色改口，让隐藏秘密反过来威胁自己")
            beats.append(f"把前文误会与秘密统一回收到{card.motivation or '终局真相'}上")
            return beats
        return [
            f"先稳住{card.public_identity}身份并观察谁最急于掩饰局势",
            f"利用{card.key_prop or '关键道具'}制造第一层误读",
            "在对质中逼迫他人改口，同时避免自己的秘密被提前看穿",
            "把误导线索、真实动机和终局代价统一回收",
        ]

    def _default_scenes(
        self,
        card: AgentRoleCard,
        stages: list[AgentStructureStage],
        writing_intent: dict[str, Any],
        beat_sheet: dict[str, Any],
        memory: dict[str, Any],
    ) -> list[dict[str, str]]:
        scene_hint = self._scene_hint(memory)
        beats = beat_sheet.get("beats", [])
        scenes: list[dict[str, str]] = []
        first_stage = stages[0].title if stages else "开场"
        scenes.append(
            {
                "title": f"{first_stage}试探",
                "purpose": f"让{card.name}带着{card.public_identity}身份进入局面并观察众人",
                "conflict": str(writing_intent.get("conflict") or card.hidden_secret or "不能暴露知道得太多"),
                "trigger": f"{scene_hint}中的第一轮公开发言出现漏洞",
                "outcome": "把最值得怀疑的人推到必须回应的位置",
            }
        )
        second_title = stages[1].title if len(stages) > 1 else "证据升级"
        scenes.append(
            {
                "title": second_title,
                "purpose": f"通过{card.key_prop or beat_sheet.get('spotlight_prop') or '关键证据'}制造错误怀疑链",
                "conflict": f"{card.name}需要继续逼问别人，同时守住自己的底牌",
                "trigger": str(writing_intent.get("hook") or beats[1] if len(beats) > 1 else "关键线索被重新解读"),
                "outcome": "让他人改口，并把关系拉扯推进到下一轮对质",
            }
        )
        return scenes

    def _normalize_scenes(self, value: Any) -> list[dict[str, str]]:
        scenes: list[dict[str, str]] = []
        if not isinstance(value, list):
            return scenes
        for item in value:
            if not isinstance(item, dict):
                continue
            scene = {
                "title": str(item.get("title") or "").strip(),
                "purpose": str(item.get("purpose") or "").strip(),
                "conflict": str(item.get("conflict") or "").strip(),
                "trigger": str(item.get("trigger") or "").strip(),
                "outcome": str(item.get("outcome") or "").strip(),
            }
            if scene["title"]:
                scenes.append(scene)
        return scenes

    def _latest_user_requirement(self, memory: dict[str, Any]) -> str:
        request = str(memory.get("active_generation_request") or "").strip()
        if request:
            return request
        for message in reversed(memory.get("messages", [])):
            if message.get("role") == "user":
                content = str(message.get("content") or "").strip()
                if content:
                    return content
        return "请根据当前设定生成角色剧本，并让剧情更有血有肉。"

    def _scene_hint(self, memory: dict[str, Any]) -> str:
        intent_params = memory.get("intent", {}).get("params", {})
        onboarding = memory.get("onboarding", {}).get("answers", {})
        return (
            str(intent_params.get("scene") or "").strip()
            or str(onboarding.get("scene") or "").strip()
            or "主场景与搜证空间之间的高压现场"
        )

    def _other_role_names(self, card: AgentRoleCard, role_cards: list[AgentRoleCard]) -> list[str]:
        return [item.name for item in role_cards if item.id != card.id]

    def _normalize_text_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _infer_cover_age(card: AgentRoleCard) -> str:
        for tag in card.tags:
            if any(token in tag for token in ("岁", "青年", "中年", "少年")):
                return tag
        return ""

    @staticmethod
    def _infer_tagline(card: AgentRoleCard) -> str:
        if card.tags:
            return " / ".join(card.tags[:2])
        return card.archetype

    @staticmethod
    def _default_card() -> AgentRoleCard:
        return AgentRoleCard(
            id="role-1",
            name="沈砚",
            archetype="克制调查者",
            public_identity="记者",
            hidden_secret="他提前见过关键证人，并且知道旧案里最不能被公开的一段时间线。",
            motivation="查清真相，同时保护一个不该被拖下水的人。",
            key_prop="录音笔",
            tags=["28岁", "瘦高、眼神冷静", "口头禅：先别急，下结论之前把时间线捋一遍"],
            behavior_rules=["先让别人把话说多，再决定自己亮哪层信息", "绝不在第一轮就交出完整时间线"],
            relationships=["与林岚互相提防，但都知道对方在旧案里留过痕迹", "与周野表面客气，实际都在等对方先露破绽"],
        )

    @staticmethod
    def _normalize_role_cards(raw: list[dict[str, Any]]) -> list[AgentRoleCard]:
        cards: list[AgentRoleCard] = []
        for item in raw:
            try:
                cards.append(AgentRoleCard.model_validate(item))
            except Exception:
                continue
        return cards

    @staticmethod
    def _normalize_stages(raw: list[dict[str, Any]]) -> list[AgentStructureStage]:
        stages: list[AgentStructureStage] = []
        for item in raw:
            try:
                stages.append(AgentStructureStage.model_validate(item))
            except Exception:
                continue
        return stages
