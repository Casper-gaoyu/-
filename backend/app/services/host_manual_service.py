from __future__ import annotations

import re
from typing import Any

from app.models.schemas import (
    AgentDmCue,
    AgentHostManualAct,
    AgentHostManualAppendixItem,
    AgentHostManualFixedEvent,
    AgentHostManualFrontMatter,
    AgentHostManualSchema,
    AgentHostManualSceneCard,
    AgentHostManualVisualAttachment,
    AgentRoleCard,
    AgentStructureStage,
)


class HostManualService:
    def build(self, record_title: str, memory: dict[str, Any]) -> AgentHostManualSchema:
        summary = memory.get("summary", {})
        template_key = str(summary.get("template_key") or memory.get("template_key") or "general").strip()
        role_cards = self._normalize_role_cards(memory.get("role_cards", []))
        stages = self._normalize_stages(memory.get("structure_plan", []))
        dm_cues = self._normalize_cues(memory.get("dm_script", []))
        facts = [str(item).strip() for item in memory.get("facts", []) if str(item).strip()]
        behaviors = [item for item in memory.get("behavior_records", []) if isinstance(item, dict)]
        draft_content = str(memory.get("draft_content") or "")
        title = str(summary.get("title") or record_title or "未命名剧本").strip() or "未命名剧本"
        era = str(summary.get("era") or "待确认").strip() or "待确认"
        genre = str(summary.get("genre") or "待确认").strip() or "待确认"
        genre_key = self._genre_key(genre)
        focus = str(summary.get("focus") or "待补充").strip() or "待补充"
        players = self._read_fact_value(facts, "参与人数：") or f"{max(len(role_cards), 1)} 人"

        acts = [self._build_act(index, stage, dm_cues, role_cards, facts) for index, stage in enumerate(stages)]
        fixed_events = self._build_fixed_events(stages, dm_cues)

        return AgentHostManualSchema(
            front_matter=AgentHostManualFrontMatter(
                title=title,
                era=era,
                genre=genre,
                recommended_players=players,
                current_focus=focus,
                synopsis=self._extract_synopsis(draft_content),
            ),
            material_checklist=self._build_material_checklist(role_cards, acts, behaviors, genre_key, template_key),
            opening_preparation=self._build_opening_preparation(dm_cues, acts, genre_key),
            player_count_variants=self._build_player_count_variants(len(role_cards), genre_key),
            fixed_events=fixed_events,
            acts=acts,
            truth_reveal=self._build_truth_reveal(facts, role_cards, behaviors),
            ending_adjudication=self._build_ending_adjudication(genre, behaviors, genre_key, template_key),
            review_points=self._build_review_points(role_cards, behaviors, genre_key),
            answer_key=self._build_answer_key(facts, acts),
            npc_reference=self._build_npc_reference(dm_cues, role_cards, genre_key),
            visual_attachments=self._build_visual_attachments(genre_key, template_key),
            appendix=self._build_appendix(role_cards, acts, behaviors, genre_key, template_key),
        )

    @staticmethod
    def _genre_key(genre: str) -> str:
        if "恐怖" in genre or "惊悚" in genre:
            return "horror"
        if "情感" in genre or "沉浸" in genre:
            return "emotion"
        if "欢乐" in genre:
            return "comedy"
        if "古风" in genre or "武侠" in genre or "权谋" in genre:
            return "power"
        if "未来" in genre or "科幻" in genre:
            return "scifi"
        return "suspense"

    def _build_act(
        self,
        index: int,
        stage: AgentStructureStage,
        dm_cues: list[AgentDmCue],
        role_cards: list[AgentRoleCard],
        facts: list[str],
    ) -> AgentHostManualAct:
        cue = next((item for item in dm_cues if item.stage_id == stage.id), None)
        scenes = self._build_scenes(index, stage, cue, role_cards, facts)
        return AgentHostManualAct(
            id=stage.id,
            title=stage.title,
            duration_minutes=stage.duration_minutes,
            objective=stage.objective,
            deliverables=stage.deliverables,
            scenes=scenes,
        )

    def _build_scenes(
        self,
        act_index: int,
        stage: AgentStructureStage,
        cue: AgentDmCue | None,
        role_cards: list[AgentRoleCard],
        facts: list[str],
    ) -> list[AgentHostManualSceneCard]:
        templates = self._resolve_scene_templates(stage.title, stage.objective)
        base_duration = max(4, round(stage.duration_minutes / max(len(templates), 1)))
        role_names = [card.name for card in role_cards[:3]]
        role_preview = "、".join(role_names) if role_names else "关键角色"
        materials = self._guess_materials(stage, role_cards, facts)
        scene_cards: list[AgentHostManualSceneCard] = []
        for scene_index, template in enumerate(templates):
            duration = base_duration if scene_index < len(templates) - 1 else max(4, stage.duration_minutes - base_duration * (len(templates) - 1))
            scene_cards.append(
                AgentHostManualSceneCard(
                    code=f"{act_index + 1}.{scene_index + 1}",
                    title=template["title"],
                    duration_minutes=duration,
                    objective=f"{template['focus']}，并服务于“{stage.objective}”。",
                    materials=materials,
                    player_count_variants=self._build_scene_player_count_variants(len(role_cards), stage.title, template["title"]),
                    dm_read_aloud=self._build_scene_read_aloud(scene_index, template["focus"], stage, cue),
                    dm_actions=[
                        f"明确说明当前环节聚焦：{template['focus']}。",
                        f"引导 {role_preview} 至少一人先给出可验证信息，再推动其他玩家跟进。",
                        "若场面卡住超过 2-3 分钟，立即补充公共提示、线索或点名提问。",
                    ],
                    trigger_conditions=[
                        f"玩家已经进入“{stage.title}”并准备围绕“{stage.objective}”展开行动。",
                        "现场出现沉默、跑题或重复对话时，可提前启动本环节。",
                    ],
                    trigger_checks=[
                        "检查主持人手边的线索、口播和记录表是否已就绪。",
                        "检查上一环节结论是否已经被玩家接受，并能顺畅衔接当前动作。",
                    ],
                    branch_handling=[
                        "玩家讨论过于发散时，直接重述当前幕目标，只保留 1-2 个主问题。",
                        "玩家过快猜中真相时，不立即确认，要求其用线索、口供或行为链自证。",
                    ],
                    resolution_rules=[
                        f"至少形成 1 个与“{stage.objective}”直接相关的有效输出，才能视为本环节完成。",
                        "若只有情绪表达没有新信息，判定为未完成，需要主持人补提示。",
                    ],
                    record_points=[
                        "记录谁主动发起关键提问、公开质疑或明确站队。",
                        "记录任何会影响结局裁定、复盘解释或支线回收的玩家行为。",
                    ],
                    clue_release=[
                        f"优先围绕“{stage.objective}”投放公共信息或人物证据。",
                        "若当前没有专属线索，可改为主持口播、NPC 消息或上一轮信息回收。",
                    ],
                    wrap_up=f"确认“{'、'.join(stage.deliverables) if stage.deliverables else stage.objective}”已经基本落地，再进入下一环。",
                )
            )
        return scene_cards

    @staticmethod
    def _resolve_scene_templates(stage_title: str, objective: str) -> list[dict[str, str]]:
        normalized = stage_title.lower()
        if "开" in stage_title or "破冰" in stage_title or "读本" in stage_title:
            return [
                {"title": "开场说明", "focus": "主持人建立规则、语境和初始氛围"},
                {"title": "角色入场与信息抛出", "focus": "让玩家用人设、立场和第一轮信息彼此定位"},
                {"title": "第一轮冲突点火", "focus": "把表面关系推进成可讨论的真实矛盾"},
            ]
        if "信息" in stage_title or "搜证" in stage_title or "发展" in normalized:
            return [
                {"title": "信息投放", "focus": "通过公共线索或主持提示抛出新信息"},
                {"title": "自由讨论 / 私聊", "focus": "让玩家围绕新信息站队、互问和交换"},
                {"title": "阶段收束", "focus": "确认这一幕该落下的有效信息已经成型"},
            ]
        if "高潮" in stage_title or "对质" in stage_title or "任务" in stage_title:
            return [
                {"title": "关键矛盾升级", "focus": "把之前埋下的冲突推到公开层面"},
                {"title": "核心选择 / 公开表态", "focus": "迫使玩家做出明确行为或立场选择"},
                {"title": "结果宣布与反应", "focus": "公布当前结果，并把后果带入下一幕"},
            ]
        if "结局" in stage_title or "终局" in stage_title:
            return [
                {"title": "终局揭示", "focus": "主持人统一回收真相、动机和关键证据"},
                {"title": "结局裁定", "focus": "根据玩家行为和选择宣布最终走向"},
                {"title": "复盘导入", "focus": "自然进入复盘、补讲和情绪收束"},
            ]
        return [
            {"title": "主持导入", "focus": f"围绕“{objective}”搭建当前环节入口"},
            {"title": "核心互动", "focus": "让玩家围绕本幕唯一目标展开行动"},
            {"title": "收束推进", "focus": "确认本幕结果并衔接到下一幕"},
        ]

    @staticmethod
    def _build_scene_player_count_variants(role_count: int, stage_title: str, scene_title: str) -> list[str]:
        if role_count <= 5:
            return [
                "少人场（<=5 人）：压缩自由讨论时长，优先保证每位玩家都至少一次公开表达。",
                f"少人场（<=5 人）：{scene_title} 阶段如信息不足，主持人可额外补 1 条公共提示。",
            ]
        if role_count >= 8:
            return [
                "多人场（>=8 人）：先按左右半场或阵营分流讨论，再统一公开汇总。",
                f"多人场（>=8 人）：{stage_title} 阶段建议增加倒计时和点名轮，避免流程失控。",
            ]
        return [
            "标准场（6-7 人）：允许开放讨论，但主持人要随时把问题收回当前幕目标。",
            "标准场（6-7 人）：若出现多条分支，优先保留最能推动下一幕的那一条。",
        ]

    @staticmethod
    def _build_scene_read_aloud(scene_index: int, focus: str, stage: AgentStructureStage, cue: AgentDmCue | None) -> str:
        if scene_index == 0:
            return cue.opening_line if cue else f"现在进入“{stage.title}”，请各位把注意力集中到“{stage.objective}”。"
        if cue and scene_index == 2:
            return cue.transition_line
        return f"这一轮请重点围绕“{focus}”行动，所有讨论都要服务于“{stage.objective}”。"

    def _build_fixed_events(self, stages: list[AgentStructureStage], dm_cues: list[AgentDmCue]) -> list[AgentHostManualFixedEvent]:
        events: list[AgentHostManualFixedEvent] = []
        for stage_index, stage in enumerate(stages):
            cue = next((item for item in dm_cues if item.stage_id == stage.id), None)
            for scene_index, scene in enumerate(self._resolve_scene_templates(stage.title, stage.objective)):
                events.append(
                    AgentHostManualFixedEvent(
                        code=f"E-{stage_index + 1}-{scene_index + 1}",
                        stage_title=stage.title,
                        scene_title=scene["title"],
                        trigger_timing=f"第 {stage_index + 1} 幕进行中，约第 {scene_index + 1} 个环节",
                        trigger_condition=f"当玩家开始围绕“{stage.objective}”行动，或现场卡住超过 2-3 分钟时触发。",
                        host_action=(cue.opening_line if cue and scene_index == 0 else cue.transition_line if cue and scene_index == 2 else f"主持人推进“{scene['focus']}”"),
                        resulting_state=f"推动“{'、'.join(stage.deliverables) if stage.deliverables else stage.objective}”落地，并把结果带入下一环。",
                    )
                )
        return events

    @staticmethod
    def _build_material_checklist(
        role_cards: list[AgentRoleCard],
        acts: list[AgentHostManualAct],
        behaviors: list[dict[str, Any]],
        genre_key: str,
        template_key: str,
    ) -> list[str]:
        scene_count = sum(len(act.scenes) for act in acts)
        lines = [
            f"玩家手册：{len(role_cards)} 份",
            f"主持版手册：1 份，含 {len(acts)} 幕 / {scene_count} 个主持环节卡",
            "真相页：1 份",
            "复盘页：1 份",
            f"行为记录：当前已录入 {len(behaviors)} 条",
        ]
        if genre_key == "horror":
            lines.extend(["氛围物料：环境音 / 灯光控制 / 异常提示卡", "安全提示：惊吓边界、暂停词、情绪缓冲说明"])
        elif genre_key == "emotion":
            lines.extend(["关系物料：角色关系卡 / 旧物卡 / 情绪触发卡", "情绪收束：复盘引导卡 / 告别提示页"])
        elif template_key == "liangyuan-chaos":
            lines.extend(["相亲互动物料：互选卡、爆料卡、社死检讨书", "轻量结局物料：最佳良缘 CP 奖状 / 翻车之王提示卡"])
        return lines

    @staticmethod
    def _build_opening_preparation(dm_cues: list[AgentDmCue], acts: list[AgentHostManualAct], genre_key: str) -> list[str]:
        opening = dm_cues[0].opening_line if dm_cues else "欢迎各位进入本场剧本。正式开始前请先确认手册、规则与阅读边界。"
        lines = [
            "完整阅读主持手册、真相页和复盘页，确认每幕目标与收束条件。",
            "按角色整理玩家手册，避免错发；按阶段预分线索、口播与补充信息。",
            opening,
            f"若本场允许人数浮动，先确认实际人数，再对照 {len(acts)} 幕流程做差分调整。",
        ]
        if genre_key == "horror":
            lines.extend(["开场前确认灯光、环境音和安全边界说明，避免惊吓直接越线。", "主持人口播要先建立异常感，再逐步投放可验证线索。"])
        elif genre_key == "emotion":
            lines.extend(["开场前确认角色关系导入顺序，避免一开始就把情绪爆点讲满。", "主持人口播要先让玩家相信彼此关系，再逐步抛出旧事和误解。"])
        return lines

    @staticmethod
    def _build_player_count_variants(role_count: int, genre_key: str) -> list[str]:
        lines = [
            "4-5 人场：压缩自由讨论时长，优先保证每位玩家至少一次公开表达。",
            "6-7 人场：这是默认推荐区间，分幕、线索发放和私聊节奏都按标准流程执行。",
            "8 人及以上：先按半场、阵营或信息层级切分讨论，再统一公开汇总。",
            f"当前角色数：{role_count}。减员时优先砍重复功能位，增员时优先补公共信息位。",
        ]
        if genre_key == "horror":
            lines.append("恐怖场人数越多，越要控制同步讨论，优先营造压迫感而不是同时抛太多信息。")
        elif genre_key == "emotion":
            lines.append("情感场人数越少，越要拉长关系交流时间；人数越多，越要避免情绪线互相打断。")
        return lines

    def _build_truth_reveal(self, facts: list[str], role_cards: list[AgentRoleCard], behaviors: list[dict[str, Any]]) -> list[str]:
        role_line = "；".join(f"{card.name}：{card.hidden_secret or '待补充'}" for card in role_cards[:4]) or "待补充"
        behavior_line = "；".join(
            f"{item.get('role_name', '角色')} / {item.get('action_type', '行为')} / {item.get('key_output', '待补充')}"
            for item in behaviors[:4]
        ) or "当前暂无记录"
        return [
            f"核心设定：{'；'.join(facts[:4]) or '待补充'}",
            f"角色秘密链：{role_line}",
            f"玩家已触发的有效行为：{behavior_line}",
        ]

    @staticmethod
    def _build_ending_adjudication(genre: str, behaviors: list[dict[str, Any]], genre_key: str, template_key: str = "general") -> list[str]:
        if genre_key == "horror":
            rules = [
                "先确认异常来源、关键真相和玩家是否真正理解恐惧代价。",
                "再确认关键幸存者、牺牲者或异常承受者是否完成收束。",
                "不要只宣布答案，要让恐惧链条和后果一起落地。",
            ]
        elif template_key == "liangyuan-chaos":
            rules = [
                "先确认是否出现双向互选、明确站队或公开情绪爆点。",
                "再确认关键社死梗、隐藏秘密或幕后反转是否已被公开触发。",
            ]
        elif "情感" in genre:
            rules = [
                "先确认公开情绪爆点、明确站队或关键告别是否已经落地。",
                "再确认关键关系、隐藏秘密和反转真相是否已经被公开回收。",
            ]
        else:
            rules = [
                "先确认核心真相是否已被玩家用线索、口供和行为链闭环证明。",
                "再确认关键角色的公开身份、隐藏秘密和核心动机是否都完成回收。",
            ]
        rules.append(f"当前行为记录数：{len(behaviors)}。若记录不足，建议主持人先补记关键行为后再裁定。")
        return rules

    def _build_review_points(self, role_cards: list[AgentRoleCard], behaviors: list[dict[str, Any]], genre_key: str) -> list[str]:
        lines = [
            f"先核对 {len(behaviors)} 条玩家主动行为，区分‘已验证触发’和‘仅口头表态’的内容。",
            f"逐个检查 {len(role_cards)} 个角色的公开身份、隐藏秘密与核心动机是否都在流程中被回收。",
            "若存在未触发支线，建议补写‘如果当时这样选，会发生什么’的替代走向。",
        ]
        if genre_key == "horror":
            lines.append("额外检查恐惧氛围是否通过场景、线索和玩家反应逐步升级，而不是只靠主持人口播。")
        elif genre_key == "emotion":
            lines.append("额外检查情绪爆点是否建立在关系链和旧事回收上，而不是只靠临时煽情。")
        return lines

    def _build_answer_key(self, facts: list[str], acts: list[AgentHostManualAct]) -> list[str]:
        lines = [f"基础答案：{fact}" for fact in facts[:6]]
        lines.extend(
            f"{act.title} 应至少回收：{'、'.join(act.deliverables) if act.deliverables else act.objective}"
            for act in acts[:6]
        )
        return lines or ["当前还没有足够的事实与流程，建议先补齐真相页。"]

    def _build_npc_reference(self, dm_cues: list[AgentDmCue], role_cards: list[AgentRoleCard], genre_key: str) -> list[str]:
        lines = [
            "主持人：负责规则宣读、线索投放、阶段收束、结局宣读。",
            "公共信息发布者：用于在场面卡住时补口播、放新线索或推新问题。",
            "结果宣读位：用于投票、裁定、终局公布和复盘导入。",
        ]
        lines.extend(
            f"{cue.stage_title} / 开场口播：{cue.opening_line} / 转场口播：{cue.transition_line}"
            for cue in dm_cues[:6]
        )
        if role_cards:
            lines.append(f"当前玩家角色数：{len(role_cards)}。若需要 NPC 出场，优先承担公告、施压、补证据、宣读结果四类功能。")
        if genre_key == "horror":
            lines.append("恐怖本的 NPC / 主持代演更适合承担异常播报、环境变化、倒计时压迫和后果宣读。")
        elif genre_key == "emotion":
            lines.append("情感本的 NPC / 主持代演更适合承担旧事见证者、关系说明者和情绪收束引导者。")
        return lines

    @staticmethod
    def _build_visual_attachments(genre_key: str, template_key: str = "general") -> list[AgentHostManualVisualAttachment]:
        base = [
            AgentHostManualVisualAttachment(id="flow_board", title="流程总览图", description="展示主持流程、分幕推进和转场顺序。"),
            AgentHostManualVisualAttachment(id="clue_board", title="线索发放看板", description="按阶段和持有方展示线索分发结构。"),
            AgentHostManualVisualAttachment(id="host_summary", title="主持人速查面板", description="展示章节数、玩家手册数、线索卡数和复盘入口。"),
        ]
        if genre_key == "horror":
            base.append(AgentHostManualVisualAttachment(id="anomaly_map", title="异常场景示意图", description="展示异常发生区域、危险点和氛围触发顺序。"))
        elif genre_key == "emotion":
            base.append(AgentHostManualVisualAttachment(id="relationship_map", title="关系链示意图", description="展示关键关系、旧事牵连和情绪爆点回收顺序。"))
        elif template_key == "liangyuan-chaos":
            base.append(AgentHostManualVisualAttachment(id="dating_flow", title="相亲互动流程卡", description="展示破冰、互选、爆料和翻车回收顺序。"))
        return base

    def _build_appendix(
        self,
        role_cards: list[AgentRoleCard],
        acts: list[AgentHostManualAct],
        behaviors: list[dict[str, Any]],
        genre_key: str,
        template_key: str,
    ) -> list[AgentHostManualAppendixItem]:
        base = [
            AgentHostManualAppendixItem(id="roles", title="角色速查", items=[card.name for card in role_cards] or ["当前无角色数据"]),
            AgentHostManualAppendixItem(id="acts", title="分幕索引", items=[act.title for act in acts] or ["当前无流程数据"]),
            AgentHostManualAppendixItem(id="behaviors", title="行为记录摘要", items=[item.get("key_output", "待补充") for item in behaviors[:8]] or ["当前无行为记录"]),
        ]
        if genre_key == "horror":
            base.append(AgentHostManualAppendixItem(id="safety", title="安全边界", items=["惊吓边界说明", "暂停词", "缓冲流程"]))
        elif genre_key == "emotion":
            base.append(AgentHostManualAppendixItem(id="emotion", title="情绪回收", items=["关系回收顺序", "旧事揭示顺序", "复盘缓冲话术"]))
        elif template_key == "liangyuan-chaos":
            base.append(AgentHostManualAppendixItem(id="dating", title="相亲互动补充", items=["互选规则", "爆料节奏", "翻车后果轻量裁定"]))
        return base

    @staticmethod
    def _normalize_role_cards(raw_cards: list[dict[str, Any]]) -> list[AgentRoleCard]:
        cards: list[AgentRoleCard] = []
        for item in raw_cards:
            try:
                cards.append(AgentRoleCard.model_validate(item))
            except Exception:
                continue
        return cards

    @staticmethod
    def _normalize_stages(raw_stages: list[dict[str, Any]]) -> list[AgentStructureStage]:
        stages: list[AgentStructureStage] = []
        for item in raw_stages:
            try:
                stages.append(AgentStructureStage.model_validate(item))
            except Exception:
                continue
        return stages

    @staticmethod
    def _normalize_cues(raw_cues: list[dict[str, Any]]) -> list[AgentDmCue]:
        cues: list[AgentDmCue] = []
        for item in raw_cues:
            try:
                cues.append(AgentDmCue.model_validate(item))
            except Exception:
                continue
        return cues

    @staticmethod
    def _read_fact_value(facts: list[str], prefix: str) -> str:
        for fact in facts:
            if fact.startswith(prefix):
                return fact[len(prefix):].strip()
        return ""

    @staticmethod
    def _extract_synopsis(draft_content: str) -> list[str]:
        lines = [line.strip() for line in str(draft_content or "").splitlines() if line.strip()]
        cleaned = [line for line in lines if not re.match(r"^[【\[]", line)]
        return cleaned[:4] or ["当前草案还没有足够的故事摘要。"]

    @staticmethod
    def _guess_materials(stage: AgentStructureStage, role_cards: list[AgentRoleCard], facts: list[str]) -> list[str]:
        materials = ["主持人话术卡"]
        if role_cards:
            materials.append("玩家手册")
        if facts:
            materials.append("公共线索 / 事实清单")
        if any("线索" in item for item in facts):
            materials.append("线索卡")
        if stage.id in {"confrontation", "ending"}:
            materials.append("记录板 / 投票结果")
        return list(dict.fromkeys(materials))
