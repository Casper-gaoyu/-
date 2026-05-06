import re
from typing import Any


class IntentService:
    HELP_QUESTION_TOKENS = (
        "如何",
        "怎么",
        "怎样",
        "哪里",
        "在哪",
        "什么是",
        "怎么用",
        "如何用",
        "能否",
        "是否",
    )
    HELP_TARGET_TOKENS = (
        "导出",
        "复盘",
        "玩家行为记录",
        "行为记录",
        "会话",
        "工作台",
        "交付包",
        "玩家手册",
        "线索卡",
        "主持",
        "dm 手册",
        "DM 手册",
    )
    ANALYSIS_QUESTION_TOKENS = (
        "为什么",
        "为何",
        "完成度",
        "进度",
        "还缺什么",
        "还差什么",
        "目前",
        "现在",
        "逻辑",
        "合理",
        "原因",
        "区别",
        "重点",
        "下一步",
    )
    ANALYSIS_TARGET_TOKENS = (
        "剧本",
        "剧情",
        "设定",
        "角色",
        "人物",
        "线索",
        "结局",
        "冲突",
        "秘密",
        "dm",
        "DM",
        "手册",
        "交付",
        "节奏",
        "结构",
        "流程",
    )
    EXPORT_TOKENS = ("导出", "预览", "pdf", "word", "复盘", "打印")
    REWRITE_TOKENS = ("重写", "重新生成", "推翻重来", "换一版", "换一个版本")
    EXTEND_TOKENS = ("补充", "完善", "细化", "增强", "扩写", "补足", "深化")
    REVISE_TOKENS = ("修改", "调整", "替换", "删改", "重排", "改成")
    RENAME_TOKENS = ("改名", "改标题", "改叫", "改为", "换成", "命名为", "重命名")
    ACTION_PREFIX_TOKENS = ("请", "请直接", "帮我", "直接", "继续", "下一步", "把", "将", "给我", "先把")

    ROUTE_META = {
        "workspace_help": {"label": "说明问答", "response_mode": "help"},
        "analysis_question": {"label": "分析当前草案", "response_mode": "answer"},
        "export_review": {"label": "导出/预览/复盘", "response_mode": "generate"},
        "rewrite": {"label": "重新生成", "response_mode": "generate"},
        "extend_content": {"label": "补充内容", "response_mode": "generate"},
        "revise_content": {"label": "修改角色/剧情/台词", "response_mode": "generate"},
        "new_generation": {"label": "生成新剧本", "response_mode": "generate"},
    }

    def analyze(
        self,
        *,
        message: str,
        user_messages: list[str],
        onboarding_answers: dict[str, str],
        previous_summary: dict[str, Any],
        has_draft: bool,
        has_role_cards: bool,
        onboarding_completed: bool,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        combined_text = "\n".join(
            [*user_messages, *[f"{key}:{value}" for key, value in onboarding_answers.items() if value]]
        )
        route = self.classify_route(
            message=message,
            user_messages=user_messages,
            onboarding_answers=onboarding_answers,
            previous_summary=previous_summary,
            has_draft=has_draft,
            has_role_cards=has_role_cards,
            onboarding_completed=onboarding_completed,
        )
        people = onboarding_answers.get("players") or self._detect_players(combined_text)
        params = {
            "type": onboarding_answers.get("genre") or previous_summary.get("genre") or self._detect_genre(combined_text),
            "people": people,
            "time": self._detect_duration(combined_text, people),
            "style": onboarding_answers.get("genre") or previous_summary.get("genre") or self._detect_genre(combined_text),
            "theme": onboarding_answers.get("conflict") or self._detect_theme(combined_text),
            "scene": self._detect_scene(combined_text),
        }
        next_action = self._resolve_next_action(
            route_key=route["key"],
            onboarding_completed=onboarding_completed,
            has_role_cards=has_role_cards,
            has_draft=has_draft,
        )
        progress_label = self._build_progress_label(
            route_key=route["key"],
            onboarding_completed=onboarding_completed,
            has_draft=has_draft,
            has_role_cards=has_role_cards,
        )
        intent_state = {
            "current_intent": route["label"],
            "next_action": next_action,
            "progress_label": progress_label,
            "params": params,
            "route_key": route["key"],
            "response_mode": route["response_mode"],
            "confidence": route["confidence"],
        }
        task_queue = self._build_task_queue(
            route_key=route["key"],
            current_intent=route["label"],
            params=params,
            onboarding_completed=onboarding_completed,
            has_draft=has_draft,
            has_role_cards=has_role_cards,
        )
        return intent_state, task_queue

    def classify_route(
        self,
        *,
        message: str,
        user_messages: list[str],
        onboarding_answers: dict[str, str],
        previous_summary: dict[str, Any],
        has_draft: bool,
        has_role_cards: bool,
        onboarding_completed: bool,
    ) -> dict[str, Any]:
        normalized = (message or "").strip()
        combined_text = "\n".join(
            [normalized, *user_messages[-3:], *[str(value) for value in onboarding_answers.values() if value]]
        )
        if not normalized:
            return self._build_route("new_generation", 0.45)

        if self._looks_like_workspace_help(normalized):
            return self._build_route("workspace_help", 0.96)

        if self._contains_any(normalized, self.EXPORT_TOKENS):
            return self._build_route("export_review", 0.86)

        if onboarding_completed and has_draft and self._looks_like_action_request(normalized):
            if self._contains_any(normalized, self.REWRITE_TOKENS):
                return self._build_route("rewrite", 0.9)
            if self._contains_any(normalized, (*self.REVISE_TOKENS, *self.RENAME_TOKENS)):
                return self._build_route("revise_content", 0.88)
            if self._contains_any(normalized, self.EXTEND_TOKENS):
                return self._build_route("extend_content", 0.86)
            return self._build_route("revise_content", 0.76)

        if onboarding_completed and self._looks_like_analysis_question(normalized, has_draft):
            return self._build_route("analysis_question", 0.9)

        if self._contains_any(normalized, self.REWRITE_TOKENS):
            return self._build_route("rewrite", 0.88)

        if self._contains_any(normalized, (*self.REVISE_TOKENS, *self.RENAME_TOKENS)):
            return self._build_route("revise_content", 0.82)

        if self._contains_any(normalized, self.EXTEND_TOKENS):
            return self._build_route("extend_content", 0.8)

        if not onboarding_completed:
            return self._build_route("new_generation", 0.72)

        if not has_draft or not previous_summary.get("title"):
            return self._build_route("new_generation", 0.74)

        if has_role_cards and self._contains_any(combined_text, self.ANALYSIS_TARGET_TOKENS):
            return self._build_route("extend_content", 0.6)

        return self._build_route("new_generation", 0.55)

    def _build_task_queue(
        self,
        *,
        route_key: str,
        current_intent: str,
        params: dict[str, str],
        onboarding_completed: bool,
        has_draft: bool,
        has_role_cards: bool,
    ) -> list[dict[str, str]]:
        people = params.get("people", "待确认")
        theme = params.get("theme", "待确认")
        scene = params.get("scene", "待确认")

        init_status = "done" if onboarding_completed else "active"
        memory_status = "done" if onboarding_completed else "active"
        outline_status = "done" if has_draft else ("active" if onboarding_completed else "pending")
        role_status = "done" if has_role_cards else ("active" if onboarding_completed else "pending")
        branch_status = "done" if has_draft else ("active" if onboarding_completed else "pending")
        dm_status = "done" if has_draft else ("active" if onboarding_completed else "pending")
        review_status = "done" if has_draft else "pending"

        if route_key == "analysis_question":
            memory_status = "active"
        elif route_key == "export_review":
            review_status = "active"
        elif route_key == "rewrite":
            outline_status = "active"
            branch_status = "active"
        elif route_key == "extend_content":
            role_status = "active" if has_draft else role_status
            branch_status = "active" if has_draft else branch_status
        elif route_key == "revise_content":
            role_status = "active" if has_role_cards else role_status
            outline_status = "active" if has_draft else outline_status

        return [
            {
                "id": "task_init",
                "module": "阶段 1 / 初始化与参数采集",
                "title": "完成参数校验、剧本定位与唯一会话初始化",
                "detail": f"当前意图为“{current_intent}”，推荐人数 {people}，主题聚焦“{theme}”。",
                "status": init_status,
            },
            {
                "id": "task_memory",
                "module": "阶段 2 / 记忆模块",
                "title": "建立记忆节点、生成规则与上下文窗口",
                "detail": "把用户参数、关键事实、流程规则写入记忆库，并准备后续检索。",
                "status": memory_status,
            },
            {
                "id": "task_outline",
                "module": "阶段 3 / 剧情大纲",
                "title": "生成剧情大纲、流程骨架与时间分配",
                "detail": f"围绕场景“{scene}”拆出分幕结构，避免一次性生成超长正文。",
                "status": outline_status,
            },
            {
                "id": "task_roles",
                "module": "阶段 3 / 角色设定",
                "title": "补齐角色卡、秘密、动机与关系图谱",
                "detail": "确保角色数量、行为约束与主控视角保持一致。",
                "status": role_status,
            },
            {
                "id": "task_branch",
                "module": "阶段 4 / 分支规则",
                "title": "生成核心分支、次级分支与触发条件",
                "detail": "控制“核心分支 + 次级分支”的层级结构，便于后续玩家选择记录。",
                "status": branch_status,
            },
            {
                "id": "task_dm",
                "module": "阶段 5 / DM 手册",
                "title": "生成开场、转场、应急与控场话术",
                "detail": "为每个阶段准备 DM 口播、节奏提示和异常兜底方案。",
                "status": dm_status,
            },
            {
                "id": "task_review",
                "module": "阶段 6 / 复盘与输出",
                "title": "整理玩家行为记录、复盘模板与导出结构",
                "detail": "为故事复盘、未触发分支说明和导出交付准备标准化结构。",
                "status": review_status,
            },
        ]

    @staticmethod
    def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
        return any(token in text for token in tokens)

    def _looks_like_workspace_help(self, text: str) -> bool:
        return self._contains_any(text, self.HELP_QUESTION_TOKENS) and self._contains_any(text, self.HELP_TARGET_TOKENS)

    def _looks_like_analysis_question(self, text: str, has_draft: bool) -> bool:
        if not has_draft:
            return False
        if self._looks_like_action_request(text):
            return False
        has_question_shape = text.endswith("?") or text.endswith("？") or self._contains_any(text, self.ANALYSIS_QUESTION_TOKENS)
        has_analysis_target = self._contains_any(text, self.ANALYSIS_TARGET_TOKENS)
        return has_question_shape and has_analysis_target

    def _looks_like_action_request(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if self._looks_like_workspace_help(stripped):
            return False

        has_action_verb = self._contains_any(
            stripped,
            (*self.REWRITE_TOKENS, *self.EXTEND_TOKENS, *self.REVISE_TOKENS, *self.RENAME_TOKENS),
        )
        has_action_prefix = any(stripped.startswith(token) for token in self.ACTION_PREFIX_TOKENS)
        if not has_action_verb and not has_action_prefix:
            return False

        if stripped.startswith(("为什么", "为何")) and not has_action_verb:
            return False
        return True

    def _build_route(self, route_key: str, confidence: float) -> dict[str, Any]:
        meta = self.ROUTE_META[route_key]
        return {
            "key": route_key,
            "label": meta["label"],
            "response_mode": meta["response_mode"],
            "confidence": confidence,
        }

    @staticmethod
    def _build_progress_label(
        *,
        route_key: str,
        onboarding_completed: bool,
        has_draft: bool,
        has_role_cards: bool,
    ) -> str:
        if not onboarding_completed:
            return "正在采集核心参数，并执行合法性校验与记忆初始化。"
        if route_key == "workspace_help":
            return "当前进入说明问答模式，不改动草案，只回答工作台使用方法。"
        if route_key == "analysis_question":
            return "当前进入分析问答模式，优先解释完成度、逻辑和下一步，而不是直接改稿。"
        if route_key == "export_review":
            return "当前进入复盘与导出整理阶段。"
        if not has_draft:
            return "核心设定已完成，正在生成第一版完整草案。"
        if not has_role_cards:
            return "当前已有草案，正在补齐角色卡和关系链。"
        return "已完成参数采集，进入记忆驱动的迭代创作阶段。"

    @staticmethod
    def _resolve_next_action(
        *,
        route_key: str,
        onboarding_completed: bool,
        has_role_cards: bool,
        has_draft: bool,
    ) -> str:
        if not onboarding_completed:
            return "继续补全参数，并完成当前步骤的合法性校验。"
        if route_key == "workspace_help":
            return "直接回答用户的使用问题，并提供下一步可点击的操作建议。"
        if route_key == "analysis_question":
            return "先解释当前进度、逻辑和差距，再决定是否需要进入改稿。"
        if route_key == "export_review":
            return "整理复盘结构、玩家行为记录和导出内容。"
        if route_key == "rewrite":
            return "先重新规划当前结构与重点章节，再重写正文。"
        if not has_role_cards:
            return "先补齐角色卡与关系链，再进入正文重写。"
        if not has_draft:
            return "先产出分阶段大纲与分支树，再补全成稿。"
        return "基于记忆节点继续细化分支、DM 手册或复盘模板。"

    @staticmethod
    def _detect_genre(text: str) -> str:
        for genre in ("欢乐", "情感", "推理", "机制", "恐怖", "现代悬疑", "悬疑", "硬核"):
            if genre in text:
                return "现代悬疑" if genre == "悬疑" else genre
        return "现代悬疑"

    @staticmethod
    def _detect_players(text: str) -> str:
        match = re.search(r"(\d+)\s*人", text)
        if match:
            return f"{match.group(1)} 人"
        return "6 人"

    @staticmethod
    def _detect_duration(text: str, players: str) -> str:
        hour_match = re.search(r"(\d+(?:\.\d+)?)\s*小时", text)
        if hour_match:
            minutes = int(float(hour_match.group(1)) * 60)
            return f"{minutes} 分钟"
        minute_match = re.search(r"(\d+)\s*分钟", text)
        if minute_match:
            return f"{minute_match.group(1)} 分钟"
        people_count = next((int(item) for item in re.findall(r"\d+", players)), 6)
        defaults = {
            4: 90,
            5: 100,
            6: 120,
            7: 150,
            8: 180,
            9: 210,
            10: 240,
        }
        return f"{defaults.get(people_count, 120)} 分钟"

    @staticmethod
    def _detect_theme(text: str) -> str:
        if "相亲" in text:
            return "相亲翻车与社交反差"
        if "校园" in text:
            return "校园旧案与关系裂痕"
        if "家族" in text:
            return "家族利益与信任崩塌"
        if "密室" in text:
            return "密室事件与真相还原"
        return "关系冲突与剧情推进"

    @staticmethod
    def _detect_scene(text: str) -> str:
        scene_rules = [
            ("相亲", "封闭联谊酒会厅"),
            ("校园", "可搜证的教学楼与旧社团空间"),
            ("民国", "民国旧宅与戏班后台"),
            ("未来", "未来空间站中枢舱"),
            ("科幻", "未来空间站中枢舱"),
            ("古风", "山门大院与密室回廊"),
            ("武侠", "山门大院与密室回廊"),
        ]
        for token, scene in scene_rules:
            if token in text:
                return scene
        return "封闭主场景 + 多点互动空间"
