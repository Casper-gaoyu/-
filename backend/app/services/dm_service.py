from typing import Any


class DMService:
    def build_cues(
        self,
        *,
        stages: list[dict[str, Any]],
        genre: str,
        conflict: str,
        goal: str,
        role_cards: list[dict[str, Any]],
        template_key: str = "",
    ) -> list[dict[str, str]]:
        names = [card.get("name", "角色") for card in role_cards]
        focal_name = names[0] if names else "关键角色"
        cues: list[dict[str, str]] = []
        for stage in stages:
            title = stage["title"]
            stage_id = stage["id"]
            cues.append(
                {
                    "stage_id": stage_id,
                    "stage_title": title,
                    "opening_line": self._opening_line(stage_id, title, genre, conflict, template_key),
                    "transition_line": self._transition_line(stage_id, title, goal, focal_name, template_key),
                    "emergency_line": self._emergency_line(stage_id, genre, focal_name, template_key),
                    "dm_tip": self._dm_tip(stage_id, stage["duration_minutes"], genre, template_key),
                }
            )
        return cues

    @staticmethod
    def _opening_line(stage_id: str, stage_title: str, genre: str, conflict: str, template_key: str = "") -> str:
        if template_key == "liangyuan-chaos" and stage_id == "ice_break":
            return f"欢迎进入‘{stage_title}’环节，今晚所有笑点和翻车都将围绕‘{conflict}’展开。"
        if stage_id == "ending":
            return "现在进入最后收束阶段，请各位根据自己掌握的信息给出最终态度。"
        return f"现在进入‘{stage_title}’环节，请所有玩家围绕‘{conflict}’继续推进。"

    @staticmethod
    def _transition_line(stage_id: str, stage_title: str, goal: str, focal_name: str, template_key: str = "") -> str:
        if stage_id == "mission":
            return f"从这一刻开始，你们的互动会直接影响‘{goal}’的走向，请留意 {focal_name} 的反应。"
        if stage_id == "climax":
            return "前面埋下的信息将在这一轮集中回收，请推动玩家给出明确立场与选择。"
        return f"请把上一环节暴露出来的信息带入‘{stage_title}’，不要让关键线索中断。"

    @staticmethod
    def _emergency_line(stage_id: str, genre: str, focal_name: str, template_key: str = "") -> str:
        if template_key == "liangyuan-chaos":
            return f"如果现场冷掉，就点名请 {focal_name} 先回应，再用爆料或反差问题把气氛重新拉起来。"
        if stage_id == "ending":
            return "如果玩家迟迟无法收束，就请他们各自用一句话总结自己最相信的真相。"
        return "如果玩家讨论发散，就把问题重新拉回‘谁受益、谁隐瞒、谁掌握关键物证’。"

    @staticmethod
    def _dm_tip(stage_id: str, duration_minutes: int, genre: str, template_key: str = "") -> str:
        if stage_id == "ice_break":
            return f"建议控制在 {duration_minutes} 分钟内，不要过早透出核心秘密。"
        if stage_id == "mission" and template_key == "liangyuan-chaos":
            return f"这是互动密度最高的环节，建议在 {duration_minutes} 分钟内至少推进两次公开反馈。"
        if stage_id == "climax":
            return f"这是压缩信息和情绪的关键阶段，建议在 {duration_minutes} 分钟内完成摊牌。"
        return f"建议在 {duration_minutes} 分钟内完成该环节，并确保玩家目标始终清晰。"
