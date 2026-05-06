import re
from typing import Any


class StructureEngine:
    STAGE_TEMPLATES = [
        ("opening", "发本导入", 10),
        ("ice_break", "破冰立人设", 15),
        ("search", "第一轮搜证", 25),
        ("private_chat", "私聊交易", 15),
        ("confrontation", "集中公聊", 20),
        ("ending", "终局还原", 15),
    ]

    def build_flow(
        self,
        *,
        genre: str,
        players: str,
        duration: str,
        scene: str,
        conflict: str,
        goal: str,
        template_key: str = "",
    ) -> list[dict[str, Any]]:
        total_minutes = self._resolve_duration_minutes(duration, players)
        stages: list[dict[str, Any]] = []
        for stage_id, title, ratio in self.STAGE_TEMPLATES:
            stages.append(
                {
                    "id": stage_id,
                    "title": title,
                    "ratio": ratio,
                    "duration_minutes": max(10, round(total_minutes * ratio / 100)),
                    "objective": self._build_objective(stage_id, genre, conflict, goal, template_key),
                    "deliverables": self._build_deliverables(stage_id, genre, scene, template_key),
                }
            )
        return stages

    @staticmethod
    def _resolve_duration_minutes(duration: str, players: str) -> int:
        duration_match = re.search(r"(\d+)", duration)
        if duration_match:
            return int(duration_match.group(1))
        player_count = next((int(item) for item in re.findall(r"\d+", players)), 6)
        return {
            4: 90,
            6: 120,
            8: 180,
            10: 240,
        }.get(player_count, 120)

    @staticmethod
    def _build_objective(stage_id: str, genre: str, conflict: str, goal: str, template_key: str = "") -> str:
        objectives = {
            "opening": f"用发本导入和 DM 开场把所有玩家直接拉进“{conflict}”对应的异常现场。",
            "ice_break": "建立角色公开身份、第一印象和初始站位，让玩家先敢说、敢演、敢互相试探。",
            "search": "通过搜证、盘问和分组行动放出第一轮有效线索，同时埋入可被误读的证据。",
            "private_chat": "让玩家交换信息、试探立场、临时结盟或互相交易，把公开矛盾升级为私人博弈。",
            "confrontation": f"集中回收口供矛盾、关系撕裂和关键证据，迫使玩家围绕“{goal}”公开站队。",
            "ending": "完成真相揭示、动机回收、DM 复盘与情绪收束，形成可宣读的闭环结局。",
        }
        if template_key == "liangyuan-chaos" and stage_id == "private_chat":
            objectives[stage_id] = f"通过配对、小游戏、爆料和社死翻车制造高互动笑点，同时继续推进“{goal}”。"
        return objectives[stage_id]

    @staticmethod
    def _build_deliverables(stage_id: str, genre: str, scene: str, template_key: str = "") -> list[str]:
        base = {
            "opening": [f"场景导入：{scene}", "DM 开场话术", "发本后的第一轮异常提示"],
            "ice_break": ["角色公开身份摘要", "破冰问题", "第一轮站位提示"],
            "search": ["首批线索", "搜证区域与触发方式", "误导线索与真线索比例"],
            "private_chat": ["私聊目标", "交换条件", "临时联盟或背刺节点"],
            "confrontation": ["核心揭示节点", "集体对峙话题", "公聊后的关键抉择"],
            "ending": ["结局宣读", "复盘摘要", "后续导出说明"],
        }
        if template_key == "liangyuan-chaos":
            base["private_chat"] = ["小游戏规则", "配对机制", "公开爆料节点"]
            base["ending"] = ["配对结果", "反转笑点", "DM 收尾话术"]
        return base[stage_id]
