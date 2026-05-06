import re
from typing import Any


ERA_FORBIDDEN = {
    "ancient_wuxia": ["手机", "监控", "直播", "微信", "电梯"],
    "modern_suspense": ["飞剑", "掌门", "渡劫", "仙门", "灵根"],
    "future_sci-fi": ["飞鸽传书", "掌门", "轻功", "门派", "江湖侠客"],
}

ERA_ALIAS = {
    "古风武侠": "ancient_wuxia",
    "现代悬疑": "modern_suspense",
    "现代都市": "modern_suspense",
    "现代校园": "modern_suspense",
    "民国小镇": "modern_suspense",
    "未来科幻": "future_sci-fi",
}


class RuleEngine:
    def build_constraints(self, memory: dict[str, Any]) -> list[str]:
        constraints = [f"必须遵守事实：{fact}" for fact in memory.get("facts", [])[-8:]]
        constraints.extend(f"玩家选择已生效：{choice}" for choice in memory.get("choices", [])[-4:])
        constraints.extend(f"生成规则：{rule}" for rule in memory.get("generation_rules", [])[-6:])
        return constraints

    def derive_facts_from_choice(self, choice_text: str) -> list[str]:
        facts: list[str] = []
        protect_match = re.search(r"保护([\u4e00-\u9fa5A-Za-z0-9]+)", choice_text)
        kill_match = re.search(r"(?:杀死|处决)([\u4e00-\u9fa5A-Za-z0-9]+)", choice_text)
        investigate_match = re.search(r"(?:调查|搜查|检查)([\u4e00-\u9fa5A-Za-z0-9]+)", choice_text)
        trust_match = re.search(r"(?:信任|支持)([\u4e00-\u9fa5A-Za-z0-9]+)", choice_text)
        betray_match = re.search(r"(?:背叛|揭穿)([\u4e00-\u9fa5A-Za-z0-9]+)", choice_text)

        if protect_match:
            facts.append(f"保护目标:{protect_match.group(1)}")
        if kill_match:
            facts.append(f"已死亡:{kill_match.group(1)}")
        if investigate_match:
            facts.append(f"已调查:{investigate_match.group(1)}")
        if trust_match:
            facts.append(f"已信任:{trust_match.group(1)}")
        if betray_match:
            facts.append(f"已背叛:{betray_match.group(1)}")
        return facts

    def check_logic_consistency(self, memory: dict[str, Any], new_script: str) -> list[str]:
        violations: list[str] = []
        world_config = memory.get("world_config", {})
        era = self._normalize_era_key(world_config.get("era") or world_config.get("custom_era") or "")
        for forbidden in ERA_FORBIDDEN.get(era, []):
            if forbidden in new_script:
                violations.append(f"时代设定冲突：出现了“{forbidden}”")

        for fact in memory.get("facts", []):
            if fact.startswith("保护目标:"):
                target = fact.split(":", 1)[1]
                if re.search(rf"{re.escape(target)}.*?(死亡|受伤|倒下)", new_script):
                    violations.append(f"与保护目标冲突：{target}")
            if fact.startswith("已死亡:"):
                target = fact.split(":", 1)[1]
                if re.search(rf"{re.escape(target)}.*?(再次出现|走进房间|重新开口)", new_script):
                    violations.append(f"死亡事实被推翻：{target}")

        return violations

    def build_logic_checks(self, memory: dict[str, Any], new_script: str) -> list[str]:
        violations = self.check_logic_consistency(memory, new_script)
        if violations:
            return [f"未通过：{item}" for item in violations]

        checks = ["通过：世界观设定未冲突", "通过：已知事实未被推翻"]
        if memory.get("choices"):
            checks.append("通过：玩家上一轮选择已经进入当前剧情上下文")
        else:
            checks.append("通过：首段剧情已按初始设定展开")
        return checks

    def build_agent_generation_rules(
        self,
        answers: dict[str, str],
        role_cards: list[dict[str, Any]],
        template_key: str = "",
    ) -> list[str]:
        genre = answers.get("genre", "")
        players = answers.get("players", "")
        conflict = answers.get("conflict", "")
        goal = answers.get("goal", "")
        protagonist = answers.get("protagonist", "")

        rules = [
            "采用分阶段生成：先世界观与大纲，再角色与分支，最后补全 DM 手册与复盘。",
            "每次生成都必须先检索记忆节点，再引用已确认事实，不得推翻既有设定。",
        ]

        if genre == "恐怖":
            rules.append("氛围约束：优先使用压迫、未知和渐进揭示，不要过早摊牌。")
        elif template_key == "liangyuan-chaos" or (genre == "欢乐" and any(token in conflict or token in goal for token in ("相亲", "配对", "翻车"))):
            rules.extend(
                [
                    "风格约束：保持高互动、强反差与可控翻车感，不要偏成纯推理。",
                    "信息拼图约束：每个角色只能持有局部真相，禁止单角色直接掌握完整幕后身份与终局答案。",
                    "角色手册约束：优先补齐封面、流程目标、公开身份、隐藏秘密、任务、第一轮发言提示、新手模板与翻车后果。",
                    "本土化约束：优先使用相亲局、社死、装人设、挂朋友圈、互相拆台、体面圆场这类中文社交语境。",
                ]
            )
        elif genre in {"硬核", "现代悬疑"}:
            rules.append("推理约束：线索必须闭环，时间线与证据链要能互相解释。")

        if players or protagonist or goal:
            combined_constraints = []
            if players:
                combined_constraints.append(f"人数={players}")
            if protagonist:
                combined_constraints.append(f"主控视角={protagonist}")
            if goal:
                combined_constraints.append(f"终局目标={goal}")
            rules.append(f"基础规格约束：{'，'.join(combined_constraints)}。")
        if conflict:
            rules.append(f"冲突闭环约束：所有分支最终都要回收到“{conflict}”。")
        if role_cards:
            rules.append("人物一致性约束：角色行为、秘密和关系链必须与当前角色卡保持一致。")
        deduped_rules = list(dict.fromkeys(rules))
        return deduped_rules[:8]

    def build_agent_logic_checks(
        self,
        *,
        memory: dict[str, Any],
        draft_content: str,
        role_cards: list[dict[str, Any]],
        structure_plan: list[dict[str, Any]],
        summary: dict[str, Any],
        template_key: str = "",
    ) -> list[str]:
        checks: list[str] = []
        answers = memory.get("onboarding", {}).get("answers", {})
        players = answers.get("players", "")
        conflict = answers.get("conflict", "")
        protagonist = answers.get("protagonist", "")
        era = summary.get("era") or answers.get("era", "")
        genre = summary.get("genre") or answers.get("genre", "")

        violations = self._check_agent_era_consistency(era, draft_content)
        if violations:
            checks.extend(f"提示：{item}" for item in violations)
        else:
            checks.append("通过：草案内容与当前时代背景未发现明显冲突")

        expected_roles = self._resolve_expected_role_count(players)
        if role_cards and len(role_cards) == expected_roles:
            checks.append(f"通过：角色卡数量与人数设定一致（{expected_roles} 张）")
        elif role_cards:
            checks.append(f"提示：当前角色卡 {len(role_cards)} 张，建议与人数设定 {expected_roles} 张对齐")
        else:
            checks.append("提示：当前还没有角色卡，后续应补齐人物设定与关系链")

        if structure_plan and len(structure_plan) >= 5:
            checks.append("通过：流程骨架已拆分为多阶段，可支持分段生成与 DM 控场")
        else:
            checks.append("提示：流程骨架阶段数偏少，建议至少包含大纲、分支、高潮与复盘阶段")

        if conflict and conflict in draft_content:
            checks.append("通过：核心冲突已经写入当前草案并作为主线驱动")
        elif conflict:
            checks.append("提示：核心冲突尚未在正文中明确展开，建议补齐主线矛盾")

        if protagonist and any(protagonist in (card.get("public_identity", "") + card.get("name", "")) for card in role_cards):
            checks.append(f"通过：主控视角“{protagonist}”已经映射到角色卡")
        elif protagonist:
            checks.append(f"提示：主控视角“{protagonist}”尚未明确绑定到角色卡，可继续补齐")

        if memory.get("memory_nodes"):
            checks.append("通过：记忆节点库已启用，可追踪参数、生成、分支和复盘线索")
        else:
            checks.append("提示：当前未记录记忆节点，后续生成会缺少检索依据")

        if template_key == "liangyuan-chaos":
            if any(token in draft_content for token in ("新手发言模板", "欢乐本本土化梗库", "相亲翻车玩法设计")):
                checks.append("通过：欢乐本专项结构已经覆盖玩法、梗库或新手话术")
            else:
                checks.append("提示：欢乐 / 相亲翻车题材建议补充玩法设计、本土化梗库或新手发言模板")

        return checks

    @staticmethod
    def _normalize_era_key(era: str) -> str:
        return ERA_ALIAS.get(era, era)

    def _check_agent_era_consistency(self, era: str, draft_content: str) -> list[str]:
        normalized = self._normalize_era_key(era)
        violations: list[str] = []
        for forbidden in ERA_FORBIDDEN.get(normalized, []):
            if forbidden in draft_content:
                violations.append(f"时代背景“{era}”中不应出现“{forbidden}”")
        return violations

    @staticmethod
    def _resolve_expected_role_count(players: str) -> int:
        numbers = [int(item) for item in re.findall(r"\d+", players)]
        if not numbers:
            return 6
        return max(2, min(numbers[0], 10))
