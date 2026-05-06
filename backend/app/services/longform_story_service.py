from __future__ import annotations

from typing import Any


class LongFormStoryService:
    LONGFORM_REQUEST_TOKENS = (
        "完整故事",
        "完整剧情",
        "故事剧情",
        "剧情故事",
        "叙事版",
        "写成故事",
        "写成剧情",
        "长文本",
        "长剧情",
        "正文",
        "剧情长稿",
        "长篇",
    )
    STORYLINE_SECTION_TITLE = "## 剧情正文长稿"

    def request_targets_longform_story(self, request: str) -> bool:
        normalized = (request or "").strip().lower()
        return any(token.lower() in normalized for token in self.LONGFORM_REQUEST_TOKENS)

    def build_agent_longform_prompt_context(self, memory: dict[str, Any], *, compact: bool = False) -> str:
        summary = memory.get("summary", {})
        answers = memory.get("onboarding", {}).get("answers", {})
        title = summary.get("title") or "未命名剧本"
        genre = answers.get("genre") or summary.get("genre") or "待确认"
        era = answers.get("era") or summary.get("era") or "待确认"
        players = answers.get("players") or "待确认"
        protagonist = answers.get("protagonist") or "关键角色"
        conflict = answers.get("conflict") or "待确认"
        goal = answers.get("goal") or "待确认"
        latest_request = self._extract_latest_request(memory) or "继续完善当前剧本"
        target_words = 1400 if compact else (2600 if self.request_targets_longform_story(latest_request) else 2000)
        world_bible = self._build_world_bible(title, genre, era, players, protagonist, conflict, goal, summary, memory)
        role_cards = self._build_role_cards(memory.get("role_cards", []))
        continuity = self._build_continuity_facts(memory)
        stage_memory = self._build_stage_memory(memory)
        draft_tail = self._tail_text(str(memory.get("draft_content") or "").strip(), 1200)
        return (
            "长文本生成参考：\n"
            f"目标长度：约 {target_words} 字，可上下浮动，但必须写成可阅读的连续剧情。\n"
            "输出定位：把 draft_content 当成剧本杀的剧情长稿，不是只列提纲。\n"
            "结构要求：至少覆盖开场引爆、关系拉扯、线索升级、终局回收四个推进层次。\n"
            "写作要求：必须出现具体场景动作、人物试探、信息差、线索触发与后果，不能只写抽象总结。\n"
            "风格要求：保留剧本杀的信息拼图感、角色立场冲突和 DM 可落地的互动节点。\n\n"
            f"【世界观圣经】\n{world_bible}\n\n"
            f"【角色卡】\n{role_cards}\n\n"
            f"【阶段摘要】\n{stage_memory}\n\n"
            f"【必须遵守的事实清单】\n{continuity}\n\n"
            f"【当前稿件结尾】\n{draft_tail or '暂无既有长稿，可直接起稿。'}"
        ).strip()

    def build_agent_storyline_section(
        self,
        *,
        title: str,
        conflict: str,
        protagonist: str,
        goal: str,
        role_cards: list[dict[str, Any]],
        memory: dict[str, Any],
    ) -> str:
        supporting_roles = [card.get("name", "") for card in role_cards[1:4] if card.get("name")]
        support_preview = "、".join(supporting_roles) or "其余角色"
        highlighted_prop = next(
            (card.get("key_prop", "") for card in role_cards if str(card.get("key_prop", "")).strip()),
            "关键证据",
        )
        secret_preview = next(
            (card.get("hidden_secret", "") for card in role_cards if str(card.get("hidden_secret", "")).strip()),
            f"与“{conflict}”相关的旧秘密",
        )
        recent_fact = next((item for item in memory.get("facts", []) if str(item).strip()), f"核心冲突：{conflict}")
        scene_hint = self._extract_scene_hint(memory)
        return (
            f"{self.STORYLINE_SECTION_TITLE}\n"
            "### 开场引爆\n"
            f"《{title}》的第一轮冲突不该从说明书开始，而要从现场失衡开始。{protagonist} 被迫在 {scene_hint} 里先面对一层看似体面的公开关系，"
            f"但 {support_preview} 的第一轮表态已经让局面偏向危险的一边。此处要让玩家一上来就感受到：围绕“{conflict}”的矛盾不是突然出现，"
            f"而是早就埋在每个人不肯说透的旧账里，谁先开口，谁就可能先暴露立场。\n\n"
            "### 关系拉扯\n"
            f"中段推进不能只靠主持人口播线索，而要让角色之间的试探自己把剧情拉起来。{protagonist} 在公开场合必须维持表面身份，"
            f"在私下交流里却需要不断判断谁在借题发挥、谁在故意装傻、谁又想利用信息差把怀疑链推到别人身上。{support_preview} 至少各自掌握一块只说一半的事实，"
            f"他们的话术、沉默和转移焦点的方式，本身就是推进剧情的动作，而不是附属说明。\n\n"
            "### 线索升级\n"
            f"当剧情进入搜证和交叉质询阶段后，{highlighted_prop} 不应只是被介绍出来，而要成为触发下一轮站队变化的关键节点。"
            f"玩家拿到线索时，先看到的是能误导判断的表层意义，随后才通过口供矛盾、时间线错位或关系裂缝，意识到真正该追问的是“{secret_preview}”。"
            f"此处必须让每条有效线索都对应一个具体后果，例如迫使角色改口、打断原有联盟，或者把隐藏动机推到台前。\n\n"
            "### 终局回收\n"
            f"后段不是简单宣布答案，而要把前文散开的怀疑、误会和证据重新并到“{goal}”上。终局揭示时，应该先让 {protagonist} 面对自己最相信的一条叙事被推翻，"
            f"再通过 {highlighted_prop} 与此前埋下的关系线，证明真正驱动“{conflict}”的不是单一行为，而是一整套被掩盖的选择链。玩家读到这里时，"
            "既要看懂真相，也要理解每个角色为什么会走到这一步，以及他们最终各自失去了什么。\n\n"
            "### 长文本写法要求\n"
            f"- 保留已确认事实：{recent_fact}\n"
            "- 每一段都要有可视化动作、情绪反应和新的信息变化，不能只做概述。\n"
            "- 把剧情写成可以直接放进剧本正文的连续内容，而不是四条提纲。"
        ).strip()

    def build_session_segment_content(self, world_config: dict[str, Any], memory: dict[str, Any], stage: str) -> str:
        title = world_config.get("title", "未命名剧本")
        conflict = world_config.get("core_conflict", "一场失控的局中局")
        role_name = world_config.get("role_name", "关键角色")
        identity = world_config.get("public_identity", "表面身份")
        secret = world_config.get("core_secret", "尚未揭开的秘密")
        goal = world_config.get("hidden_task") or world_config.get("public_task") or "逼近真相"
        fact_text = "；".join(str(item) for item in memory.get("facts", [])[-4:] if str(item).strip()) or "旧设定仍在铺开"
        choice_text = "；".join(str(item) for item in memory.get("choices", [])[-2:] if str(item).strip()) or "玩家尚未做出关键选择"
        stage_hook = {
            "发本导入": "先用异常现场、DM 话术和第一轮不对劲的关系把玩家拽进局里，让所有人意识到这不是普通聚会。",
            "破冰立人设": "通过公开发言、第一印象和站位试探，把角色表面关系快速搭起来，同时埋下立场裂缝。",
            "第一轮搜证": "让搜证、盘问和线索分发同时发生，先给出能被误读的表层证据，再抬高怀疑链。",
            "私聊交易": "把局面从公开表态推进到私下试探、信息交易与临时结盟，让每个人都开始计算代价。",
            "集中公聊": "把时间线、动机链、口供矛盾和关键证据集中碰撞，迫使所有人公开站队。",
            "终局还原": "回收误导信息、隐藏动机和角色代价，让真相落到可复盘、可宣读的收束上。",
        }.get(stage, "继续推进冲突。")
        return (
            f"【{stage}】《{title}》进入新的剧情长文本阶段。{role_name} 仍以“{identity}”的身份停留在场上，但真正压住所有人的不是表面关系，"
            f"而是围绕“{conflict}”不断发酵的怀疑。此刻已经确认的事实包括：{fact_text}。最近玩家做出的关键选择则把局面进一步推向：{choice_text}。\n\n"
            f"{stage_hook} {role_name} 需要在公开互动里维持体面，在私下判断里追查谁最害怕“{secret}”被说破。剧情推进时，应该让线索释放、角色试探和情绪失控交织发生，"
            "而不是只由主持人单向说明。每一次新的信息出现，都要让玩家重新理解上一轮发言的含义。\n\n"
            f"这一阶段的结尾必须把讨论焦点重新收束到“{goal}”上，并自然抛出下一轮要继续追问的对象、证据或关系裂缝，让后续推进能无缝承接。"
        ).strip()

    def _build_world_bible(
        self,
        title: str,
        genre: str,
        era: str,
        players: str,
        protagonist: str,
        conflict: str,
        goal: str,
        summary: dict[str, Any],
        memory: dict[str, Any],
    ) -> str:
        focus = summary.get("focus") or "继续推进主线"
        scene_hint = self._extract_scene_hint(memory)
        return (
            f"- 标题：{title}\n"
            f"- 类型：{genre}\n"
            f"- 时代/世界观：{era}\n"
            f"- 参与人数：{players}\n"
            f"- 主控视角：{protagonist}\n"
            f"- 核心冲突：{conflict}\n"
            f"- 最终目标：{goal}\n"
            f"- 当前创作重点：{focus}\n"
            f"- 主要互动场景：{scene_hint}"
        ).strip()

    def _build_role_cards(self, role_cards: list[dict[str, Any]]) -> str:
        if not role_cards:
            return "- 暂无角色卡，需根据剧情自动补足关键人物。"
        blocks: list[str] = []
        for index, card in enumerate(role_cards[:8], start=1):
            relationships = "；".join(str(item) for item in card.get("relationships", [])[:3] if str(item).strip()) or "待补充关系链"
            rules = "；".join(str(item) for item in card.get("behavior_rules", [])[:3] if str(item).strip()) or "优先维护人设"
            blocks.append(
                (
                    f"{index}. {card.get('name', '角色')} / 人物定位：{card.get('archetype', '关键关系人')}\n"
                    f"   公开身份：{card.get('public_identity', '待补充')}\n"
                    f"   隐藏秘密：{card.get('hidden_secret', '待补充')}\n"
                    f"   核心动机：{card.get('motivation', '待补充')}\n"
                    f"   关键道具：{card.get('key_prop', '待补充')}\n"
                    f"   关系链：{relationships}\n"
                    f"   行为约束：{rules}"
                )
            )
        return "\n".join(blocks).strip()

    def _build_continuity_facts(self, memory: dict[str, Any]) -> str:
        items: list[str] = []
        for fact in memory.get("facts", [])[:10]:
            text = str(fact).strip()
            if text:
                items.append(f"- {text}")
        for line in memory.get("memory_summary", [])[:6]:
            text = str(line).strip()
            if text:
                items.append(f"- {text}")
        if not items:
            items.append("- 暂无历史事实，需在首轮剧情里建立稳定设定。")
        return "\n".join(items[:14]).strip()

    def _build_stage_memory(self, memory: dict[str, Any]) -> str:
        sections: list[str] = []
        for item in memory.get("structure_plan", [])[:5]:
            title = str(item.get("title", "阶段")).strip()
            objective = str(item.get("objective", "")).strip()
            if title or objective:
                sections.append(f"- {title}：{objective or '目标待补充'}")
        for node in memory.get("memory_nodes", [])[-4:]:
            label = str(node.get("label", "记忆节点")).strip()
            content = self._short_text(str(node.get("content", "")).strip(), 90)
            if label or content:
                sections.append(f"- {label}：{content or '内容待补充'}")
        compressed = str(memory.get("compressed_context", "")).strip()
        if compressed:
            sections.append(f"- 压缩上下文：{self._short_text(compressed, 120)}")
        return "\n".join(sections[:10]).strip() or "- 暂无阶段摘要。"

    def _extract_scene_hint(self, memory: dict[str, Any]) -> str:
        intent_params = memory.get("intent", {}).get("params", {})
        onboarding = memory.get("onboarding", {}).get("answers", {})
        return (
            str(intent_params.get("scene") or "").strip()
            or str(onboarding.get("scene") or "").strip()
            or "封闭主场景与可分头互动的搜证空间"
        )

    def _extract_latest_request(self, memory: dict[str, Any]) -> str:
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
    def _short_text(text: str, limit: int) -> str:
        stripped = (text or "").strip()
        if len(stripped) <= limit:
            return stripped
        return stripped[:limit].rstrip() + "..."

    @staticmethod
    def _tail_text(text: str, limit: int) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return ""
        if len(stripped) <= limit:
            return stripped
        return f"...{stripped[-limit:]}"
