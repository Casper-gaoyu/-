import copy
import json
from typing import Any

import redis
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import settings
from app.models.db_models import ScriptSession
from app.services.rule_engine import RuleEngine
from app.services.llm_service import LLMService


class MemoryService:
    def __init__(self) -> None:
        self.redis_client: redis.Redis | None = None
        self.llm_service = LLMService(RuleEngine())
        if settings.redis_url:
            try:
                client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                client.ping()
                self.redis_client = client
            except redis.RedisError:
                self.redis_client = None

    def build_initial_memory(self, world_config: dict[str, Any]) -> dict[str, Any]:
        resolved_era = world_config.get("custom_era") if world_config.get("era") == "custom" else world_config.get("era")
        facts = [
            f"世界观:{resolved_era}",
            f"核心冲突:{world_config.get('core_conflict', '')}",
            f"角色身份:{world_config.get('role_name', '')}/{world_config.get('public_identity', '')}",
            f"核心秘密:{world_config.get('core_secret', '')}",
        ]
        return {
            "world_config": world_config,
            "history": [],
            "choices": [],
            "facts": [fact for fact in facts if fact and not fact.endswith(":")],
            "last_branches": [],
            "llm_profile": self.llm_service.build_llm_profile(),
        }

    def load(self, record: ScriptSession) -> dict[str, Any]:
        if self.redis_client:
            try:
                cached = self.redis_client.get(self._key(record.id))
                if cached:
                    return json.loads(cached)
            except (redis.RedisError, json.JSONDecodeError):
                pass
        return record.memory or self.build_initial_memory(record.world_config)

    def save(self, record: ScriptSession, memory: dict[str, Any]) -> None:
        record.memory = copy.deepcopy(memory)
        flag_modified(record, "memory")
        if not self.redis_client:
            return
        try:
            self.redis_client.set(self._key(record.id), json.dumps(record.memory, ensure_ascii=False), ex=3600 * 24)
        except redis.RedisError:
            return

    def delete(self, session_id: str) -> None:
        if not self.redis_client:
            return
        try:
            self.redis_client.delete(self._key(session_id))
        except redis.RedisError:
            return

    @staticmethod
    def _key(session_id: str) -> str:
        return f"script:memory:{session_id}"

    def prepare_generation_memory(self, memory: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        generation_memory = copy.deepcopy(memory)
        estimated_tokens = self.estimate_tokens(generation_memory)
        compression_applied = False

        if estimated_tokens > settings.token_budget and len(generation_memory.get("history", [])) > 2:
            history = generation_memory.get("history", [])
            preserved_history = history[-2:]
            compressed_history = self._build_history_summary(history[:-2])
            generation_memory["history"] = [{"stage": "记忆摘要", "content": compressed_history}, *preserved_history]
            generation_memory["compressed_history"] = compressed_history
            compression_applied = True
            estimated_tokens = self.estimate_tokens(generation_memory)

        stats = {
            "token_budget": settings.token_budget,
            "estimated_tokens": estimated_tokens,
            "compression_applied": compression_applied,
            "history_entries": len(memory.get("history", [])),
        }
        memory["last_memory_stats"] = stats
        if compression_applied:
            memory["compressed_history"] = generation_memory.get("compressed_history", "")
        return generation_memory, stats

    def prepare_agent_memory(self, memory: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        generation_memory = copy.deepcopy(memory)
        estimated_tokens = self.estimate_agent_tokens(generation_memory)
        compression_applied = False
        draft_snapshot_applied = False

        messages = generation_memory.get("messages", [])
        memory_nodes = generation_memory.get("memory_nodes", [])
        near_budget = estimated_tokens > int(settings.token_budget * 0.75)
        if (estimated_tokens > settings.token_budget or near_budget) and (len(messages) > 6 or len(memory_nodes) > 4):
            preserved_messages = messages[-6:]
            preserved_nodes = memory_nodes[-4:]
            compressed_context = self._build_agent_context_summary(messages[:-6], memory_nodes[:-4])
            generation_memory["messages"] = preserved_messages
            generation_memory["memory_nodes"] = preserved_nodes
            generation_memory["compressed_context"] = compressed_context
            compression_applied = True
            estimated_tokens = self.estimate_agent_tokens(generation_memory)

        if estimated_tokens > settings.token_budget and generation_memory.get("draft_content", "").strip():
            generation_memory["draft_content"] = self._build_draft_snapshot(generation_memory.get("draft_content", ""))
            draft_snapshot_applied = True
            compression_applied = True
            estimated_tokens = self.estimate_agent_tokens(generation_memory)

        stats = {
            "token_budget": settings.token_budget,
            "estimated_tokens": estimated_tokens,
            "compression_applied": compression_applied,
            "draft_snapshot_applied": draft_snapshot_applied,
            "message_entries": len(memory.get("messages", [])),
            "memory_nodes": len(memory.get("memory_nodes", [])),
        }
        memory["memory_stats"] = stats
        if compression_applied:
            memory["compressed_context"] = generation_memory.get("compressed_context", "")
        return generation_memory, stats

    @staticmethod
    def estimate_tokens(memory: dict[str, Any]) -> int:
        text_fragments: list[str] = []
        for entry in memory.get("history", []):
            text_fragments.append(entry.get("content", ""))
        text_fragments.extend(memory.get("choices", []))
        text_fragments.extend(memory.get("facts", []))
        if memory.get("compressed_history"):
            text_fragments.append(memory["compressed_history"])
        joined = "\n".join(fragment for fragment in text_fragments if fragment)
        return max(1, len(joined) // 2)

    @staticmethod
    def estimate_agent_tokens(memory: dict[str, Any]) -> int:
        text_fragments: list[str] = []
        for message in memory.get("messages", []):
            text_fragments.append(message.get("content", ""))
        text_fragments.append(memory.get("draft_content", ""))
        text_fragments.extend(memory.get("facts", []))
        text_fragments.extend(memory.get("memory_summary", []))
        text_fragments.extend(memory.get("generation_rules", []))
        for record in memory.get("behavior_records", []):
            text_fragments.append(json.dumps(record, ensure_ascii=False))
        for node in memory.get("memory_nodes", []):
            text_fragments.append(node.get("content", ""))
            text_fragments.extend(node.get("tags", []))
        for key in ("summary", "role_cards", "intent", "task_queue", "structure_plan", "dm_script", "output_outline"):
            value = memory.get(key)
            if value:
                text_fragments.append(json.dumps(value, ensure_ascii=False))
        if memory.get("compressed_context"):
            text_fragments.append(memory["compressed_context"])
        joined = "\n".join(fragment for fragment in text_fragments if fragment)
        return max(1, len(joined) // 2)

    @staticmethod
    def build_agent_memory_summary(memory: dict[str, Any]) -> list[str]:
        summary: list[str] = []
        summary.extend(f"事实：{fact}" for fact in memory.get("facts", [])[-4:])
        summary.extend(
            (
                f"行为：{item.get('timestamp', '—')} / "
                f"{item.get('role_name', '未命名角色')} / "
                f"{item.get('key_output', '')[:50]}"
            )
            for item in memory.get("behavior_records", [])[-3:]
        )
        summary.extend(
            f"节点：{node.get('label', '记忆节点')} -> {node.get('content', '')[:60]}"
            for node in memory.get("memory_nodes", [])[-4:]
        )
        if memory.get("compressed_context"):
            summary.append(f"压缩上下文：{memory['compressed_context']}")
        if not summary:
            summary.append("当前尚未积累有效记忆节点。")
        return summary

    @staticmethod
    def _build_history_summary(history: list[dict[str, Any]]) -> str:
        if not history:
            return "暂无需要压缩的历史剧情。"

        summary_lines = []
        for entry in history[-4:]:
            stage = entry.get("stage", "阶段")
            content = entry.get("content", "")
            trimmed = content[:80] + ("..." if len(content) > 80 else "")
            summary_lines.append(f"{stage}:{trimmed}")
        return " | ".join(summary_lines)

    @staticmethod
    def _build_draft_snapshot(draft_content: str) -> str:
        cleaned = (draft_content or "").strip()
        if not cleaned:
            return ""
        if len(cleaned) <= 2400:
            return cleaned

        sections: list[str] = []
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if line.startswith("【") and line.endswith("】") and line not in sections:
                sections.append(line)
            if len(sections) == 8:
                break

        head = cleaned[:1200].rstrip()
        tail = cleaned[-900:].lstrip()
        section_summary = " / ".join(sections) if sections else "未提取到分节标题"
        return (
            "以下是当前长稿的压缩快照，请在保留既有设定的前提下继续修改，并返回完整稿件。\n"
            f"章节索引：{section_summary}\n"
            "开头片段：\n"
            f"{head}\n\n"
            "[中间内容已压缩，仍需保持和现有角色、线索、结构一致]\n\n"
            "结尾片段：\n"
            f"{tail}"
        )

    @staticmethod
    def _build_agent_context_summary(messages: list[dict[str, Any]], memory_nodes: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for message in messages[-4:]:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            trimmed = content[:60] + ("..." if len(content) > 60 else "")
            parts.append(f"{role}:{trimmed}")
        for node in memory_nodes[-4:]:
            label = node.get("label", "记忆节点")
            content = node.get("content", "")
            trimmed = content[:60] + ("..." if len(content) > 60 else "")
            parts.append(f"{label}:{trimmed}")
        return " | ".join(parts) if parts else "暂无需要压缩的对话与记忆节点。"
