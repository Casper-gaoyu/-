from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


class ConceptImportService:
    MAX_FILE_BYTES = 2 * 1024 * 1024
    SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".json", ".csv", ".yml", ".yaml", ".docx"}

    def extract_text(self, filename: str, content_type: str, raw_bytes: bytes) -> str:
        suffix = Path(filename or "").suffix.lower()
        if len(raw_bytes or b"") > self.MAX_FILE_BYTES:
            raise ValueError("上传文件过大，请控制在 2MB 以内。")
        if suffix and suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError("暂不支持该文件类型，请上传 txt、md、json、csv、yaml 或 docx。")

        if suffix == ".docx":
            text = self._extract_docx_text(raw_bytes)
        else:
            text = self._decode_text_bytes(raw_bytes)

        cleaned = self._clean_text(text)
        if len(cleaned) < 20:
            raise ValueError("文件内容过短，暂时无法识别有效构想。")
        return cleaned

    def analyze_text(self, text: str, filename: str = "") -> dict[str, Any]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = (
            self._match_value(text, [r"(?:剧本名称|剧本标题|标题|名称)\s*[:：]\s*(.+)"])
            or self._extract_first_heading(text)
            or Path(filename or "").stem
            or "导入构想"
        )
        players = self._match_value(text, [r"(\d+\s*人(?:本|局)?)"])
        genre = self._detect_genre(text)
        era = self._detect_era(text)
        conflict = self._match_value(
            text,
            [
                r"(?:核心冲突|主要冲突|故事冲突|冲突)\s*[:：]\s*(.+)",
                r"(?:故事梗概|剧情梗概|故事简介|剧情简介)\s*[:：]\s*(.+)",
            ],
        )
        protagonist = self._match_value(text, [r"(?:主控视角|主角|主人公|视角角色)\s*[:：]\s*(.+)"])
        goal = self._match_value(text, [r"(?:玩家目标|最终目标|游戏目标|目标)\s*[:：]\s*(.+)"])
        role_cards = self._extract_role_cards(text)
        facts = self._extract_facts(lines, conflict, goal)
        summary = self._summarize_import(lines)
        return {
            "title": title[:200],
            "genre": genre,
            "era": era,
            "players": players,
            "conflict": conflict,
            "protagonist": protagonist,
            "goal": goal,
            "role_cards": role_cards,
            "facts": facts,
            "summary": summary,
        }

    def merge_analyses(self, analyses: list[dict[str, Any]], filenames: list[str]) -> dict[str, Any]:
        if not analyses:
            return {
                "title": "",
                "genre": "",
                "era": "",
                "players": "",
                "conflict": "",
                "protagonist": "",
                "goal": "",
                "role_cards": [],
                "facts": [],
                "summary": "",
            }

        def first_value(key: str) -> str:
            for item in analyses:
                value = str(item.get(key) or "").strip()
                if value:
                    return value
            return ""

        role_cards: list[dict[str, Any]] = []
        seen_role_names: set[str] = set()
        for item in analyses:
            for card in item.get("role_cards", []):
                name = str(card.get("name") or "").strip()
                if not name or name in seen_role_names:
                    continue
                role_cards.append(card)
                seen_role_names.add(name)

        facts: list[str] = []
        seen_facts: set[str] = set()
        for item in analyses:
            for fact in item.get("facts", []):
                text = str(fact or "").strip()
                if not text or text in seen_facts:
                    continue
                facts.append(text)
                seen_facts.add(text)

        summaries = []
        for filename, item in zip(filenames, analyses):
            summary = str(item.get("summary") or "").strip()
            if summary:
                summaries.append(f"{filename}：{summary}")

        title = first_value("title")
        if len(filenames) > 1 and title and all(title != Path(name).stem for name in filenames):
            merged_title = title
        else:
            merged_title = title or "合并导入构想"

        return {
            "title": merged_title[:200],
            "genre": first_value("genre"),
            "era": first_value("era"),
            "players": first_value("players"),
            "conflict": first_value("conflict"),
            "protagonist": first_value("protagonist"),
            "goal": first_value("goal"),
            "role_cards": role_cards[:12],
            "facts": facts[:16],
            "summary": " | ".join(summaries)[:500] if summaries else "",
        }

    @staticmethod
    def classify_filename(filename: str) -> str:
        normalized = str(filename or "").replace("\\", "/").lower()
        if any(token in normalized for token in ("角色", "人物", "人设", "关系", "role", "character")):
            return "roles"
        if any(token in normalized for token in ("线索", "证据", "道具", "clue", "props", "物证")):
            return "clues"
        if any(token in normalized for token in ("世界观", "设定", "背景", "大纲", "剧情", "梗概", "world", "outline")):
            return "world"
        return "other"

    @staticmethod
    def _decode_text_bytes(raw_bytes: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("latin-1", errors="ignore")

    def _extract_docx_text(self, raw_bytes: bytes) -> str:
        try:
            with zipfile.ZipFile(BytesIO(raw_bytes)) as archive:
                xml_bytes = archive.read("word/document.xml")
        except Exception as exc:
            raise ValueError("无法读取 docx 文件内容。") from exc

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise ValueError("docx 内容解析失败。") from exc

        paragraphs: list[str] = []
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        for paragraph in root.iter(f"{namespace}p"):
            texts = [node.text or "" for node in paragraph.iter(f"{namespace}t")]
            line = "".join(texts).strip()
            if line:
                paragraphs.append(line)
        return "\n".join(paragraphs)

    @staticmethod
    def _clean_text(text: str) -> str:
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def _extract_first_heading(text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
            if stripped.startswith("《") and "》" in stripped:
                return stripped.split("》", 1)[0].strip("《》 ")
            return stripped[:80]
        return ""

    @staticmethod
    def _match_value(text: str, patterns: list[str]) -> str:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return str(match.group(1) or "").strip("：: \n\t。；;")
        return ""

    @staticmethod
    def _detect_genre(text: str) -> str:
        mapping = [
            ("恐怖", "恐怖"),
            ("惊悚", "恐怖"),
            ("情感", "情感"),
            ("欢乐", "欢乐"),
            ("搞笑", "欢乐"),
            ("硬核", "硬核"),
            ("推理", "现代悬疑"),
            ("悬疑", "现代悬疑"),
        ]
        for keyword, value in mapping:
            if keyword in text:
                return value
        return ""

    @staticmethod
    def _detect_era(text: str) -> str:
        mapping = [
            ("现代校园", "现代校园"),
            ("现代都市", "现代都市"),
            ("都市", "现代都市"),
            ("民国", "民国小镇"),
            ("古风", "古风武侠"),
            ("武侠", "古风武侠"),
            ("未来", "未来科幻"),
            ("科幻", "未来科幻"),
        ]
        for keyword, value in mapping:
            if keyword in text:
                return value
        return ""

    def _extract_role_cards(self, text: str) -> list[dict[str, Any]]:
        blocks = re.split(r"\n\s*\n", text)
        cards: list[dict[str, Any]] = []
        for index, block in enumerate(blocks):
            if not any(token in block for token in ("角色", "姓名", "公开身份", "隐藏秘密", "动机", "道具")):
                continue
            name = self._match_value(block, [r"(?:角色名|角色|姓名)\s*[:：]\s*([^\n]+)"])
            if not name:
                heading_match = re.match(r"^#{1,4}\s*([^\n]+)$", block.strip().splitlines()[0])
                if heading_match and any(token in block for token in ("身份", "秘密", "动机")):
                    name = heading_match.group(1).strip()
            if not name:
                continue
            cards.append(
                {
                    "id": f"import-role-{index + 1}",
                    "name": name[:40],
                    "archetype": self._match_value(block, [r"(?:人物定位|角色定位| archetype )\s*[:：]\s*([^\n]+)"]) or "关键关系人",
                    "public_identity": self._match_value(block, [r"(?:公开身份|身份)\s*[:：]\s*([^\n]+)"]) or "公开身份待补充",
                    "hidden_secret": self._match_value(block, [r"(?:隐藏秘密|秘密)\s*[:：]\s*([^\n]+)"]) or "隐藏秘密待补充",
                    "motivation": self._match_value(block, [r"(?:核心动机|动机)\s*[:：]\s*([^\n]+)"]) or "动机待补充",
                    "key_prop": self._match_value(block, [r"(?:关键道具|关键物件|道具)\s*[:：]\s*([^\n]+)"]) or "关键道具待补充",
                    "tags": [],
                    "behavior_rules": [],
                    "relationships": [],
                }
            )
            if len(cards) >= 8:
                break
        return cards

    @staticmethod
    def _extract_facts(lines: list[str], conflict: str, goal: str) -> list[str]:
        facts: list[str] = []
        for line in lines:
            cleaned = line.lstrip("-*•0123456789. ").strip()
            if len(cleaned) < 6:
                continue
            if any(token in cleaned for token in ("标题", "名称", "类型", "时代", "玩家目标")):
                continue
            facts.append(cleaned[:120])
            if len(facts) >= 8:
                break
        if conflict:
            facts.insert(0, f"导入构想核心冲突：{conflict}")
        if goal:
            facts.insert(1 if facts else 0, f"导入构想目标：{goal}")
        unique: list[str] = []
        seen: set[str] = set()
        for item in facts:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique[:8]

    @staticmethod
    def _summarize_import(lines: list[str]) -> str:
        if not lines:
            return ""
        joined = " ".join(lines[:6]).strip()
        joined = re.sub(r"\s+", " ", joined)
        return joined[:220]
