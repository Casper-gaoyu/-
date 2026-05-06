from __future__ import annotations

from typing import Any


class TemplateRuntimeService:
    @staticmethod
    def build_runtime_flags(template_config: dict[str, Any]) -> dict[str, Any]:
        key = str(template_config.get("template_key") or "general").strip()
        family = str(template_config.get("template_family") or "general").strip()
        return {
            "template_key": key,
            "template_label": str(template_config.get("template_label") or "通用剧本模板").strip(),
            "template_family": family,
            "is_liangyuan_template": key == "liangyuan-chaos",
            "is_comedy_dating_template": family == "comedy-dating",
        }
