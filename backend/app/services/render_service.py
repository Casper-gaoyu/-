class RenderService:
    def build_output_outline(self, role_count: int) -> list[dict[str, str]]:
        return [
            {
                "id": "overview",
                "title": "剧本信息",
                "description": "汇总类型、人数、时长、主题、场景与整体风格。",
                "format_hint": "封面 + 基础设定",
            },
            {
                "id": "roles",
                "title": "角色卡",
                "description": f"输出 {role_count} 张标准化角色卡，包含身份、秘密、动机、约束与关系。",
                "format_hint": "分角色卡片 + 关系说明",
            },
            {
                "id": "flow",
                "title": "流程分幕",
                "description": "按破冰、信息获取、互动、高潮、结局分阶段组织正文。",
                "format_hint": "流程树 + 时间分配",
            },
            {
                "id": "content",
                "title": "任务与线索",
                "description": "整理任务目标、线索来源、触发条件、核心分支与次级分支后果。",
                "format_hint": "任务清单 + 线索清单 + 分支树",
            },
            {
                "id": "choices",
                "title": "玩家行为记录",
                "description": "记录玩家关键选择、触发节点与对应的剧情影响，供后续检索与复盘。",
                "format_hint": "时间线 + 选择影响表",
            },
            {
                "id": "dm",
                "title": "DM 手册",
                "description": "输出开场、转场、应急圆场、结局宣读与控场提醒。",
                "format_hint": "可直接朗读脚本",
            },
            {
                "id": "ending",
                "title": "结局与复盘",
                "description": "提供多结局说明、未选分支结果、关键剧情节点回收与复盘模板。",
                "format_hint": "结局树 + 复盘报告",
            },
            {
                "id": "export",
                "title": "导出格式",
                "description": "支持 Word、PDF、纯文本与分角色导出。",
                "format_hint": "统一渲染输出",
            },
        ]
