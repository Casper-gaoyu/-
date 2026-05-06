from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_TEMPLATE_KEY = "general"


def _build_general_template() -> dict[str, Any]:
    return {
        "template_key": "general",
        "template_label": "通用剧本模板",
        "template_family": "general",
        "description": "适用于推理、情感、机制、恐怖等常见剧本杀类型的通用创作模板。",
        "tags": ["通用", "多类型", "可共创"],
        "defaults": {
            "players": "6 人",
        },
        "input_form": {
            "title": "通用模板创作入口",
            "subtitle": "先确定类型、人数、时代和冲突，系统会在此基础上继续细化世界观、人物和剧情结构。",
            "preview_title": "模板交付结构预览",
            "preview_highlights": [
                "生成结构化剧本稿、角色设定、DM 手册和交付内容。",
                "支持对话共创与创作面板双向同步。",
                "适配悬疑、情感、恐怖、机制等常见题材。",
            ],
            "fields": [
                {
                    "id": "genre",
                    "label": "剧本类型",
                    "required": True,
                    "input": "radio",
                    "locked": False,
                    "default_value": "",
                    "options": ["现代悬疑", "情感", "恐怖", "机制", "欢乐"],
                    "placeholder": "请选择你想做的类型",
                },
                {
                    "id": "players",
                    "label": "参与人数",
                    "required": True,
                    "input": "select",
                    "locked": False,
                    "default_value": "6 人",
                    "options": ["4 人", "6 人", "8 人", "10 人"],
                    "placeholder": "选择或输入玩家人数",
                },
                {
                    "id": "era",
                    "label": "时代背景 / 故事场景",
                    "required": True,
                    "input": "radio_with_text",
                    "locked": False,
                    "default_value": "",
                    "options": ["现代都市", "现代校园", "民国小镇", "古风江湖", "未来科幻"],
                    "placeholder": "也可以补充更具体的场景，例如废弃疗养院、边陲小镇、海上邮轮",
                },
                {
                    "id": "conflict",
                    "label": "核心冲突 / 爽点",
                    "required": True,
                    "input": "multi_select_with_text",
                    "locked": False,
                    "default_value": "",
                    "options": ["密室命案", "旧案重启", "身份反转", "阵营对抗", "误会与救赎"],
                    "placeholder": "也可以直接写你想要的反转、冲突或情绪钩子",
                },
                {
                    "id": "special_requirements",
                    "label": "特殊要求",
                    "required": False,
                    "input": "textarea",
                    "locked": False,
                    "default_value": "",
                    "options": [],
                    "placeholder": "例如新手友好、重搜证、偏沉浸、不要超自然元素、需要强情绪回收",
                },
            ],
            "buttons": {
                "submit": "创建剧本会话",
                "reset": "重置需求",
                "preview": "模板预览",
            },
        },
        "dialog_guide_config": {
            "progress_stages": [
                {"progress": 0, "label": "未开始", "message": "等待确认基础创作参数。"},
                {"progress": 20, "label": "类型确认", "message": "已确认剧本类型，将据此收窄世界观和节奏。"},
                {"progress": 40, "label": "世界观补齐", "message": "正在明确时代、场景与基础故事框架。"},
                {"progress": 60, "label": "角色设定中", "message": "正在整理公开身份、隐藏秘密和角色动机。"},
                {"progress": 80, "label": "结构成型", "message": "正在补齐分幕结构、关键反转与主持信息。"},
                {"progress": 100, "label": "已完成", "message": "基础草案和交付骨架已经可继续迭代。"},
            ],
            "steps": [
                {"id": "genre", "label": "剧本类型", "question": "你想先做哪一类剧本杀？", "placeholder": "例如现代悬疑、情感、恐怖、机制、欢乐", "options": ["现代悬疑", "情感", "恐怖", "机制", "欢乐"]},
                {"id": "era", "label": "时代背景", "question": "故事发生在什么样的时代或空间里？", "placeholder": "例如现代校园、民国小镇、未来空间站", "options": ["现代都市", "现代校园", "民国小镇", "古风江湖", "未来科幻"]},
                {"id": "players", "label": "参与人数", "question": "这局预计多少人参与？", "placeholder": "例如 6 人", "options": ["4 人", "6 人", "8 人", "10 人"]},
                {"id": "conflict", "label": "核心冲突", "question": "这局最核心的冲突或反转是什么？", "placeholder": "例如密室命案、旧案重启、误会与救赎", "options": ["密室命案", "旧案重启", "身份反转", "阵营对抗"]},
                {"id": "protagonist", "label": "主控视角", "question": "你希望哪类角色承担主控视角或关键推进？", "placeholder": "例如记者、学生、侦探、医生", "options": ["记者", "学生", "侦探", "医生", "组织者"]},
                {"id": "goal", "label": "玩家目标", "question": "玩家最终最重要的目标是什么？", "placeholder": "例如找出真凶、揭开真相、完成站队", "options": ["找出真凶", "揭开真相", "完成站队", "完成任务"]},
            ],
            "skip_message": "基础设定已经收集完成，接下来会进入完整创作阶段。",
        },
        "co_creation_config": {
            "default_enabled": True,
            "toggle_label": "共创模式",
            "enabled_text": "共创",
            "disabled_text": "一键生成",
            "generate_current_label": "一键生成当前片段",
            "stage_locked_notice": "已确认的世界观、角色核心和剧情钩子会被锁定，后续只做局部优化。",
            "panel_sync_notice": "已同步你的修改，将基于新内容继续创作。",
            "input_sync_notice": "创作面板已更新，下一轮对话会严格遵循这些修改。",
            "stages": [
                {
                    "id": "worldview",
                    "title": "阶段 1：世界观共创",
                    "instruction": "先锁定故事基础。我会给你 2 个世界观方向，你来选定或改写。",
                    "placeholder": "例如保留第二个方案，但把地点改成封闭疗养院",
                    "generate_prompt": "请基于当前已确认信息，直接补完整的世界观与故事基础设定，不要改动已锁定内容。",
                    "options": [
                        {"id": "world-mystery", "label": "密闭悬案场景", "prompt": "选择世界观方案：密闭悬案场景", "value": "密闭悬案场景", "description": "封闭空间、关系压迫、线索集中回收。"},
                        {"id": "world-emotion", "label": "关系旧案场景", "prompt": "选择世界观方案：关系旧案场景", "value": "关系旧案场景", "description": "旧关系重聚、情绪对撞、真相缓慢揭开。"},
                    ],
                },
                {
                    "id": "roles",
                    "title": "阶段 2：角色共创",
                    "instruction": "接下来锁定人物基础。我会围绕公开身份、隐藏秘密和核心动机组织角色初稿。",
                    "placeholder": "例如把第二个角色改得更内向，公开身份改成实习记者",
                    "generate_prompt": "请基于已锁定世界观，直接补完整的角色公开设定、隐藏秘密和核心动机，不要改动已确认世界观。",
                    "options": [
                        {"id": "role-balanced", "label": "均衡群像", "prompt": "角色方向选择：均衡群像", "value": "均衡群像", "description": "每个角色戏份接近，便于新手开口。"},
                        {"id": "role-secretive", "label": "秘密拉扯", "prompt": "角色方向选择：秘密拉扯", "value": "秘密拉扯", "description": "强调角色互相试探与多层隐瞒。"},
                    ],
                },
                {
                    "id": "core_plot",
                    "title": "阶段 3：剧情核心共创",
                    "instruction": "现在锁定反转、冲突与核心爽点。我会优先提供 3 个冲突走向。",
                    "placeholder": "例如保留身份反转，但把真正爆发点放到第三幕对质",
                    "generate_prompt": "请基于当前已锁定内容，直接补完整的核心冲突、关键反转和分幕骨架，不要重写已锁定角色核心。",
                    "options": [
                        {"id": "plot-lockedroom", "label": "密室反转", "prompt": "剧情核心选择：密室反转", "value": "密室反转", "description": "强调空间封闭、时间线和现场逻辑。"},
                        {"id": "plot-timeline", "label": "时间线反转", "prompt": "剧情核心选择：时间线反转", "value": "时间线反转", "description": "通过信息差和顺序错位制造强回收。"},
                        {"id": "plot-identity", "label": "身份反转", "prompt": "剧情核心选择：身份反转", "value": "身份反转", "description": "角色立场与真实身份反差明显。"},
                    ],
                },
                {
                    "id": "detail_iteration",
                    "title": "阶段 4：细节迭代共创",
                    "instruction": "这一阶段由你主导。我只针对你指定的局部细节做精准优化，不覆盖已锁定框架。",
                    "placeholder": "例如把第二幕搜证改成对抗问答；把林岚写得更克制",
                    "generate_prompt": "请基于当前已锁定设定，补全当前用户点名的局部片段；如果用户没有明确方向，就提供 2-3 个细节优化建议。",
                    "options": [
                        {"id": "detail-scenes", "label": "细化分幕", "prompt": "请继续细化当前分幕与场景推进，但不要改动已锁定核心设定。", "value": "细化分幕", "description": "优先补场景、动作、推进节奏。"},
                        {"id": "detail-roles", "label": "细化人设", "prompt": "请继续细化当前人物性格、秘密表述和行为细节，但不要改动已锁定核心设定。", "value": "细化人设", "description": "优先补角色语言、细节和动机层次。"},
                        {"id": "detail-ideas", "label": "给我建议", "prompt": "我暂时卡住了，请先给我 2-3 个细节优化方向建议，不要直接大改当前框架。", "value": "给我建议", "description": "用户卡壳时给方向，不越权。"},
                    ],
                },
            ],
        },
        "creation_panel_config": {
            "nav_labels": {
                "draft": "剧本稿",
                "roles": "人物",
                "facts": "设定",
                "handoff": "交付包",
            },
            "operation_labels": {
                "save": "保存修改",
                "export": "导出交付",
                "back": "返回需求页",
                "history": "版本历史",
                "co_creation_toggle": "共创模式",
                "sync_notice": "同步提示",
            },
            "role_card_fields": ["公开身份", "隐藏秘密", "核心动机", "关键道具", "角色关系"],
            "setting_sections": ["世界观设定", "锁定事实", "剧情冲突", "分幕节奏"],
            "draft_structure": ["世界观", "角色基础", "剧情核心", "细节迭代"],
        },
        "delivery_package_config": {
            "title": "完整交付包",
            "description": "导出时会自动汇总当前剧本稿、角色设定、玩家手册、角色剧本、线索卡和复盘内容。",
            "modules": [
                {"id": "role_scripts", "label": "角色剧本", "count_label": "按角色分别导出"},
                {"id": "player_books", "label": "玩家手册", "count_label": "面向玩家的阅读稿"},
                {"id": "clue_cards", "label": "线索卡", "count_label": "用于打印和发放"},
                {"id": "dm_manual", "label": "DM 手册", "count_label": "主持流程与控场提示"},
                {"id": "review_sheet", "label": "复盘检查", "count_label": "行为记录与复盘草案"},
            ],
        },
    }


def _build_liangyuan_template() -> dict[str, Any]:
    return {
        "template_key": "liangyuan-chaos",
        "template_label": "良缘翻车局",
        "template_family": "comedy-dating",
        "description": "面向 6 人现代欢乐相亲翻车本的专属模板，强调社死、配对、熟人修罗场和翻车反转。",
        "tags": ["欢乐", "相亲", "翻车", "社死", "共创"],
        "defaults": {
            "genre": "欢乐",
            "players": "6 人",
            "era": "现代相亲局",
            "conflict": "相亲翻车",
            "goal": "完成配对",
            "protagonist": "组织者",
        },
        "input_form": {
            "title": "良缘翻车局专属创作入口",
            "subtitle": "围绕相亲局、社死名场面和关系翻车来创作，系统会自动适配模板专属 DM 与交付包逻辑。",
            "preview_title": "良缘翻车局交付预览",
            "preview_highlights": [
                "适配 6 人现代欢乐相亲翻车本。",
                "强调社死翻车、配对压力和熟人修罗场。",
                "自动联动角色剧本、主持流程和相亲局专属交付内容。",
            ],
            "fields": [
                {
                    "id": "genre",
                    "label": "剧本类型",
                    "required": True,
                    "input": "radio",
                    "locked": True,
                    "default_value": "欢乐",
                    "options": ["欢乐"],
                    "placeholder": "该模板固定为欢乐相亲本",
                },
                {
                    "id": "players",
                    "label": "参与人数",
                    "required": True,
                    "input": "select",
                    "locked": False,
                    "default_value": "6 人",
                    "options": ["4 人", "6 人", "8 人"],
                    "placeholder": "默认 6 人，支持偶数人数变体",
                },
                {
                    "id": "era",
                    "label": "相亲局场景",
                    "required": True,
                    "input": "radio_with_text",
                    "locked": False,
                    "default_value": "现代相亲局",
                    "options": ["现代相亲局", "社区联谊会", "职场相亲局"],
                    "placeholder": "也可以补充更具体的场地，例如高端会所、社区中心、露营联谊",
                },
                {
                    "id": "conflict",
                    "label": "核心翻车点",
                    "required": True,
                    "input": "multi_select_with_text",
                    "locked": False,
                    "default_value": "社死翻车",
                    "options": ["社死翻车", "互选修罗场", "前任乱入", "身份造假"],
                    "placeholder": "也可以直接写你想要的社死名场面或翻车原因",
                },
                {
                    "id": "special_requirements",
                    "label": "特殊要求",
                    "required": False,
                    "input": "textarea",
                    "locked": False,
                    "default_value": "",
                    "options": [],
                    "placeholder": "例如梗不要低俗、结局要强反转、社死要逐幕升级、CP 线要更明显",
                },
            ],
            "buttons": {
                "submit": "创建良缘翻车局会话",
                "reset": "重置需求",
                "preview": "模板预览",
            },
        },
        "dialog_guide_config": {
            "progress_stages": [
                {"progress": 0, "label": "未开始", "message": "等待确认相亲局基础设定。"},
                {"progress": 20, "label": "场景确认", "message": "已确认相亲局场景，开始收窄翻车氛围。"},
                {"progress": 40, "label": "嘉宾设定中", "message": "正在整理嘉宾公开身份、秘密与人设冲突。"},
                {"progress": 60, "label": "翻车核心中", "message": "正在锁定社死、互选、前任乱入等关键爽点。"},
                {"progress": 80, "label": "流程补齐", "message": "正在补齐相亲流程、破冰互动与终局翻转。"},
                {"progress": 100, "label": "已完成", "message": "良缘翻车局基础草案已可继续迭代。"},
            ],
            "steps": [
                {"id": "genre", "label": "剧本类型", "question": "这个模板固定是欢乐相亲本，确认按这个方向继续吗？", "placeholder": "良缘翻车局固定为欢乐相亲本", "options": ["欢乐"]},
                {"id": "era", "label": "相亲场景", "question": "相亲局准备放在什么场景里？", "placeholder": "例如现代相亲角、社区联谊、职场局", "options": ["现代相亲局", "社区联谊会", "职场相亲局"]},
                {"id": "players", "label": "参与人数", "question": "这场翻车局准备安排多少位嘉宾？", "placeholder": "例如 6 人", "options": ["4 人", "6 人", "8 人"]},
                {"id": "conflict", "label": "核心翻车点", "question": "这局最想打哪种翻车爽点？", "placeholder": "例如社死翻车、互选修罗场、前任乱入", "options": ["社死翻车", "互选修罗场", "前任乱入", "身份造假"]},
                {"id": "protagonist", "label": "主控视角", "question": "谁最适合带着玩家进入这场局？", "placeholder": "例如组织者、嘉宾、假身份潜入者", "options": ["组织者", "普通嘉宾", "网红嘉宾", "隐藏身份者"]},
                {"id": "goal", "label": "玩家目标", "question": "你希望结局把玩家带向什么结果？", "placeholder": "例如完成配对、翻车反转、公开真相", "options": ["完成配对", "翻车反转", "公开真相", "互选成功"]},
            ],
            "skip_message": "良缘翻车局的基础参数已经确认完成，接下来进入具体共创。",
        },
        "co_creation_config": {
            "default_enabled": True,
            "toggle_label": "共创模式",
            "enabled_text": "共创",
            "disabled_text": "一键生成",
            "generate_current_label": "一键生成当前片段",
            "stage_locked_notice": "已确认的相亲局场景、角色核心和翻车钩子会被锁定，后续不会擅自改动。",
            "panel_sync_notice": "已同步你的修改，将按新的相亲局设定继续推进。",
            "input_sync_notice": "创作面板的修改已经记录，下一轮会严格遵循这些设定。",
            "stages": [
                {
                    "id": "worldview",
                    "title": "阶段 1：世界观共创",
                    "instruction": "先锁定相亲局的基础氛围和舞台。我会给你 2 个相亲局场景方向。",
                    "placeholder": "例如保留第二个方案，但把场地改成社区广场联谊",
                    "generate_prompt": "请基于当前已确认信息，直接补完整的相亲局世界观、规则和基础舞台，不要改动已锁定内容。",
                    "options": [
                        {"id": "ly-world-social", "label": "社区相亲角", "prompt": "选择世界观方案：社区相亲角", "value": "社区相亲角", "description": "接地气、亲友围观、社死感强。"},
                        {"id": "ly-world-fancy", "label": "高端联谊局", "prompt": "选择世界观方案：高端联谊局", "value": "高端联谊局", "description": "人设包装更重，身份造假空间更大。"},
                    ],
                },
                {
                    "id": "roles",
                    "title": "阶段 2：角色共创",
                    "instruction": "接下来锁定嘉宾人设，重点整理公开身份、隐藏秘密和核心动机。",
                    "placeholder": "例如把男一改成嘴硬内向型，把女二的秘密改成隐瞒前任关系",
                    "generate_prompt": "请基于已锁定相亲局世界观，直接补完整的嘉宾公开身份、隐藏秘密和核心动机，不要改动已确认场景。",
                    "options": [
                        {"id": "ly-role-chaos", "label": "熟人修罗场", "prompt": "角色方向选择：熟人修罗场", "value": "熟人修罗场", "description": "强调角色之间原本就认识，翻车更刺激。"},
                        {"id": "ly-role-cp", "label": "CP 互选压力", "prompt": "角色方向选择：CP 互选压力", "value": "CP 互选压力", "description": "重点放在配对和选择压力。"},
                    ],
                },
                {
                    "id": "core_plot",
                    "title": "阶段 3：剧情核心共创",
                    "instruction": "现在锁定翻车反转和核心爽点。我会优先给你相亲局专属冲突选项。",
                    "placeholder": "例如保留互选修罗场，但把真正翻车安排在第三幕爆料环节",
                    "generate_prompt": "请基于当前已锁定内容，直接补完整的翻车核心、互选冲突和分幕骨架，不要重写已锁定嘉宾核心。",
                    "options": [
                        {"id": "ly-plot-social", "label": "社死翻车", "prompt": "剧情核心选择：社死翻车", "value": "社死翻车", "description": "爆料、造假、当众拆穿，名场面优先。"},
                        {"id": "ly-plot-pairing", "label": "互选修罗场", "prompt": "剧情核心选择：互选修罗场", "value": "互选修罗场", "description": "关系错位、互选错位、全员尴尬升级。"},
                        {"id": "ly-plot-ex", "label": "前任乱入", "prompt": "剧情核心选择：前任乱入", "value": "前任乱入", "description": "旧关系炸场，翻车直接升级。"},
                    ],
                },
                {
                    "id": "detail_iteration",
                    "title": "阶段 4：细节迭代共创",
                    "instruction": "你来指定局部细节，我只做精准优化，不覆盖已确认的翻车主线。",
                    "placeholder": "例如把第二幕社死游戏改成真心话大冒险；把主持人口吻改得更损一点",
                    "generate_prompt": "请基于当前已锁定设定，补完整当前用户点名的局部相亲局片段；如果用户没有明确方向，就提供 2-3 个细节优化建议。",
                    "options": [
                        {"id": "ly-detail-scenes", "label": "细化互动环节", "prompt": "请继续细化相亲局互动流程、游戏环节和爆料推进，但不要改动已锁定核心设定。", "value": "细化互动环节", "description": "优先补互动流程和社死节奏。"},
                        {"id": "ly-detail-roles", "label": "细化嘉宾人设", "prompt": "请继续细化嘉宾说话方式、公开伪装和翻车细节，但不要改动已锁定核心设定。", "value": "细化嘉宾人设", "description": "优先补人设细节和冲突表达。"},
                        {"id": "ly-detail-ideas", "label": "给我建议", "prompt": "我暂时卡住了，请先给我 2-3 个相亲局细节优化方向建议，不要直接大改当前框架。", "value": "给我建议", "description": "用户卡壳时只给方向，不越权。"},
                    ],
                },
            ],
        },
        "creation_panel_config": {
            "nav_labels": {
                "draft": "剧本稿",
                "roles": "嘉宾",
                "facts": "设定",
                "handoff": "交付包",
            },
            "operation_labels": {
                "save": "保存修改",
                "export": "导出交付",
                "back": "返回需求页",
                "history": "版本历史",
                "co_creation_toggle": "共创模式",
                "sync_notice": "同步提示",
            },
            "role_card_fields": ["公开身份", "隐藏秘密", "核心动机", "关键道具", "人物关系"],
            "setting_sections": ["相亲局舞台", "锁定事实", "翻车核心", "互动流程"],
            "draft_structure": ["相亲局开场", "嘉宾人设", "翻车核心", "细节打磨"],
        },
        "delivery_package_config": {
            "title": "良缘翻车局完整交付包",
            "description": "导出时会自动带出嘉宾剧本、玩家手册、DM 手册、线索卡和相亲局复盘内容。",
            "modules": [
                {"id": "role_scripts", "label": "角色剧本", "count_label": "按嘉宾分别导出"},
                {"id": "player_books", "label": "玩家手册", "count_label": "包含相亲局玩法说明"},
                {"id": "clue_cards", "label": "线索卡", "count_label": "包含翻车证据与爆料素材"},
                {"id": "dm_manual", "label": "DM 手册", "count_label": "包含控场、破冰、翻车推进"},
                {"id": "review_sheet", "label": "复盘检查", "count_label": "用于回收爆点与行为记录"},
            ],
        },
    }


class TemplateConfigService:
    def __init__(self) -> None:
        self._templates: dict[str, dict[str, Any]] = {
            "general": _build_general_template(),
            "liangyuan-chaos": _build_liangyuan_template(),
        }

    def list_templates(self) -> list[dict[str, Any]]:
        return [deepcopy(item) for item in self._templates.values()]

    def get_template(self, template_key: str | None) -> dict[str, Any]:
        key = (template_key or DEFAULT_TEMPLATE_KEY).strip().lower()
        return deepcopy(self._templates.get(key) or self._templates[DEFAULT_TEMPLATE_KEY])

    def resolve_template_key(self, template_key: str | None, brief: str = "") -> str:
        normalized = (template_key or "").strip().lower()
        if normalized in self._templates:
            return normalized

        text = (brief or "").strip()
        if any(token in text for token in ("良缘翻车局", "相亲翻车", "相亲局", "社死翻车", "CP 配对", "互选修罗场")):
            return "liangyuan-chaos"
        return DEFAULT_TEMPLATE_KEY
