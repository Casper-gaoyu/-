from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CharacterProfile(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = ""
    role: str = ""
    motivation: str = ""


class TimelineEntry(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    time: str = ""
    public_story: str = ""
    real_action: str = ""


class InventoryItem(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = ""
    description: str = ""


class WorldConfig(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str
    era: Literal["ancient_wuxia", "modern_suspense", "future_sci-fi", "custom"]
    custom_era: str = ""
    script_type: str
    core_conflict: str
    role_name: str
    public_identity: str
    killer_status: Literal["\u662f", "\u5426", "\u5e2e\u51f6"]
    character_background: str
    core_secret: str
    motivation: str
    public_task: str
    hidden_task: str
    key_props: str = ""
    key_clues: str = ""
    characters: list[CharacterProfile] = Field(default_factory=list)
    timeline_entries: list[TimelineEntry] = Field(default_factory=list)
    inventory_items: list[InventoryItem] = Field(default_factory=list)

    @field_validator("timeline_entries", mode="before")
    @classmethod
    def ensure_timeline_entries(cls, value: list[dict] | None) -> list[dict]:
        return value or [{}, {}]

    @field_validator("inventory_items", mode="before")
    @classmethod
    def ensure_inventory_items(cls, value: list[dict] | None) -> list[dict]:
        return value or [{}, {}]

    @property
    def resolved_era(self) -> str:
        return self.custom_era if self.era == "custom" and self.custom_era else self.era


class BranchOption(BaseModel):
    id: str
    title: str
    description: str


class NpcDialogue(BaseModel):
    speaker: str
    intent: str
    line: str


class MemoryStats(BaseModel):
    token_budget: int
    estimated_tokens: int
    compression_applied: bool
    history_entries: int


class AdvanceRequest(BaseModel):
    choice_index: int | None = None


class ScriptStateResponse(BaseModel):
    session_id: str
    title: str
    stage: str
    content: str
    branches: list[BranchOption]
    npc_dialogues: list[NpcDialogue]
    memory_summary: list[str]
    memory_stats: MemoryStats
    logic_checks: list[str]
    is_finished: bool
    provider: str


class CreateSessionResponse(BaseModel):
    session_id: str
    title: str
    segment: ScriptStateResponse


class ReviewResponse(BaseModel):
    session_id: str
    title: str
    review_content: str
    provider: str


class AuthRegisterRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=100)


class AuthLoginRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=100)


class UserProfileResponse(BaseModel):
    id: int
    username: str


class AuthResponse(BaseModel):
    token: str
    user: UserProfileResponse


class AgentMessage(BaseModel):
    id: str = ""
    role: Literal["user", "assistant"]
    content: str
    created_at: str = ""
    revision_id: str = ""
    can_rollback: bool = False
    rollback_label: str = ""


class AgentSuggestion(BaseModel):
    id: str
    label: str
    prompt: str
    category: str = "create"


class AgentRoleCard(BaseModel):
    id: str
    name: str
    archetype: str
    public_identity: str
    hidden_secret: str
    motivation: str
    key_prop: str
    tags: list[str] = Field(default_factory=list)
    behavior_rules: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)


class AgentBehaviorRecord(BaseModel):
    id: str = ""
    timestamp: str = ""
    role_name: str = ""
    action_type: str = ""
    key_output: str = ""
    dm_note: str = ""


class AgentOnboardingState(BaseModel):
    enabled: bool
    completed: bool
    progress: int
    total_questions: int
    template_key: str = "general"
    current_question_id: str = ""
    current_question: str = ""
    current_placeholder: str = ""
    answers: dict[str, str] = Field(default_factory=dict)


class AgentRoleScript(BaseModel):
    id: str
    role_name: str
    title: str
    cover_age: str = ""
    cover_identity: str = ""
    cover_tagline: str = ""
    source: Literal["auto", "manual"] = "auto"
    updated_at: str = ""
    content: str
    generation_mode: str = ""
    fallback_used: bool = False
    content_length: int = 0
    llm_provider: str = ""
    llm_model: str = ""


class AgentRoleScriptPromptBlueprint(BaseModel):
    id: str
    role_name: str
    user_requirement: str = ""
    objective: str = ""
    style: str = ""
    must_keep: list[str] = Field(default_factory=list)
    focus: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    source: Literal["auto", "manual"] = "auto"


class AgentWorkspaceSummary(BaseModel):
    title: str
    era: str
    genre: str
    focus: str
    revision_count: int = 0
    template_key: str = "general"
    template_label: str = "通用剧本模板"
    template_family: str = "general"


class AgentIntentParams(BaseModel):
    type: str = ""
    people: str = ""
    time: str = ""
    style: str = ""
    theme: str = ""
    scene: str = ""


class AgentIntentState(BaseModel):
    current_intent: str
    next_action: str
    progress_label: str
    params: AgentIntentParams = Field(default_factory=AgentIntentParams)


class AgentTaskItem(BaseModel):
    id: str
    module: str
    title: str
    detail: str
    status: Literal["done", "active", "pending"]


class AgentStructureStage(BaseModel):
    id: str
    title: str
    ratio: int
    duration_minutes: int
    objective: str
    deliverables: list[str] = Field(default_factory=list)


class AgentDmCue(BaseModel):
    stage_id: str
    stage_title: str
    opening_line: str
    transition_line: str
    emergency_line: str
    dm_tip: str


class AgentOutputSection(BaseModel):
    id: str
    title: str
    description: str
    format_hint: str


class AgentHostManualFrontMatter(BaseModel):
    title: str
    era: str
    genre: str
    recommended_players: str
    current_focus: str
    synopsis: list[str] = Field(default_factory=list)


class AgentHostManualSceneCard(BaseModel):
    code: str
    title: str
    duration_minutes: int
    objective: str
    materials: list[str] = Field(default_factory=list)
    player_count_variants: list[str] = Field(default_factory=list)
    dm_read_aloud: str
    dm_actions: list[str] = Field(default_factory=list)
    trigger_conditions: list[str] = Field(default_factory=list)
    trigger_checks: list[str] = Field(default_factory=list)
    branch_handling: list[str] = Field(default_factory=list)
    resolution_rules: list[str] = Field(default_factory=list)
    record_points: list[str] = Field(default_factory=list)
    clue_release: list[str] = Field(default_factory=list)
    wrap_up: str


class AgentHostManualAct(BaseModel):
    id: str
    title: str
    duration_minutes: int
    objective: str
    deliverables: list[str] = Field(default_factory=list)
    scenes: list[AgentHostManualSceneCard] = Field(default_factory=list)


class AgentHostManualFixedEvent(BaseModel):
    code: str
    stage_title: str
    scene_title: str
    trigger_timing: str
    trigger_condition: str
    host_action: str
    resulting_state: str


class AgentHostManualAppendixItem(BaseModel):
    id: str
    title: str
    items: list[str] = Field(default_factory=list)


class AgentHostManualVisualAttachment(BaseModel):
    id: str
    title: str
    description: str
    asset_kind: str = "generated"


class AgentHostManualSchema(BaseModel):
    front_matter: AgentHostManualFrontMatter
    material_checklist: list[str] = Field(default_factory=list)
    opening_preparation: list[str] = Field(default_factory=list)
    player_count_variants: list[str] = Field(default_factory=list)
    fixed_events: list[AgentHostManualFixedEvent] = Field(default_factory=list)
    acts: list[AgentHostManualAct] = Field(default_factory=list)
    truth_reveal: list[str] = Field(default_factory=list)
    ending_adjudication: list[str] = Field(default_factory=list)
    review_points: list[str] = Field(default_factory=list)
    answer_key: list[str] = Field(default_factory=list)
    npc_reference: list[str] = Field(default_factory=list)
    visual_attachments: list[AgentHostManualVisualAttachment] = Field(default_factory=list)
    appendix: list[AgentHostManualAppendixItem] = Field(default_factory=list)


class AgentMemoryStats(BaseModel):
    token_budget: int
    estimated_tokens: int
    compression_applied: bool
    message_entries: int
    memory_nodes: int


class AgentLlmProviderStatus(BaseModel):
    provider: str
    enabled: bool
    model: str
    reason: str = ""


class AgentLlmRuntimeProfileResponse(BaseModel):
    recommended_provider: str
    recommended_model: str
    providers: list[AgentLlmProviderStatus] = Field(default_factory=list)
    single_provider_mode: bool = False
    mock_fallback_enabled: bool = True


class AgentLlmDiagnosticStatus(BaseModel):
    provider: str
    enabled: bool
    model: str
    status: Literal["ok", "warning", "error", "skipped"]
    detail: str = ""
    latency_ms: int | None = None


class AgentLlmDiagnosticsResponse(BaseModel):
    checked_at: str
    recommended_provider: str
    recommended_model: str
    providers: list[AgentLlmDiagnosticStatus] = Field(default_factory=list)
    single_provider_mode: bool = False
    mock_fallback_enabled: bool = True


class AgentLlmStrategyCard(BaseModel):
    id: str
    title: str
    provider: str
    model: str
    recommendation: str
    strengths: list[str] = Field(default_factory=list)
    tradeoffs: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)


class AgentLlmStrategyResponse(BaseModel):
    document_title: str
    quickstart_summary: str
    recommended_primary_provider: str
    recommended_primary_model: str
    engineering_focus: list[str] = Field(default_factory=list)
    cards: list[AgentLlmStrategyCard] = Field(default_factory=list)


class AgentPendingPlan(BaseModel):
    request: str
    focus_sections: list[str] = Field(default_factory=list)
    protected_sections: list[str] = Field(default_factory=list)
    impacted_roles: list[str] = Field(default_factory=list)
    impacted_stages: list[str] = Field(default_factory=list)
    impacted_clues: list[str] = Field(default_factory=list)
    excluded_roles: list[str] = Field(default_factory=list)
    excluded_stages: list[str] = Field(default_factory=list)
    excluded_clues: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    next_action: str = ""
    awaiting_confirmation: bool = True


class AgentSessionCreateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    brief: str = Field(min_length=1, max_length=4000)
    llm_provider: str = ""
    llm_model: str = ""
    template_key: str = ""
    template_form_data: dict[str, str] = Field(default_factory=dict)


class AgentMessageRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=4000)


class AgentOnboardingControlRequest(BaseModel):
    action: Literal["jump", "skip", "reset"]
    question_id: str = ""


class AgentRollbackRequest(BaseModel):
    revision_id: str = Field(min_length=1)


class AgentDraftUpdateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    draft_content: str = Field(min_length=1, max_length=20000)
    title: str = Field(default="", max_length=200)


class AgentConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(default="", max_length=200)
    genre: str = ""
    era: str = ""
    players: str = ""
    conflict: str = ""
    protagonist: str = ""
    goal: str = ""
    llm_provider: str = ""
    llm_model: str = ""
    draft_mode: str = ""
    role_script_length_mode: str = ""
    co_creation_enabled: bool | None = None


class AgentRoleCardsUpdateRequest(BaseModel):
    role_cards: list[AgentRoleCard] = Field(default_factory=list)


class AgentRoleScriptsUpdateRequest(BaseModel):
    role_scripts: list[AgentRoleScript] = Field(default_factory=list)


class AgentBehaviorRecordsUpdateRequest(BaseModel):
    behavior_records: list[AgentBehaviorRecord] = Field(default_factory=list)


class AgentFactsUpdateRequest(BaseModel):
    facts: list[str] = Field(default_factory=list)


class AgentRoleScriptPromptBlueprintsUpdateRequest(BaseModel):
    role_script_prompt_blueprints: list[AgentRoleScriptPromptBlueprint] = Field(default_factory=list)


class AgentConceptImportAnalysis(BaseModel):
    title: str = ""
    genre: str = ""
    era: str = ""
    players: str = ""
    conflict: str = ""
    protagonist: str = ""
    goal: str = ""
    role_cards: list[AgentRoleCard] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    summary: str = ""


class AgentConceptImportPreviewResponse(BaseModel):
    class SourceItem(BaseModel):
        filename: str
        category: Literal["world", "roles", "clues", "other"] = "other"
        analysis: AgentConceptImportAnalysis

    filename: str
    filenames: list[str] = Field(default_factory=list)
    analysis: AgentConceptImportAnalysis
    sources: list[SourceItem] = Field(default_factory=list)


class AgentConceptImportApplyRequest(BaseModel):
    analysis: AgentConceptImportAnalysis
    apply_fields: list[Literal["title", "genre", "era", "players", "conflict", "protagonist", "goal", "role_cards", "facts"]] = Field(default_factory=list)
    replace_role_names: list[str] = Field(default_factory=list)
    selected_facts: list[str] = Field(default_factory=list)


class AgentSessionListItem(BaseModel):
    session_id: str
    title: str
    last_message: str
    updated_at: str
    provider: str
    is_hidden: bool = False
    status: Literal["drafting", "completed", "edited"]
    onboarding_completed: bool
    revision_count: int
    template_key: str = "general"
    template_label: str = "通用剧本模板"


class AgentTemplateFormField(BaseModel):
    id: str
    label: str
    required: bool = False
    input: str = "text"
    locked: bool = False
    default_value: str = ""
    options: list[str] = Field(default_factory=list)
    placeholder: str = ""
    hint: str = ""


class AgentTemplateModuleItem(BaseModel):
    id: str
    label: str
    count_label: str = ""


class AgentTemplateInputFormConfig(BaseModel):
    title: str
    subtitle: str
    preview_title: str = ""
    preview_highlights: list[str] = Field(default_factory=list)
    fields: list[AgentTemplateFormField] = Field(default_factory=list)
    buttons: dict[str, str] = Field(default_factory=dict)


class AgentTemplateGuideStep(BaseModel):
    id: str
    label: str
    question: str
    placeholder: str = ""
    options: list[str] = Field(default_factory=list)


class AgentTemplateGuideProgressStage(BaseModel):
    progress: int
    label: str
    message: str


class AgentTemplateDialogGuideConfig(BaseModel):
    progress_stages: list[AgentTemplateGuideProgressStage] = Field(default_factory=list)
    steps: list[AgentTemplateGuideStep] = Field(default_factory=list)
    skip_message: str = ""


class AgentCoCreationOption(BaseModel):
    id: str
    label: str
    prompt: str
    value: str = ""
    description: str = ""


class AgentCoCreationStageConfig(BaseModel):
    id: str
    title: str
    instruction: str
    placeholder: str = ""
    generate_prompt: str = ""
    options: list[AgentCoCreationOption] = Field(default_factory=list)


class AgentTemplateCoCreationConfig(BaseModel):
    default_enabled: bool = True
    toggle_label: str = "共创模式"
    enabled_text: str = "共创"
    disabled_text: str = "一键生成"
    generate_current_label: str = "一键生成当前片段"
    stage_locked_notice: str = ""
    panel_sync_notice: str = ""
    input_sync_notice: str = ""
    stages: list[AgentCoCreationStageConfig] = Field(default_factory=list)


class AgentTemplateCreationPanelConfig(BaseModel):
    nav_labels: dict[str, str] = Field(default_factory=dict)
    operation_labels: dict[str, str] = Field(default_factory=dict)
    role_card_fields: list[str] = Field(default_factory=list)
    setting_sections: list[str] = Field(default_factory=list)
    draft_structure: list[str] = Field(default_factory=list)


class AgentTemplateDeliveryPackageConfig(BaseModel):
    title: str
    description: str = ""
    modules: list[AgentTemplateModuleItem] = Field(default_factory=list)


class AgentTemplateConfigResponse(BaseModel):
    template_key: str
    template_label: str
    template_family: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    defaults: dict[str, str] = Field(default_factory=dict)
    input_form: AgentTemplateInputFormConfig
    dialog_guide_config: AgentTemplateDialogGuideConfig
    co_creation_config: AgentTemplateCoCreationConfig
    creation_panel_config: AgentTemplateCreationPanelConfig
    delivery_package_config: AgentTemplateDeliveryPackageConfig


class AgentCoCreationState(BaseModel):
    enabled: bool = True
    current_stage_id: str = ""
    current_stage_title: str = ""
    stage_index: int = 0
    total_stages: int = 0
    awaiting_user: bool = True
    sync_notice: str = ""
    locked_fields: list[str] = Field(default_factory=list)
    locked_values: dict[str, str] = Field(default_factory=dict)
    user_notes: dict[str, str] = Field(default_factory=dict)
    stage_options: list[AgentCoCreationOption] = Field(default_factory=list)


class AgentStateResponse(BaseModel):
    session_id: str
    title: str
    messages: list[AgentMessage]
    draft_content: str
    suggestions: list[AgentSuggestion]
    facts: list[str]
    role_cards: list[AgentRoleCard]
    role_scripts: list[AgentRoleScript] = Field(default_factory=list)
    behavior_records: list[AgentBehaviorRecord] = Field(default_factory=list)
    onboarding: AgentOnboardingState
    co_creation: AgentCoCreationState
    summary: AgentWorkspaceSummary
    intent: AgentIntentState
    task_queue: list[AgentTaskItem]
    structure_plan: list[AgentStructureStage]
    dm_script: list[AgentDmCue]
    output_outline: list[AgentOutputSection]
    logic_checks: list[str] = Field(default_factory=list)
    memory_summary: list[str] = Field(default_factory=list)
    memory_stats: AgentMemoryStats
    pending_plan: AgentPendingPlan | None = None
    host_manual: AgentHostManualSchema
    provider: str
    llm_provider: str = ""
    llm_model: str = ""
    role_script_length_mode: str = ""
    template_config: AgentTemplateConfigResponse | None = None
