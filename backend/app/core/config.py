from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env", override=True)

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_database_url(database_url: str) -> str:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return database_url

    raw_path = database_url.removeprefix(prefix)
    if not raw_path:
        return database_url

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return database_url

    resolved = (BASE_DIR / candidate).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{resolved.as_posix()}"


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "剧本杀剧本智能生成助手")
    environment: str = os.getenv("ENVIRONMENT", "development")
    api_prefix: str = "/api"
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(DATA_DIR / 'story_assistant.db').as_posix()}",
    )
    redis_url: str | None = os.getenv("REDIS_URL") or None
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock").strip().lower()
    llm_single_provider_only: bool = _env_flag("LLM_SINGLE_PROVIDER_ONLY")
    llm_single_provider: str = os.getenv("LLM_SINGLE_PROVIDER", "").strip().lower()
    llm_single_model: str = os.getenv("LLM_SINGLE_MODEL", "").strip()
    llm_disable_mock_fallback: bool = _env_flag("LLM_DISABLE_MOCK_FALLBACK")
    doubao_api_key: str | None = os.getenv("DOUBAO_API_KEY") or os.getenv("ARK_API_KEY") or None
    doubao_model: str = os.getenv("DOUBAO_MODEL", os.getenv("ARK_MODEL", "doubao-seed-1-6-251015"))
    doubao_base_url: str = os.getenv(
        "DOUBAO_BASE_URL",
        os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
    ).rstrip("/")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    dashscope_api_key: str | None = os.getenv("DASHSCOPE_API_KEY") or None
    dashscope_model: str = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")
    siliconflow_api_key: str | None = os.getenv("SILICONFLOW_API_KEY") or None
    siliconflow_model: str = os.getenv("SILICONFLOW_MODEL", "Pro/zai-org/GLM-5")
    siliconflow_base_url: str = os.getenv(
        "SILICONFLOW_BASE_URL",
        "https://api.siliconflow.cn/v1",
    ).rstrip("/")
    zhipu_api_key: str | None = os.getenv("ZHIPU_API_KEY") or None
    zhipu_model: str = os.getenv("ZHIPU_MODEL", "glm-4-flash")
    zhipu_base_url: str = os.getenv(
        "ZHIPU_BASE_URL",
        "https://open.bigmodel.cn/api/paas/v4",
    ).rstrip("/")
    token_budget: int = int(os.getenv("TOKEN_BUDGET", "8000"))
    agent_primary_timeout_seconds: int = int(os.getenv("AGENT_PRIMARY_TIMEOUT_SECONDS", "45"))
    agent_retry_timeout_seconds: int = int(os.getenv("AGENT_RETRY_TIMEOUT_SECONDS", "60"))
    llm_http_retry_attempts: int = int(os.getenv("LLM_HTTP_RETRY_ATTEMPTS", "2"))
    llm_diagnostic_timeout_seconds: int = int(os.getenv("LLM_DIAGNOSTIC_TIMEOUT_SECONDS", "30"))
    provider_failure_cooldown_seconds: int = int(os.getenv("PROVIDER_FAILURE_COOLDOWN_SECONDS", "90"))
    real_provider_chain_budget_seconds: int = int(os.getenv("REAL_PROVIDER_CHAIN_BUDGET_SECONDS", "90"))
    allowed_origins: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.database_url = _normalize_database_url(self.database_url)
        raw = os.getenv(
            "ALLOWED_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        )
        self.allowed_origins = [item.strip() for item in raw.split(",") if item.strip()]


settings = Settings()
