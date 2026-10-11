from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Generation. "auto" = Groq if GROQ_API_KEY is set, else Hugging Face.
    llm_provider: str = "auto"  # auto | groq | hf
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    hf_token: str = ""
    hf_model_id: str = ""
    hf_endpoint_url: str = ""
    max_new_tokens: int = 1024
    temperature: float = 0.2

    # Intent gate (free): rules + a small Groq model decide small talk vs. real question.
    intent_model: str = ""  # empty = use GROQ_MODEL; set to a smaller Groq model to save quota
    intent_max_words: int = 8
    intent_threshold: float = 0.7
    # Preferred decider: free structured-decision model on OpenRouter (falls back to Groq if no key / on error).
    openrouter_api_key: str = ""
    openrouter_intent_model: str = "inception/mercury-decide:free"
    openrouter_systemone_url: str = "https://openrouter.ai/api/v1/systemone"

    # Hosted embeddings + reranker (Jina). If JINA_API_KEY is set they replace the local models.
    jina_api_key: str = ""
    jina_embedding_model: str = "jina-embeddings-v3"
    jina_reranker_model: str = "jina-reranker-v2-base-multilingual"
    # Jina rerank scores are 0..1 (not logits). UNCALIBRATED starting value; tune with the eval.
    jina_min_rerank_score: float = 0.1
    # Token controls (rerank is the main cost: every candidate chunk is sent on every question).
    jina_rerank_candidates: int = 8  # chunks sent to the reranker per question
    jina_rerank_max_chars: int = 1200  # ~300 tokens per chunk
    jina_token_budget: int = 0  # 0 = unlimited; otherwise stop calling Jina once this many tokens are used

    # Local models (CPU), used when no JINA_API_KEY
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Access control / limits (set APP_PASSWORD on any public deployment)
    app_password: str = ""  # empty = no login (local use). Browser asks for it via HTTP Basic auth; any username.
    max_upload_mb: int = 10

    # Storage
    data_dir: Path = ROOT / "data"
    collection: str = "default"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    retrieve_k: int = 20
    final_k: int = 4
    # Top reranker logit below this => answer with a "weak context" caveat.
    # Spot check on one small document: relevant questions -6.1 to +2.5, unrelated ~-10.2.
    # Re-calibrate against the eval set once real documents are in.
    min_rerank_score: float = -8.0

    # Optional agentic nodes (enable after evaluating they help)
    enable_query_rewrite: bool = False
    enable_llm_grading: bool = False
    max_attempts: int = 2

    @property
    def provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        return "groq" if self.groq_api_key else "hf"

    @property
    def use_jina(self) -> bool:
        return bool(self.jina_api_key)

    @property
    def rerank_threshold(self) -> float:
        return self.jina_min_rerank_score if self.use_jina else self.min_rerank_score

    @property
    def llm_configured(self) -> bool:
        if self.provider == "groq":
            return bool(self.groq_api_key)
        return bool(self.hf_model_id or self.hf_endpoint_url)

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "meta.db"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"


settings = Settings()
