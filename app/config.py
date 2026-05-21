"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables and .env file."""

    # LLM
    groq_api_key: str = ""
    primary_model: str = "llama-3.3-70b-versatile"
    reasoning_model: str = "deepseek-r1-distill-llama-70b"

    # Embeddings
    nvidia_api_key: str = ""
    embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    embedding_dimension: int = 2048

    # Qdrant — set QDRANT_URL for cloud (Koyeb), leave blank for local dev
    qdrant_url: str = ""           # e.g. https://xxxx.us-east4-0.gcp.cloud.qdrant.io
    qdrant_api_key: str = ""       # Qdrant Cloud API key
    qdrant_host: str = "localhost" # used only when QDRANT_URL is not set
    qdrant_port: int = 6333        # used only when QDRANT_URL is not set
    qdrant_collection_name: str = "academic_papers"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    upload_dir: str = "./data/uploads"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5

    # Chainlit
    fastapi_base_url: str = "http://localhost:8000"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
