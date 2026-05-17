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
    embedding_model: str = "nvidia/llama-3.2-nemoretriever-300m-embed-v1"
    embedding_dimension: int = 2048

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
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
