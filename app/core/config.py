from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 4
    secret_key: str = "change-me"
    api_key_header: str = "X-API-Key"
    valid_api_keys: str = "dev-key"

    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.0

    # Pinecone
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "rag-index"
    pinecone_namespace: str = "default"
    pinecone_embedding_dim: int = 1536

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    similarity_threshold: float = 0.70

    # Logging
    log_level: str = "INFO"

    @property
    def api_keys(self) -> List[str]:
        return [k.strip() for k in self.valid_api_keys.split(",") if k.strip()]

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache()
def get_settings() -> Settings:
    return Settings()