from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "qwen3-embedding:latest"
    chat_model: str = "deepseek-r1:70b"
    embed_dim: int = 1024
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 5
    max_agent_iterations: int = 4
    vision_model: str = "qwen3-vl-caption:latest"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "ragminio"
    minio_secret_key: str = "ragminiosecret"
    minio_bucket: str = "rag-images"
    minio_secure: bool = False


settings = Settings()
