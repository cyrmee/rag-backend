from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    # Embeddings + vision captioning stay on Ollama's native API; chat runs
    # on a separate OpenAI-compatible (vLLM) server - see app/generation.py.
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "qwen3-embedding:latest"
    chat_base_url: str = "http://localhost:8101/v1"
    chat_api_key: str | None = None
    chat_model: str = "chat"
    # Must match the chat server's --max-model-len: vLLM fixes the context
    # window at server start (no per-request equivalent of Ollama's
    # num_ctx), so this only tells the app how much room it has - e.g.
    # app/attachments.py budgets attached-file text as a slice of it.
    chat_num_ctx: int = 32768
    embed_dim: int = 1024
    chunk_size: int = 500
    top_k: int = 15
    max_agent_iterations: int = 4
    # Council mode (/ask with council=true, see app/council.py): how many
    # search angles the content planner may propose, and how many pooled
    # document chunks (after fusing every angle's results) reach the model.
    council_angles: int = 6
    council_max_chunks: int = 20
    vision_model: str = "qwen3-vl-caption:latest"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "ragminio"
    minio_secret_key: str = "ragminiosecret"
    minio_bucket: str = "rag-images"
    minio_secure: bool = False
    searxng_base_url: str = "http://localhost:8080"


settings = Settings()
