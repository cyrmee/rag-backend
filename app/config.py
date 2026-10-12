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
    max_agent_iterations: int = 10
    # Council mode (/ask with council=true, see app/council.py): how many of
    # the who/what/when/where/why/how search angles to use (6 = all that
    # apply), and how many pooled document chunks (after fusing every
    # angle's results) reach the model.
    council_angles: int = 6
    council_max_chunks: int = 20
    # When a request doesn't say which mode to use, route complex questions
    # to council mode automatically (app/council.py is_complex).
    auto_council: bool = True
    # Research mode (/ask with mode="research", see app/research.py): how
    # many gather/read rounds (the first plus up to N-1 gap-filling ones),
    # how many documents the planner may pick from the summaries, how many
    # chunks each search query may contribute, how many chunks a round
    # reads at most, and how many tokens of excerpts go into one reading
    # call. These set the latency: every chunk read is prompt text the
    # chat server has to process, outside the answer's own prompt.
    research_max_rounds: int = 2
    research_max_documents: int = 20
    research_chunks_per_query: int = 25
    research_max_chunks: int = 250
    research_batch_tokens: int = 7000
    # Graph retrieval (app/graph.py): extract entities and facts from every
    # chunk at ingestion and add the knowledge graph's ranking as one more
    # evidence source in council and research mode. Off: nothing is
    # extracted and retrieval is unchanged. Existing documents need
    # scripts/maintenance/backfill_graph.py once it's switched on.
    graph_retrieval: bool = False
    vision_model: str = "qwen3-vl-caption:latest"
    # Captions are asked for with Qwen3-VL's chat template and an empty
    # think block prefilled (app/vision.py) - the only way the Ollama
    # qwen3-vl tags skip their thinking. Off: the plain caption prompt is
    # sent, for a vision model with another template.
    vision_no_think_prefill: bool = True
    # Document parsing (pymupdf, python-docx, LibreOffice) is CPU work and
    # runs in this many worker processes, so one long document's parsing
    # overlaps other documents' model calls instead of stalling them.
    extract_workers: int = 4
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "ragminio"
    minio_secret_key: str = "ragminiosecret"
    minio_bucket: str = "rag-images"
    minio_secure: bool = False
    # Host:port browsers use to reach MinIO, for the presigned links to
    # original files in /ask sources. MINIO_ENDPOINT is how the backend
    # itself reaches MinIO ("minio:9000" inside docker compose, "localhost"
    # on one machine) - neither resolves from a user's browser. Unset means
    # links are signed for MINIO_ENDPOINT.
    minio_public_endpoint: str | None = None
    searxng_base_url: str = "http://localhost:8080"
    # Authentication (app/auth.py): every route but /health needs a
    # Keycloak access token. The issuer is the realm URL exactly as it
    # appears in tokens' "iss" claim (the address browsers log in at);
    # the JWKS URL is where this process fetches the signing keys, which
    # can differ inside docker compose. Unset means issuer + the standard
    # certs path. Tokens must name the audience in "aud" - an audience
    # mapper on the frontend's client adds it.
    keycloak_issuer: str = "http://localhost:8180/realms/rag"
    keycloak_jwks_url: str | None = None
    keycloak_audience: str = "rag-backend"


settings = Settings()
