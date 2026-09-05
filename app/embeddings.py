import httpx

from app.config import settings


async def embed_text(text: str) -> list[float]:
    async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=60.0) as client:
        resp = await client.post(
            "/api/embed",
            json={
                "model": settings.embed_model,
                "input": text,
                "dimensions": settings.embed_dim,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"][0]
