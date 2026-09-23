import httpx

from app.config import settings

MAX_RESULTS = 5
MAX_SNIPPET_CHARS = 500


async def search_web(query: str) -> list[dict]:
    """Queries the self-hosted SearXNG instance (docker-compose service
    `searxng`) and returns up to MAX_RESULTS {title, url, content} dicts.
    SearXNG aggregates results from several upstream search engines, so
    this is the only thing in the backend that ever talks to the open
    internet - the model itself never does.  Returns [] on any failure
    (SearXNG down, no results, etc.) rather than raising, so a bad web
    search doesn't take down the whole agent turn - the model just sees
    an empty result set and can say so."""
    try:
        async with httpx.AsyncClient(base_url=settings.searxng_base_url, timeout=15.0) as client:
            resp = await client.get("/search", params={"q": query, "format": "json"})
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError):
        return []

    results = []
    for row in data.get("results", [])[:MAX_RESULTS]:
        content = (row.get("content") or "")[:MAX_SNIPPET_CHARS]
        results.append({
            "title": row.get("title") or "",
            "url": row.get("url") or "",
            "content": content,
        })
    return results
