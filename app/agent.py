import logging

from pgvector import Vector

from app.config import settings
from app.db import get_connection
from app.embeddings import embed_text
from app.generation import DESCRIBE_IMAGE_TOOL, RETRIEVE_TOOL, chat_with_tools
from app.storage import get_image_bytes
from app.vision import describe_image as vision_describe_image

logger = logging.getLogger(__name__)

NO_RESULTS_MESSAGE = "No results found for this query."
AGENT_TOOLS = [RETRIEVE_TOOL, DESCRIBE_IMAGE_TOOL]


async def retrieve(query: str, top_k: int | None = None) -> list[str]:
    """Same embedding + retrieval logic as the existing /ask route: embed
    the query, order documents by cosine distance, return the content
    strings of the top matches."""
    query_vector = Vector(await embed_text(query))
    limit = top_k if top_k is not None else settings.top_k

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select content
                from documents
                order by embedding <=> %s
                limit %s
                """,
                (query_vector, limit),
            )
            rows = await cur.fetchall()

    return [row[0] for row in rows]


async def _retrieve_for_agent(query: str, top_k: int | None = None) -> tuple[list[str], list[str]]:
    """Like retrieve(), but also tags image_caption chunks with their
    source_image_path so the agent can cite it in a follow-up
    describe_image call (Phase 10). Returns (plain_sources, tagged_for_model)."""
    query_vector = Vector(await embed_text(query))
    limit = top_k if top_k is not None else settings.top_k

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                select content, source_type, source_image_path
                from documents
                order by embedding <=> %s
                limit %s
                """,
                (query_vector, limit),
            )
            rows = await cur.fetchall()

    plain = [content for content, _, _ in rows]
    tagged = [
        f"[image_path={image_path}] {content}"
        if source_type == "image_caption" and image_path
        else content
        for content, source_type, image_path in rows
    ]
    return plain, tagged


SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer only using information "
    "returned by the `retrieve` tool - you have no other knowledge of the "
    "documents. For questions with multiple parts, call `retrieve` once per "
    "part (or with refined queries) rather than relying on a single search. "
    "Some retrieved chunks are pre-computed captions of a chart/figure image, "
    "tagged like [image_path=...]. If such a caption doesn't have the detail "
    "you need (e.g. an exact axis value it summarized loosely), call "
    "`describe_image` with that exact image_path for a fresh, deeper look - "
    "don't call it with a path that wasn't given to you. If, after retrying "
    "with different queries or a deeper image look, the context still "
    "doesn't contain the answer, say so honestly instead of guessing."
)


async def _handle_retrieve_call(call: dict, iteration: int, all_sources: list[str]) -> str:
    query = call.get("function", {}).get("arguments", {}).get("query")
    if not isinstance(query, str) or not query.strip():
        logger.warning(
            "iteration %d: skipping malformed retrieve call (no usable query): %r",
            iteration, call,
        )
        return "Error: no valid query argument was provided for this call."

    plain_results, tagged_results = await _retrieve_for_agent(query)
    all_sources.extend(plain_results)
    logger.info(
        "iteration %d: retrieve(%r) -> %d result(s), running total %d",
        iteration, query, len(plain_results), len(all_sources),
    )
    return "\n---\n".join(tagged_results) if tagged_results else NO_RESULTS_MESSAGE


async def _handle_describe_image_call(call: dict, iteration: int) -> str:
    image_path = call.get("function", {}).get("arguments", {}).get("image_path")
    if not isinstance(image_path, str) or not image_path.strip():
        logger.warning(
            "iteration %d: skipping malformed describe_image call (no usable image_path): %r",
            iteration, call,
        )
        return "Error: no valid image_path argument was provided for this call."

    try:
        image_bytes = await get_image_bytes(image_path)
    except Exception:
        logger.warning("iteration %d: describe_image object not found: %r", iteration, image_path)
        return f"Error: no image found at path {image_path!r}."

    caption = await vision_describe_image(image_bytes)
    if caption is None:
        logger.warning("iteration %d: describe_image failed for %r", iteration, image_path)
        return f"Error: could not generate a description for {image_path!r} right now."

    logger.info("iteration %d: describe_image(%r) -> fresh caption generated", iteration, image_path)
    return caption


async def run_agentic_ask(question: str, max_iterations: int | None = None) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    all_sources: list[str] = []
    iterations = max_iterations if max_iterations is not None else settings.max_agent_iterations

    message: dict = {}
    for iteration in range(iterations):
        message = await chat_with_tools(messages, tools=AGENT_TOOLS)
        messages.append(message)

        if message.get("tool_calls"):
            for call in message["tool_calls"]:
                name = call.get("function", {}).get("name")
                if name == "describe_image":
                    tool_content = await _handle_describe_image_call(call, iteration)
                else:
                    tool_content = await _handle_retrieve_call(call, iteration, all_sources)

                messages.append({"role": "tool", "content": tool_content})
        else:
            return {"answer": message["content"], "sources": all_sources}

    # Ran out of iterations. If the last turn was a tool call, its message
    # has no answer content — force one final, tool-less turn so the model
    # summarizes whatever it already retrieved instead of returning empty.
    if message.get("tool_calls"):
        messages.append({
            "role": "user",
            "content": (
                "You're out of retrieval attempts. Answer now using only "
                "what you've already retrieved above."
            ),
        })
        message = await chat_with_tools(messages, allow_tools=False)

    return {"answer": message.get("content", ""), "sources": all_sources}
