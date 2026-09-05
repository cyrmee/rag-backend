import logging

from pgvector import Vector

from app.config import settings
from app.db import get_connection
from app.embeddings import embed_text
from app.generation import chat_with_tools

logger = logging.getLogger(__name__)

NO_RESULTS_MESSAGE = "No results found for this query."


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


SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Answer only using information "
    "returned by the `retrieve` tool - you have no other knowledge of the "
    "documents. For questions with multiple parts, call `retrieve` once per "
    "part (or with refined queries) rather than relying on a single search. "
    "If, after retrying with different queries, the retrieved context still "
    "doesn't contain the answer, say so honestly instead of guessing."
)


async def run_agentic_ask(question: str, max_iterations: int | None = None) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    all_sources: list[str] = []
    iterations = max_iterations if max_iterations is not None else settings.max_agent_iterations

    message: dict = {}
    for iteration in range(iterations):
        message = await chat_with_tools(messages)
        messages.append(message)

        if message.get("tool_calls"):
            for call in message["tool_calls"]:
                query = call.get("function", {}).get("arguments", {}).get("query")
                if not isinstance(query, str) or not query.strip():
                    logger.warning(
                        "iteration %d: skipping malformed tool call (no usable query): %r",
                        iteration, call,
                    )
                    messages.append({
                        "role": "tool",
                        "content": "Error: no valid query argument was provided for this call.",
                    })
                    continue

                results = await retrieve(query)
                all_sources.extend(results)
                logger.info(
                    "iteration %d: retrieve(%r) -> %d result(s), running total %d",
                    iteration, query, len(results), len(all_sources),
                )
                messages.append({
                    "role": "tool",
                    "content": "\n---\n".join(results) if results else NO_RESULTS_MESSAGE,
                })
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
