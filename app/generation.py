import asyncio
import json
import logging
import re
import uuid
from typing import AsyncIterator

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Chat goes to an OpenAI-compatible server (vLLM), not Ollama - this module
# is the only place that knows that. Everything it returns is normalized
# back to the internal shape the rest of the app was written against:
# {"role", "content", "thinking", "tool_calls"}, with each tool call's
# function.arguments already parsed into a dict.


def _chat_client(limits: httpx.Limits | None = None) -> httpx.AsyncClient:
    headers = {"Authorization": f"Bearer {settings.chat_api_key}"} if settings.chat_api_key else {}
    return httpx.AsyncClient(
        base_url=settings.chat_base_url, timeout=300.0, headers=headers, limits=limits or httpx.Limits(),
    )


async def _raise_for_status(resp: httpx.Response) -> None:
    """raise_for_status(), but logs the server's error body first - vLLM
    explains a 400 there (e.g. "maximum context length is 32768 tokens"),
    and httpx's exception message leaves it out."""
    if resp.is_error:
        await resp.aread()
        logger.error("chat server returned %d: %s", resp.status_code, resp.text[:1000])
    resp.raise_for_status()


# vLLM's /tokenize lives at the server root, not under /v1. If it's
# unavailable (another OpenAI-compatible server), counts fall back to this
# deliberately pessimistic ratio - spreadsheet text measured ~1.7 chars per
# token here, English prose ~4.8, so a fixed ratio has to assume the worst.
FALLBACK_CHARS_PER_TOKEN = 1.5

# count_text_tokens fires one /tokenize per result (often 20-30 at once);
# they're cheap, but there's no reason to open that many sockets to the
# chat server at a time.
TOKENIZE_CONNECTIONS = 8


def _tokenize_url() -> str:
    return settings.chat_base_url.rstrip("/").removesuffix("/v1") + "/tokenize"


async def count_prompt_tokens(messages: list[dict], tools: list[dict] | None) -> tuple[int, int]:
    """Exact prompt size of `messages` (+ tool definitions) as the chat
    server will see it, and the server's own context limit. Returns
    (tokens, max_model_len)."""
    try:
        async with _chat_client() as client:
            resp = await client.post(
                _tokenize_url(),
                json={"model": settings.chat_model, "messages": messages, **({"tools": tools} if tools else {})},
            )
            resp.raise_for_status()
            data = resp.json()
        return data["count"], data.get("max_model_len") or settings.chat_num_ctx
    except (httpx.HTTPError, KeyError, ValueError):
        logger.warning("tokenize unavailable, estimating prompt size", exc_info=True)
        chars = len(json.dumps(messages)) + len(json.dumps(tools or []))
        return int(chars / FALLBACK_CHARS_PER_TOKEN), settings.chat_num_ctx


async def count_text_tokens(texts: list[str]) -> list[int]:
    """Token count of each text on its own (no chat template), in parallel."""
    if not texts:
        return []
    try:
        async with _chat_client(httpx.Limits(max_connections=TOKENIZE_CONNECTIONS)) as client:
            responses = await asyncio.gather(*(
                client.post(_tokenize_url(), json={"model": settings.chat_model, "prompt": t, "add_special_tokens": False})
                for t in texts
            ))
        for resp in responses:
            resp.raise_for_status()
        return [resp.json()["count"] for resp in responses]
    except (httpx.HTTPError, KeyError, ValueError):
        logger.warning("tokenize unavailable, estimating text sizes", exc_info=True)
        return [int(len(t) / FALLBACK_CHARS_PER_TOKEN) + 1 for t in texts]


def _reasoning(msg: dict) -> str:
    """vLLM's reasoning parser puts the model's thinking in a separate
    field - `reasoning_content` in older versions, `reasoning` in newer
    ones. Either maps to what this app calls "thinking"."""
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


def _parse_arguments(raw, name: str | None) -> dict:
    """OpenAI-format tool calls carry arguments as a JSON string; callers
    here expect a dict. Malformed JSON degrades to {} (and a log line) so
    the tool handlers' own "no usable argument" path deals with it instead
    of the whole turn blowing up."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("tool call %r: unparseable arguments %r", name, raw)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("tool call %r: arguments are not a JSON object: %r", name, raw)
        return {}
    return parsed


def _normalize_tool_call(call_id: str | None, name: str | None, raw_arguments) -> dict:
    return {
        # Every tool result has to echo its call's id back as tool_call_id;
        # synthesize one if the server ever omits it so that pairing holds.
        "id": call_id or f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": _parse_arguments(raw_arguments, name)},
    }


def _chat_payload(
    messages: list[dict], tools: list[dict] | None, allow_tools: bool, stream: bool, tool_choice: dict | str | None
) -> dict:
    payload = {"model": settings.chat_model, "messages": messages, "stream": stream}
    # Omitted entirely (not []) to force a plain text answer.
    if allow_tools:
        payload["tools"] = tools if tools is not None else DEFAULT_TOOLS
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    return payload


def force_tool(name: str) -> dict:
    """OpenAI-format tool_choice that makes the model call `name` on this
    turn - it still writes the arguments itself, it just can't skip the
    call or pick a different tool."""
    return {"type": "function", "function": {"name": name}}


async def generate_answer(prompt: str) -> tuple[str, str]:
    """Returns (answer, thinking). Used for short internal jobs (titles,
    document summaries, query decomposition), so the model's thinking
    phase is switched off - it roughly multiplies latency for no gain on
    tasks this simple. `thinking` is still read from the server's reasoning
    field (or an inline <think>...</think> block) in case a model ignores
    that switch, but is normally empty."""
    async with _chat_client() as client:
        resp = await client.post(
            "/chat/completions",
            json={
                "model": settings.chat_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # Qwen3's chat template flag; templates without it ignore it.
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        await _raise_for_status(resp)
        data = resp.json()

    message = data["choices"][0]["message"]
    raw_answer = message.get("content") or ""
    thinking = _reasoning(message)

    answer = _THINK_RE.sub("", raw_answer).strip()
    return answer, thinking


_TITLE_PROMPT = (
    "Summarize this question as a short conversation title: 3-6 words, "
    "plain text, no quotes, no trailing punctuation, no preamble like "
    '"Title:".\n\nQuestion: {question}\n\nTitle:'
)


async def generate_title(question: str) -> str:
    """A short label for a conversation list, generated once from its first
    question. Falls back to a truncated version of the question itself if
    the model wraps its answer in quotes/preamble it ignored, or returns
    something clearly not a short title."""
    raw, _ = await generate_answer(_TITLE_PROMPT.format(question=question))
    title = raw.strip().strip('"').strip("'").strip()
    if not title or len(title) > 80:
        title = question.strip()[:60]
    return title


RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "Search the document database for relevant chunks",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}

DESCRIBE_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "describe_image",
        "description": (
            "Get a fresh, detailed vision-model description of a specific "
            "figure/chart, for a deeper look than the pre-computed caption "
            "already in a retrieved chunk. Only call this with an "
            "image_path that appeared in a retrieved chunk's "
            "[image_path=...] tag - not a made-up path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "The image_path from a retrieved chunk's [image_path=...] tag",
                }
            },
            "required": ["image_path"],
        },
    },
}

LIST_DOCUMENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "list_documents",
        "description": (
            "Lists/counts ingested documents by filename - for questions "
            "about the document corpus itself (e.g. 'how many documents do "
            "we have', 'what documents do we have', 'how many files "
            "mention X in their name/folder'). Do NOT use this for "
            "questions about what's inside a document's content - use "
            "retrieve for that. Returns a total count plus up to 50 "
            "matching filenames with their chunk counts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename_contains": {
                    "type": "string",
                    "description": (
                        "Optional case-insensitive substring to filter filenames/folder "
                        "paths by. Omit to list/count every document."
                    ),
                }
            },
        },
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Searches the public internet - for anything current or "
            "outside the document archive (today's news, current "
            "regulations/exchange rates, a fact the retrieved documents "
            "don't cover). Do NOT use this for questions about the "
            "organization's own documents/policies/configuration - use "
            "retrieve for that, even if a web search might also turn up "
            "something related. Returns a short list of "
            "{title, url, content} results, not full pages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The web search query"}
            },
            "required": ["query"],
        },
    },
}

DEFAULT_TOOLS = [RETRIEVE_TOOL]

async def chat_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    allow_tools: bool = True,
    tool_choice: dict | str | None = None,
) -> dict:
    """Sends `messages` to the chat server's /chat/completions with `tools`
    exposed (defaults to just the retrieve tool). Returns the assistant
    message dict (may contain "tool_calls", arguments parsed to dicts).
    Pass allow_tools=False to force a direct text answer (e.g. when the
    caller's iteration budget is exhausted and it needs a final response),
    or tool_choice=force_tool(name) to force one specific tool call."""
    async with _chat_client() as client:
        resp = await client.post(
            "/chat/completions",
            json=_chat_payload(messages, tools, allow_tools, stream=False, tool_choice=tool_choice),
        )
        await _raise_for_status(resp)
        data = resp.json()

    raw = data["choices"][0]["message"]
    message = {"role": "assistant", "content": raw.get("content") or "", "thinking": _reasoning(raw)}
    if raw.get("tool_calls"):
        message["tool_calls"] = [
            _normalize_tool_call(c.get("id"), c.get("function", {}).get("name"), c.get("function", {}).get("arguments"))
            for c in raw["tool_calls"]
        ]
    return message


async def stream_chat_with_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    allow_tools: bool = True,
    tool_choice: dict | str | None = None,
) -> AsyncIterator[dict]:
    """Streaming counterpart to chat_with_tools. Parses the server's SSE
    stream and yields chunks in the internal shape: {"message": {...},
    "done": bool}, where each message carries incremental "content" and/or
    "thinking" deltas. OpenAI-format streams send tool calls as fragments
    (id and name first, then argument JSON piece by piece, keyed by index);
    those are accumulated here and emitted only once, fully-formed and
    parsed, on the final done chunk - never partially."""
    pending: dict[int, dict] = {}

    async with _chat_client() as client:
        async with client.stream(
            "POST",
            "/chat/completions",
            json=_chat_payload(messages, tools, allow_tools, stream=True, tool_choice=tool_choice),
        ) as resp:
            await _raise_for_status(resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}

                content = delta.get("content") or ""
                thinking = _reasoning(delta)
                if content or thinking:
                    yield {"message": {"role": "assistant", "content": content, "thinking": thinking}, "done": False}

                for fragment in delta.get("tool_calls") or []:
                    acc = pending.setdefault(fragment.get("index", 0), {"id": None, "name": None, "arguments": ""})
                    if fragment.get("id"):
                        acc["id"] = fragment["id"]
                    fn = fragment.get("function") or {}
                    if fn.get("name"):
                        acc["name"] = fn["name"]
                    if fn.get("arguments"):
                        acc["arguments"] += fn["arguments"]

                if choice.get("finish_reason"):
                    break

    final = {"role": "assistant", "content": ""}
    if pending:
        final["tool_calls"] = [
            _normalize_tool_call(acc["id"], acc["name"], acc["arguments"])
            for _, acc in sorted(pending.items())
        ]
    yield {"message": final, "done": True}
