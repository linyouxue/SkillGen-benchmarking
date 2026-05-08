"""LLM client wrappers: chat, embeddings, web search, concurrency, and token tracking."""

from __future__ import annotations
import contextvars
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)


# Token usage tracking

_stage_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "skillgen_stage", default="unknown"
)
_token_stats_lock = threading.Lock()
# {(model, stage, call_type): {"calls": int, "prompt": int,
#                              "completion": int, "total": int}}
_token_stats: dict[tuple[str, str, str], dict[str, int]] = {}


@contextmanager
def stage_scope(name: str):
    """Context manager that tags all LLM / embedding calls made inside it with
    the given stage name. Nesting is supported; inner scopes override.

    Works across threads spawned via :func:`run_concurrent` because we snapshot
    the parent context at submit time.
    """
    token = _stage_var.set(name)
    try:
        yield
    finally:
        _stage_var.reset(token)


def _record_usage(resp, *, model: str, call_type: str) -> None:
    """Accumulate the ``usage`` block from an OpenAI-style response.

    Silently ignored when the response lacks usage (e.g., some streaming or
    error paths). We clamp to int to tolerate ``None`` from flaky providers.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    pt = int(getattr(usage, "prompt_tokens", 0) or 0)
    ct = int(getattr(usage, "completion_tokens", 0) or 0)
    tt = int(getattr(usage, "total_tokens", 0) or (pt + ct))
    key = (model, _stage_var.get(), call_type)
    with _token_stats_lock:
        cell = _token_stats.setdefault(
            key, {"calls": 0, "prompt": 0, "completion": 0, "total": 0}
        )
        cell["calls"] += 1
        cell["prompt"] += pt
        cell["completion"] += ct
        cell["total"] += tt


def get_token_stats() -> list[dict]:
    """Flat list of usage rows, safe to JSON-serialize."""
    with _token_stats_lock:
        return [
            {
                "model": model,
                "stage": stage,
                "call_type": call_type,
                "calls": v["calls"],
                "prompt_tokens": v["prompt"],
                "completion_tokens": v["completion"],
                "total_tokens": v["total"],
            }
            for (model, stage, call_type), v in sorted(_token_stats.items())
        ]


def reset_token_stats() -> None:
    with _token_stats_lock:
        _token_stats.clear()


def dump_token_stats(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(get_token_stats(), indent=2))
    return path

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None

_router_client: OpenAI | None = None
_openai_client: OpenAI | None = None
_ROUTER_MAX_RETRIES = 6
_ROUTER_RETRY_BASE_DELAY = 2.0
# Per-request HTTP timeout for OpenRouter. Without this the OpenAI SDK's
# default is effectively unbounded, so a single stuck upstream request can
# freeze an entire eval (as happened on gpt-oss-20b AlfWorld, where a lone
# instance hung for ~50 min at 149/150). 180s is long enough for big
# reasoning models to respond but short enough to recover via retry.
_ROUTER_REQUEST_TIMEOUT = float(os.environ.get("OPENROUTER_HTTP_TIMEOUT", "180"))


def _get_router_client() -> OpenAI:
    """OpenRouter client for LLM chat completions."""
    global _router_client
    if _router_client is None:
        _router_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=_ROUTER_REQUEST_TIMEOUT,
        )
    return _router_client


def _get_openai_client() -> OpenAI:
    """OpenAI client for embeddings and web search."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    return _openai_client


def _router_chat_create(**kwargs):
    """Chat completion with exponential-backoff retries for transient errors."""
    last_exc = None
    for attempt in range(1, _ROUTER_MAX_RETRIES + 1):
        try:
            return _get_router_client().chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            cls_name = exc.__class__.__name__
            is_retryable = (
                isinstance(exc, json.JSONDecodeError)
                or "JSONDecodeError" in cls_name
                or "Expecting value" in msg
                or "429" in msg
                or "502" in msg
                or "503" in msg
                or "504" in msg
                or "RateLimit" in cls_name
                or "rate-limited" in msg.lower()
                or "rate limited" in msg.lower()
                or "timeout" in msg.lower()
                or "timed out" in msg.lower()
                or "temporarily unavailable" in msg.lower()
                or "APITimeout" in cls_name
                or "APIConnection" in cls_name
                or "ReadTimeout" in cls_name
                or "ConnectTimeout" in cls_name
            )
            if not is_retryable or attempt == _ROUTER_MAX_RETRIES:
                raise
            time.sleep(_ROUTER_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    raise last_exc


# Chat (via OpenRouter)

def chat(prompt: str, *, system: str = "", model: str = "openai/gpt-5.4-mini",
         temperature: float = 0.0, max_tokens: int = 4096,
         tools: list[dict] | None = None) -> str | dict:
    """Single-turn chat completion via OpenRouter."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    kwargs = dict(model=model, messages=msgs,
                  temperature=temperature, max_tokens=max_tokens)
    if tools:
        kwargs["tools"] = tools
    resp = _router_chat_create(**kwargs)
    _record_usage(resp, model=model, call_type="chat")
    msg = resp.choices[0].message
    if msg.tool_calls:
        return {"content": msg.content, "tool_calls": [
            {"id": tc.id, "function": tc.function.name,
             "arguments": json.loads(tc.function.arguments)}
            for tc in msg.tool_calls
        ]}
    return msg.content


def _repair_json(raw: str) -> dict:
    """Best-effort repair of truncated or malformed JSON from LLM responses.

    Handles the most common case: the model's output was cut off mid-string
    because max_tokens was too low, leaving an unterminated JSON string.
    Falls back to json_repair if available, otherwise tries to close open
    structures manually.
    """
    # Fast path: try the standard parser first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try json_repair (pip install json-repair) if available
    try:
        import json_repair  # type: ignore
        return json_repair.repair_json(raw, return_objects=True)  # type: ignore
    except ImportError:
        pass

    # Manual fallback: close any open string, then close open braces/brackets
    # This handles the "Unterminated string" case
    s = raw.rstrip()
    # If ends inside a string, close it
    quote_count = s.count('"') - s.count('\\"')
    if quote_count % 2 == 1:
        s += '"'
    # Close open structures by counting unmatched { and [
    opens = []
    in_str = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '"' and (i == 0 or s[i - 1] != '\\'):
            in_str = not in_str
        elif not in_str:
            if c in ('{', '['):
                opens.append(c)
            elif c == '}' and opens and opens[-1] == '{':
                opens.pop()
            elif c == ']' and opens and opens[-1] == '[':
                opens.pop()
        i += 1
    for ch in reversed(opens):
        s += '}' if ch == '{' else ']'

    return json.loads(s)


def chat_json(prompt: str, *, system: str = "", model: str = "openai/gpt-5.4-mini",
              temperature: float = 0.0, max_tokens: int = 4096) -> dict:
    """Chat completion that parses the response as JSON, via OpenRouter.

    Automatically retries with doubled max_tokens if the response is truncated
    (JSONDecodeError due to unterminated string), up to _JSON_MAX_TOKENS_CAP.
    """
    _JSON_MAX_TOKENS_CAP = 32768
    current_max = max_tokens
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})

    last_exc: Exception | None = None
    while True:
        resp = _router_chat_create(
            model=model, messages=msgs,
            temperature=temperature, max_tokens=current_max,
            response_format={"type": "json_object"},
        )
        _record_usage(resp, model=model, call_type="chat_json")
        raw = resp.choices[0].message.content or ""
        finish_reason = resp.choices[0].finish_reason

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            last_exc = exc
            # If the model was cut off mid-response, retry with more tokens
            if finish_reason == "length" and current_max < _JSON_MAX_TOKENS_CAP:
                current_max = min(current_max * 2, _JSON_MAX_TOKENS_CAP)
                continue
            # Otherwise try to repair the JSON in-place
            try:
                return _repair_json(raw)
            except Exception:
                raise last_exc


def chat_multi_turn(messages: list[dict], *, model: str = "openai/gpt-5.4-mini",
                    temperature: float = 0.0, max_tokens: int = 4096,
                    tools: list[dict] | None = None):
    """Multi-turn chat completion via OpenRouter; retries on empty-choices responses."""
    kwargs = dict(model=model, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if tools:
        kwargs["tools"] = tools

    last_resp = None
    for attempt in range(3):
        resp = _router_chat_create(**kwargs)
        last_resp = resp
        _record_usage(resp, model=model, call_type="chat_multi_turn")
        choices = getattr(resp, "choices", None)
        if choices:
            msg = choices[0].message
            if msg is not None:
                return msg
        time.sleep(0.5 * (attempt + 1))

    logger.warning(
        "chat_multi_turn: %s returned empty choices 3 times; returning empty message.",
        model,
    )
    class _EmptyMsg:
        role = "assistant"
        content = ""
        tool_calls = None
    return _EmptyMsg()


# Embedding (via OpenAI directly - OpenRouter doesn't serve embeddings)

_EMBED_BATCH_SIZE = 100
_EMBED_MAX_CHARS = 2000


def embed(texts: list[str], *, model: str = "text-embedding-3-small") -> list[list[float]]:
    """Return embeddings via OpenAI API, batching to stay within API limits."""
    if not texts:
        return []

    truncated = [t[:_EMBED_MAX_CHARS] for t in texts]

    results: list[list[float]] = []
    for i in range(0, len(truncated), _EMBED_BATCH_SIZE):
        batch = truncated[i : i + _EMBED_BATCH_SIZE]
        resp = _get_openai_client().embeddings.create(model=model, input=batch)
        _record_usage(resp, model=model, call_type="embed")
        results.extend(d.embedding for d in sorted(resp.data, key=lambda x: x.index))
    return results


# Web Search (via OpenAI Responses API)

def web_search(query: str, *, model: str = "gpt-5.4-mini") -> str:
    """Use OpenAI's built-in web search tool. Always uses an OpenAI model ID."""
    # Strip OpenRouter prefix if present (e.g. "openai/gpt-4o" -> "gpt-4o")
    openai_model = model.split("/")[-1] if "/" in model else model
    resp = _get_openai_client().responses.create(
        model=openai_model,
        tools=[{"type": "web_search_preview"}],
        input=query,
    )
    return resp.output_text


# Concurrency

def run_concurrent(
    fn,
    args_list: list,
    *,
    max_workers: int = 16,
    progress_desc: str | None = None,
) -> list:
    """Run fn(*args) concurrently. Returns results in order."""
    if not args_list:
        return []

    results = [None] * len(args_list)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Snapshot parent context per-task so the active `stage_scope` (and any
        # other ContextVars) propagate into each worker thread.
        future_to_idx = {}
        for idx, args in enumerate(args_list):
            ctx = contextvars.copy_context()
            fut = pool.submit(ctx.run, fn, *args)
            future_to_idx[fut] = idx
        progress = None
        if progress_desc and tqdm is not None:
            progress = tqdm(
                total=len(args_list),
                desc=progress_desc,
                leave=False,
                dynamic_ncols=True,
            )
        try:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                if progress is not None:
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()
    return results
