"""LLM access with per-run cost accounting.

Every agent goes through `call_llm` / `call_structured` rather than touching
ChatOpenAI directly, because the spend cap is only enforceable if all traffic
passes one chokepoint. Usage is written back into `state["token_usage"]` keyed by
node, so the dashboard can show where a run's budget actually went.
"""

from __future__ import annotations

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from ..config import get_settings
from ..state.schema import RunState, TokenUsage

T = TypeVar("T", bound=BaseModel)

#: USD per 1M tokens, (input, output). Used for the per-run cap; update alongside
#: OpenAI's pricing page. An unknown model is treated as free rather than
#: guessed at — the cap is a safety net, not billing.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


class CostCapExceeded(RuntimeError):
    """The run hit its LLM spend cap and must stop."""


def _model(name: str, temperature: float) -> ChatOpenAI:
    settings = get_settings()
    # base_url is passed only when set, so the default path stays plain OpenAI.
    # A non-empty value points every call at a compatible free provider instead.
    extra = {"base_url": settings.openai_base_url} if settings.openai_base_url else {}
    return ChatOpenAI(
        model=name,
        temperature=temperature,
        api_key=settings.openai_api_key,
        timeout=90,
        max_retries=2,
        **extra,
    )


def total_cost(state: RunState) -> float:
    """Total USD spent on LLM calls so far in this run."""
    return sum(u.cost_usd for u in state.get("token_usage", {}).values())


def _price(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _PRICING.get(model)
    if not rates:
        return 0.0
    return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000


def _check_cap(state: RunState) -> None:
    cap = get_settings().max_run_cost_usd
    spent = total_cost(state)
    if spent >= cap:
        raise CostCapExceeded(
            f"Run has spent ${spent:.3f} of its ${cap:.2f} LLM budget."
        )


def _record(state: RunState, node: str, model: str, meta: dict) -> None:
    usage = meta.get("token_usage") or meta.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))

    book = state.setdefault("token_usage", {})
    entry = book.setdefault(node, TokenUsage())
    entry.prompt_tokens += prompt_tokens
    entry.completion_tokens += completion_tokens
    entry.cost_usd += _price(model, prompt_tokens, completion_tokens)


def call_llm(
    state: RunState,
    node: str,
    system: str,
    user: str,
    *,
    cheap: bool = False,
    temperature: float = 0.0,
) -> str:
    """Send one prompt and return the text reply, billing it to `node`."""
    _check_cap(state)
    settings = get_settings()
    name = settings.cheap_model if cheap else settings.reasoning_model

    reply = _model(name, temperature).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    _record(state, node, name, reply.response_metadata or {})
    return str(reply.content)


def call_structured(
    state: RunState,
    node: str,
    system: str,
    user: str,
    schema: type[T],
    *,
    cheap: bool = False,
    temperature: float = 0.0,
) -> T:
    """Send one prompt and get back a validated `schema` instance.

    `include_raw` is on so usage metadata survives — the structured-output
    wrapper otherwise hands back only the parsed model and the run's spend
    would silently under-count.
    """
    _check_cap(state)
    settings = get_settings()
    name = settings.cheap_model if cheap else settings.reasoning_model

    bound = _model(name, temperature).with_structured_output(schema, include_raw=True)
    out = bound.invoke([SystemMessage(content=system), HumanMessage(content=user)])

    raw = out.get("raw") if isinstance(out, dict) else None
    if raw is not None:
        _record(state, node, name, getattr(raw, "response_metadata", {}) or {})

    parsed = out.get("parsed") if isinstance(out, dict) else out
    if parsed is None:
        raise ValueError(f"{node}: model did not return valid {schema.__name__}")
    return parsed  # type: ignore[return-value]
