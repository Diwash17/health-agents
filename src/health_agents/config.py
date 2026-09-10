"""Environment loading and per-agent model selection.

Each agent gets its own model, because their jobs make genuinely different demands:

- `router`      one 3-way classification, runs on EVERY turn. Wants small and fast; the
                only hard requirement is tool calling, which structured output uses.
- `summarizer`  reads a dense numeric report without inventing values. The hardest job,
                and the one that most rewards a larger model.
- `report_writer` a formatter. Every fact is supplied by the interview, so it needs
                instruction-following and restraint, not knowledge.
- `qa`          open-ended health questions. Wants breadth and readable prose.

Resolution order per role: `GROQ_<ROLE>_MODEL` -> `GROQ_MODEL` -> `DEFAULT_MODEL`.
So setting nothing works, setting one var changes everything, and setting a role var
changes just that agent.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

__all__ = [
    "MissingApiKey",
    "DEFAULT_MODEL",
    "ROLE_ENV_VARS",
    "TEMPERATURE",
    "model_for",
    "model_map",
    "require_api_key",
    "get_llm",
]

# Groq retires model ids periodically. List what your key can reach with:
#   uv run python -c "import groq,dotenv;dotenv.load_dotenv();\
#       print([m.id for m in groq.Groq().models.list().data])"
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Which env var overrides which agent.
ROLE_ENV_VARS: dict[str, str] = {
    "router": "GROQ_ROUTER_MODEL",
    "summarizer": "GROQ_SUMMARIZER_MODEL",
    "report_writer": "GROQ_REPORT_MODEL",
    "qa": "GROQ_QA_MODEL",
}

# Groq rejects temperature=0 outright; langchain-groq substitutes 1e-8, so this is
# "as deterministic as the provider allows" rather than literally zero.
TEMPERATURE = 0

# Groq's free tier rate-limits readily. langchain-groq defaults to 2 retries with
# exponential backoff; a console app can afford to wait a little longer than that.
MAX_RETRIES = 4
REQUEST_TIMEOUT = 60.0


class MissingApiKey(RuntimeError):
    """Raised when no Groq key is configured, with instructions instead of a traceback."""


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def model_for(role: str) -> str:
    """The model id this agent should use. Raises KeyError for an unknown role."""
    var = ROLE_ENV_VARS[role]
    load_dotenv()
    return _env(var) or _env("GROQ_MODEL") or DEFAULT_MODEL


def model_map() -> dict[str, str]:
    """role -> model id, for every agent. Used by the CLI's `/models` command."""
    return {role: model_for(role) for role in ROLE_ENV_VARS}


def require_api_key() -> str:
    load_dotenv()
    api_key = _env("GROQ_API_KEY")
    if not api_key:
        raise MissingApiKey(
            "GROQ_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key:\n"
            "    cp .env.example .env\n"
            "Get one at https://console.groq.com/keys"
        )
    return api_key


@lru_cache(maxsize=None)
def _client(model: str, api_key: str, temperature: float):
    """One client per distinct model, so roles sharing a model share a connection."""
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
        max_retries=MAX_RETRIES,
        request_timeout=REQUEST_TIMEOUT,
    )


def get_llm(role: str = "qa"):
    """The chat client for `role`, cached by resolved model."""
    return _client(model_for(role), require_api_key(), TEMPERATURE)


@lru_cache(maxsize=1)
def available_models(api_key: str) -> frozenset[str]:
    """Model ids this key can reach. Cached; raises on a network/auth failure."""
    import groq

    return frozenset(m.id for m in groq.Groq(api_key=api_key).models.list().data)


def validate_models() -> tuple[list[str], str | None]:
    """Check every configured model id against what the key can actually reach.

    Returns `(problems, lookup_error)`. A configured id that does not exist is the
    failure this catches: without it, a typo or a model Groq has since retired only
    surfaces when that particular agent first runs — so a bad summarizer model can sit
    unnoticed until someone asks for a summary.

    `lookup_error` is set when the model list itself could not be fetched; that is not a
    config problem and must not block startup.
    """
    try:
        known = available_models(require_api_key())
    except Exception as exc:  # noqa: BLE001 - offline is not a config error
        return [], f"{type(exc).__name__}: {exc}"

    problems = []
    for role, model in model_map().items():
        if model not in known:
            hint = ""
            # Almost always a missing vendor prefix, e.g. gpt-oss-20b vs openai/gpt-oss-20b.
            matches = sorted(k for k in known if k.rsplit("/", 1)[-1] == model)
            if matches:
                hint = f" Did you mean {matches[0]}?"
            problems.append(
                f"{role}: {ROLE_ENV_VARS[role]}={model} is not available to this key.{hint}"
            )
    return problems, None
