"""Model resolution: per-agent override -> shared GROQ_MODEL -> built-in default."""

from __future__ import annotations

import pytest

from health_agents import config
from health_agents.config import (
    DEFAULT_MODEL,
    ROLE_ENV_VARS,
    MissingApiKey,
    model_for,
    model_map,
    require_api_key,
)

# Isolation from the developer's .env comes from the autouse `hermetic_env` fixture in
# conftest.py, so these tests only see the vars they set themselves.


def test_every_role_falls_back_to_the_built_in_default():
    assert model_map() == {role: DEFAULT_MODEL for role in ROLE_ENV_VARS}


def test_shared_groq_model_applies_to_every_role(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    assert set(model_map().values()) == {"qwen/qwen3.8-27b"}


def test_a_role_override_wins_over_the_shared_model(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b")

    mapping = model_map()
    assert mapping["router"] == "openai/gpt-oss-20b"
    assert mapping["summarizer"] == "qwen/qwen3.8-27b"
    assert mapping["qa"] == "qwen/qwen3.8-27b"


@pytest.mark.parametrize("role", sorted(ROLE_ENV_VARS))
def test_each_role_can_be_overridden_independently(monkeypatch, role):
    monkeypatch.setenv(ROLE_ENV_VARS[role], "chosen/model")
    mapping = model_map()
    assert mapping[role] == "chosen/model"
    assert all(v == DEFAULT_MODEL for k, v in mapping.items() if k != role)


def test_whitespace_only_override_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("GROQ_QA_MODEL", "   ")
    assert model_map()["qa"] == "qwen/qwen3.8-27b"


def test_values_are_stripped(monkeypatch):
    monkeypatch.setenv("GROQ_ROUTER_MODEL", "  openai/gpt-oss-20b \n")
    assert model_for("router") == "openai/gpt-oss-20b"


def test_unknown_role_is_a_programming_error():
    with pytest.raises(KeyError):
        model_for("translator")


def test_role_env_vars_cover_exactly_the_graph_roles():
    from health_agents.graph import AGENT_ROLES

    assert set(ROLE_ENV_VARS) == set(AGENT_ROLES)


def test_each_agent_is_wired_to_its_own_resolved_model(monkeypatch):
    """The graph must hand each node the client for *its* role, not a shared one."""
    from health_agents.graph import build_graph

    monkeypatch.setenv("GROQ_MODEL", "main/model")
    monkeypatch.setenv("GROQ_ROUTER_MODEL", "tiny/model")

    handed_out = []

    class Marker:
        def __init__(self, model):
            self.model = model

        def with_structured_output(self, schema, **kwargs):
            return self

    def fake_get_llm(role="qa"):
        marker = Marker(model_for(role))
        handed_out.append((role, marker.model))
        return marker

    monkeypatch.setattr(config, "get_llm", fake_get_llm)
    build_graph()

    assert dict(handed_out) == {
        "router": "tiny/model",
        "summarizer": "main/model",
        "report_writer": "main/model",
        "qa": "main/model",
    }


KNOWN = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"})


@pytest.fixture
def known_models(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(config, "available_models", lambda _key: KNOWN)


def test_validate_models_passes_when_every_id_exists(monkeypatch, known_models):
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    assert config.validate_models() == ([], None)


def test_validate_models_flags_an_unknown_id(monkeypatch, known_models):
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("GROQ_SUMMARIZER_MODEL", "llama-3.3-70b-versatile")

    problems, error = config.validate_models()

    assert error is None
    assert len(problems) == 1
    assert "GROQ_SUMMARIZER_MODEL" in problems[0]
    assert "summarizer" in problems[0]


def test_validate_models_suggests_a_missing_vendor_prefix(monkeypatch, known_models):
    """The exact mistake of writing gpt-oss-20b instead of openai/gpt-oss-20b."""
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    monkeypatch.setenv("GROQ_REPORT_MODEL", "gpt-oss-20b")

    problems, _ = config.validate_models()

    assert "Did you mean openai/gpt-oss-20b?" in problems[0]


def test_validate_models_reports_every_bad_role(monkeypatch, known_models):
    monkeypatch.setenv("GROQ_MODEL", "nope/one")
    problems, _ = config.validate_models()
    assert len(problems) == len(ROLE_ENV_VARS)


def test_a_lookup_failure_is_not_treated_as_a_config_problem(monkeypatch):
    """Being offline must not block startup, and must not look like a bad model id."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "anything/at-all")

    def boom(_key):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(config, "available_models", boom)

    problems, error = config.validate_models()
    assert problems == []
    assert "ConnectionError" in error


def test_missing_api_key_explains_how_to_fix_it(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(MissingApiKey, match=r"\.env"):
        require_api_key()


def test_api_key_is_stripped(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "  abc123  ")
    assert require_api_key() == "abc123"
