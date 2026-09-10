"""A scripted stand-in for a chat model, so the whole graph runs with no network."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from health_agents import config
from health_agents.nodes.router import RouteDecision


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Isolate every test from the developer's own .env.

    Without this, tests that touch model resolution pass or fail depending on which
    GROQ_* vars happen to be set locally — which is exactly how a test that had been
    green started failing after an unrelated .env edit.
    """
    for name in ("GROQ_MODEL", *config.ROLE_ENV_VARS.values()):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)


class _StructuredStub:
    """Stands in for `llm.with_structured_output(RouteDecision)`."""

    def __init__(self, model: "FakeChatModel"):
        self._model = model

    def invoke(self, messages, **kwargs):
        self._model.structured_calls.append(messages)
        if self._model.router_error is not None:
            raise self._model.router_error
        return RouteDecision(intent=self._model.intent)


class FakeChatModel(BaseChatModel):
    """Returns canned text and records what it was asked.

    `responses` is consumed in order; once exhausted, `fallback` is returned forever, so
    a test only scripts the replies it actually asserts on.
    """

    responses: list[str] = []
    fallback: str = "(fake response)"
    intent: str = "qa"
    router_error: Any = None
    calls: list = []
    structured_calls: list = []

    def __init__(self, **data):
        super().__init__(**data)
        # Fresh per instance; class-level defaults above are only for the schema.
        self.calls = []
        self.structured_calls = []
        self.responses = list(data.get("responses") or [])

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(messages)
        text = self.responses.pop(0) if self.responses else self.fallback
        return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])

    def with_structured_output(self, schema, **kwargs):
        return _StructuredStub(self)


@pytest.fixture
def fake_llm():
    return FakeChatModel()
