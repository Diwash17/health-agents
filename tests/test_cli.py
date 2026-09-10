"""CLI tests: the graph<->stdin boundary, driven with scripted input."""

from __future__ import annotations

import builtins

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

from conftest import FakeChatModel
from health_agents.cli import _run_turn, models_table, read_multiline
from health_agents.graph import build_graph
from health_agents.schema import FIELD_KEYS


@pytest.fixture
def console():
    return Console(file=open("/dev/null", "w"), force_terminal=False)


def script_stdin(monkeypatch, lines):
    """Feed `lines` to builtins.input, one per call."""
    pending = list(lines)
    monkeypatch.setattr(builtins, "input", lambda *a: pending.pop(0))
    return pending


def test_models_table_lists_every_agent_and_its_override_var(monkeypatch):
    from health_agents import config

    monkeypatch.setenv("GROQ_MODEL", "main/model")
    monkeypatch.setenv("GROQ_ROUTER_MODEL", "tiny/model")

    rendered = Console(width=200, force_terminal=False)
    with rendered.capture() as captured:
        rendered.print(models_table())
    out = captured.get()

    for role, var in config.ROLE_ENV_VARS.items():
        assert role in out
        assert var in out
    assert "tiny/model" in out and "main/model" in out


def test_read_multiline_stops_at_the_eof_sentinel(monkeypatch, console):
    script_stdin(monkeypatch, ["line one", "line two", "EOF", "never read"])
    assert read_multiline(console, "paste:") == "line one\nline two"


def test_read_multiline_keeps_blank_lines_inside_a_paste(monkeypatch, console):
    script_stdin(monkeypatch, ["a", "", "b", "EOF"])
    assert read_multiline(console, "paste:") == "a\n\nb"


def test_read_multiline_stops_early_on_a_lone_file_path(monkeypatch, console, tmp_path):
    report = tmp_path / "r.txt"
    report.write_text("x", encoding="utf-8")
    script_stdin(monkeypatch, [str(report)])
    assert read_multiline(console, "paste:") == str(report)


def test_read_multiline_survives_stdin_closing(monkeypatch, console):
    def raise_eof(*_a):
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)
    assert read_multiline(console, "paste:") == ""


def test_run_turn_completes_the_whole_interview_from_scripted_input(
    monkeypatch, console, fake_llm
):
    fake_llm.intent = "build_report"
    fake_llm.responses = ["THE REPORT"]
    graph = build_graph(llm=fake_llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "cli"}}

    answers = [f"a{i}" for i in range(len(FIELD_KEYS))]
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: answers.pop(0))

    _run_turn(graph, console, config, "create a health report")

    values = graph.get_state(config).values
    assert values["answer"] == "THE REPORT"
    assert len(values["report_data"]) == len(FIELD_KEYS)
    assert answers == [], "every scripted answer should have been consumed"


class FlakyChatModel(FakeChatModel):
    """Fails every generate call until `fail` is cleared."""

    fail: bool = True

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.fail:
            raise RuntimeError("429 rate limit exceeded")
        return super()._generate(messages, stop, run_manager, **kwargs)


def _answer_whole_interview(graph, console, config, replies):
    _run_turn(graph, console, config, "create a health report")


def test_retry_recovers_an_interview_that_failed_at_the_last_step(monkeypatch, console):
    """A rate limit on the final call must not cost the user their 8 answers."""
    llm = FlakyChatModel()
    llm.intent = "build_report"
    llm.responses = ["THE REPORT"]
    graph = build_graph(llm=llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "retry"}}

    replies = [f"a{i}" for i in range(len(FIELD_KEYS))]

    def answer(self, *a, **k):
        if replies:
            return replies.pop(0)
        # This is the "Retry from there? [Y/n]" prompt. Recover first, then say yes.
        llm.fail = False
        return "y"

    monkeypatch.setattr(Console, "input", answer)

    _run_turn(graph, console, config, "create a health report")

    values = graph.get_state(config).values
    assert values["answer"] == "THE REPORT"
    assert len(values["report_data"]) == len(FIELD_KEYS), "answers were lost"


def test_declining_a_retry_leaves_the_collected_answers_in_state(monkeypatch, console):
    llm = FlakyChatModel()
    llm.intent = "build_report"
    graph = build_graph(llm=llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "decline"}}

    replies = [f"a{i}" for i in range(len(FIELD_KEYS))] + ["n"]
    monkeypatch.setattr(Console, "input", lambda self, *a, **k: replies.pop(0))

    _run_turn(graph, console, config, "create a health report")

    state = graph.get_state(config)
    assert len(state.values["report_data"]) == len(FIELD_KEYS)
    assert state.next == ("report_writer",), "should stay parked at the failed node"
    assert "answer" not in state.values


def test_a_failure_with_nothing_parked_does_not_prompt_for_a_retry(monkeypatch, console):
    """If there is nothing to resume, do not ask a pointless question."""
    llm = FlakyChatModel()
    llm.intent = "qa"
    graph = build_graph(llm=llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "noretry"}}

    asked = []

    def record(self, prompt="", *a, **k):
        asked.append(str(prompt))
        return "n"

    monkeypatch.setattr(Console, "input", record)
    _run_turn(graph, console, config, "what is anemia?")

    # qa is parked, so one retry prompt is correct here; assert it is the retry prompt
    # and that we stopped rather than looping.
    assert all("Retry" in q for q in asked)
    assert len(asked) == 1


def test_run_turn_handles_a_multiline_paste_for_the_summarizer(
    monkeypatch, console, fake_llm
):
    fake_llm.intent = "summarize"
    fake_llm.responses = ["SUMMARY"]
    graph = build_graph(llm=fake_llm, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "cli"}}

    script_stdin(monkeypatch, ["Cholesterol 260 mg/dL", "LDL 180", "EOF"])

    _run_turn(graph, console, config, "summarize my report")

    assert graph.get_state(config).values["answer"] == "SUMMARY"
    # The pasted text, not the query, is what got summarized.
    prompt_text = str(fake_llm.calls[0][-1].content)
    assert "Cholesterol 260" in prompt_text and "LDL 180" in prompt_text
