"""End-to-end graph tests: the whole interrupt/resume interview, with no network."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from health_agents.graph import build_graph
from health_agents.schema import FIELD_KEYS


def run(graph, payload, config):
    """Drive the graph until it stops, returning the interrupt payload if it paused."""
    interrupted = None
    for event in graph.stream(payload, config):
        if "__interrupt__" in event:
            interrupted = event["__interrupt__"][0].value
    return interrupted


@pytest.fixture
def graph(fake_llm):
    return build_graph(llm=fake_llm, checkpointer=InMemorySaver())


@pytest.fixture
def config():
    return {"configurable": {"thread_id": "test-thread"}}


def test_interview_asks_every_field_once_in_order(graph, fake_llm, config):
    fake_llm.intent = "build_report"
    fake_llm.responses = ["FINISHED REPORT"]

    asked = []
    payload = {"query": "create a health report for me"}
    for _ in range(len(FIELD_KEYS) + 1):
        interrupted = run(graph, payload, config)
        if interrupted is None:
            break
        asked.append(interrupted["prompt"])
        assert interrupted["multiline"] is False
        payload = Command(resume=f"answer {len(asked)}")
    else:
        pytest.fail("interview did not terminate")

    assert len(asked) == len(FIELD_KEYS)
    # Each question corresponds to its field, in declared order.
    from health_agents.schema import question_for

    assert asked == [question_for(key) for key in FIELD_KEYS]

    values = graph.get_state(config).values
    assert values["answer"] == "FINISHED REPORT"
    assert list(values["report_data"]) == list(FIELD_KEYS)


def test_report_writer_is_called_exactly_once(graph, fake_llm, config):
    fake_llm.intent = "build_report"

    payload = {"query": "make a report"}
    while (interrupted := run(graph, payload, config)) is not None:
        payload = Command(resume="x")

    # One generative call for the whole interview, not one per answered question.
    assert len(fake_llm.calls) == 1


def test_blank_answer_re_asks_the_same_field(graph, fake_llm, config):
    fake_llm.intent = "build_report"

    first = run(graph, {"query": "make a report"}, config)
    again = run(graph, Command(resume="   "), config)
    assert again["prompt"] == first["prompt"]


def test_a_second_report_in_the_same_session_starts_from_scratch(graph, fake_llm, config):
    fake_llm.intent = "build_report"

    payload = {"query": "make a report"}
    while run(graph, payload, config) is not None:
        payload = Command(resume="x")

    # Same thread, so report_data from the first report is still in state.
    second_question = run(graph, {"query": "make another report"}, config)
    assert second_question is not None, "router did not reset report_data"

    from health_agents.schema import question_for

    assert second_question["prompt"] == question_for(FIELD_KEYS[0])


def test_summarizer_with_a_txt_path_never_interrupts(graph, fake_llm, config, tmp_path):
    report = tmp_path / "labs.txt"
    report.write_text("Haemoglobin 9.1 g/dL (low)\n", encoding="utf-8")
    fake_llm.intent = "summarize"
    fake_llm.responses = ["SUMMARY"]

    interrupted = run(graph, {"query": f"summarize {report}"}, config)

    assert interrupted is None
    assert graph.get_state(config).values["answer"] == "SUMMARY"


def test_summarizer_without_a_path_asks_for_a_multiline_paste(graph, fake_llm, config):
    fake_llm.intent = "summarize"
    fake_llm.responses = ["SUMMARY"]

    interrupted = run(graph, {"query": "summarize my report"}, config)
    assert interrupted is not None
    assert interrupted["multiline"] is True

    assert run(graph, Command(resume="Cholesterol 260 mg/dL"), config) is None
    assert graph.get_state(config).values["answer"] == "SUMMARY"


def test_summarizer_reports_an_unsupported_file_instead_of_summarizing_the_path(
    graph, fake_llm, config, tmp_path
):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    fake_llm.intent = "summarize"

    assert run(graph, {"query": f"summarize {pdf}"}, config) is None
    answer = graph.get_state(config).values["answer"]
    assert ".txt" in answer
    assert fake_llm.calls == [], "should not have called the model at all"


def test_qa_remembers_the_previous_turn(graph, fake_llm, config):
    fake_llm.intent = "qa"
    fake_llm.responses = ["Anemia means low haemoglobin.", "It is treated with iron."]

    run(graph, {"query": "what is anemia?"}, config)
    run(graph, {"query": "and how is it treated?"}, config)

    # The second call must have seen the first exchange.
    second_call = fake_llm.calls[1]
    texts = [getattr(m, "content", "") for m in second_call]
    assert "what is anemia?" in texts
    assert "Anemia means low haemoglobin." in texts

    assert len(graph.get_state(config).values["messages"]) == 4


def test_a_new_thread_starts_with_no_memory(graph, fake_llm):
    fake_llm.intent = "qa"

    run(graph, {"query": "first"}, {"configurable": {"thread_id": "a"}})
    fresh = {"configurable": {"thread_id": "b"}}
    run(graph, {"query": "second"}, fresh)

    assert len(graph.get_state(fresh).values["messages"]) == 2
