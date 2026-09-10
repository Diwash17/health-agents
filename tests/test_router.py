from health_agents.nodes.router import make_router


def test_router_returns_the_models_intent(fake_llm):
    fake_llm.intent = "summarize"
    assert make_router(fake_llm)({"query": "explain my labs"})["route"] == "summarize"


def test_router_clears_report_data_when_starting_a_report(fake_llm):
    fake_llm.intent = "build_report"
    update = make_router(fake_llm)({"query": "make me a report", "report_data": {"age": "34"}})
    assert update["route"] == "build_report"
    assert update["report_data"] == {}


def test_router_clears_stale_report_text_when_summarizing(fake_llm):
    fake_llm.intent = "summarize"
    update = make_router(fake_llm)({"query": "summarize this", "report_text": "old report"})
    assert update["report_text"] == ""


def test_router_leaves_other_state_alone_for_qa(fake_llm):
    fake_llm.intent = "qa"
    update = make_router(fake_llm)({"query": "hello"})
    assert update == {"route": "qa"}


def test_router_falls_back_to_qa_when_the_model_fails(fake_llm):
    fake_llm.router_error = RuntimeError("groq is down")
    assert make_router(fake_llm)({"query": "anything"})["route"] == "qa"
