"""Wiring the five nodes together.

    START -> router ->  summarizer     -> END
                    ->  qa             -> END
                    ->  report_builder -> report_builder (loop, one field per pass)
                                       -> report_writer  -> END
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from health_agents.nodes.qa import make_qa
from health_agents.nodes.report_builder import make_report_writer, report_builder
from health_agents.nodes.router import make_router
from health_agents.nodes.summarizer import make_summarizer
from health_agents.schema import next_missing_field
from health_agents.state import AppState

__all__ = ["build_graph", "AGENT_ROLES"]

# The roles that get their own model. Kept here because the graph is what decides how
# many distinct LLM-using agents exist; config.ROLE_ENV_VARS must cover exactly these
# (there is a test asserting they stay in step).
AGENT_ROLES = ("router", "summarizer", "report_writer", "qa")


def _pick_agent(state):
    return state.get("route", "qa")


def _interview_done(state):
    """Loop back for the next field, or hand off to the writer when complete."""
    if next_missing_field(state.get("report_data")) is None:
        return "report_writer"
    return "report_builder"


def build_graph(llm=None, checkpointer=None):
    """Compile the graph.

    Pass `llm` to use one model for every agent — that is how the tests inject a fake and
    run the whole flow offline. Leave it as None and each agent gets its own model,
    resolved from the environment per `config.ROLE_ENV_VARS`.

    A checkpointer is mandatory, not optional: `interrupt()` saves state under a
    thread_id, and `Command(resume=...)` raises without one.
    """
    if llm is not None:
        def for_role(_role):
            return llm
    else:
        from health_agents.config import get_llm

        for_role = get_llm

    builder = StateGraph(AppState)
    builder.add_node("router", make_router(for_role("router")))
    builder.add_node("summarizer", make_summarizer(for_role("summarizer")))
    builder.add_node("report_builder", report_builder)
    builder.add_node("report_writer", make_report_writer(for_role("report_writer")))
    builder.add_node("qa", make_qa(for_role("qa")))

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        _pick_agent,
        {
            "summarize": "summarizer",
            "build_report": "report_builder",
            "qa": "qa",
        },
    )
    builder.add_conditional_edges(
        "report_builder",
        _interview_done,
        ["report_builder", "report_writer"],
    )
    for terminal in ("summarizer", "report_writer", "qa"):
        builder.add_edge(terminal, END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
