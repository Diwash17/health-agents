"""The report-builder agent: two nodes, split for a specific reason.

`report_builder` asks for exactly ONE field per execution and makes no LLM call at all.
`report_writer` makes a single prose call once every field is in.

The split is forced by how `interrupt()` resumes: LangGraph re-runs the interrupted node
from its first line, with `interrupt()` returning the resume value instead of halting. So
everything above the `interrupt()` call runs twice per field. Keeping that region pure
(`next_missing_field`) makes the repeat harmless; putting the LLM call in a separate node
means it happens exactly once instead of once per answered question.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt

from health_agents.prompts import REPORT_WRITER_SYSTEM, report_user_prompt
from health_agents.schema import next_missing_field, question_for

__all__ = ["report_builder", "make_report_writer"]


def report_builder(state):
    """Ask for the next missing field and record the answer. No LLM involved."""
    data = dict(state.get("report_data") or {})

    key = next_missing_field(data)  # pure: safe to re-run on resume
    if key is None:
        return {}

    answer = interrupt({"prompt": question_for(key), "multiline": False})
    data[key] = str(answer).strip()
    return {"report_data": data}


def make_report_writer(llm):
    """Build the node that turns collected details into the finished report."""

    def report_writer(state):
        data = state.get("report_data") or {}
        response = llm.invoke(
            [
                SystemMessage(REPORT_WRITER_SYSTEM),
                HumanMessage(report_user_prompt(data)),
            ]
        )
        return {"answer": response.content}

    return report_writer
