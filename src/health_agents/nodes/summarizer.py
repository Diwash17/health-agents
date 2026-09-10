"""The summarizer agent: resolve a report from a path or a paste, then condense it."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from health_agents.io_utils import UnsupportedSource, resolve_source
from health_agents.prompts import SUMMARIZER_SYSTEM, summarize_user_prompt

__all__ = ["make_summarizer"]

PASTE_PROMPT = (
    "Paste the report text, or give a path to a .txt file.\n"
    "  (finish with a line containing only EOF)"
)


def _resolve_or_none(text):
    """resolve_source, but a rejected file becomes a message rather than a crash."""
    try:
        return resolve_source(text), None
    except UnsupportedSource as exc:
        return None, str(exc)


def make_summarizer(llm):
    def summarizer(state):
        text = (state.get("report_text") or "").strip()

        if not text:
            resolved, problem = _resolve_or_none(state.get("query"))
            if problem:
                return {"answer": problem}
            text = (resolved or "").strip()

        if not text:
            # Nothing to work with, so pause and ask. The payload tells the CLI *what*
            # is needed; the CLI decides *how* to collect it.
            pasted = interrupt({"prompt": PASTE_PROMPT, "multiline": True})
            # The paste may itself be a path.
            resolved, problem = _resolve_or_none(pasted)
            if problem:
                return {"answer": problem}
            text = (resolved or str(pasted)).strip()

        if not text:
            return {"answer": "No report text received, so there is nothing to summarize."}

        response = llm.invoke(
            [SystemMessage(SUMMARIZER_SYSTEM), HumanMessage(summarize_user_prompt(text))]
        )
        return {"report_text": text, "answer": response.content}

    return summarizer
