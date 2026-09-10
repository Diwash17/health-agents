"""The single state object every node reads and writes."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

__all__ = ["AppState", "Route"]

Route = Literal["summarize", "build_report", "qa"]


class AppState(TypedDict, total=False):
    """Shared graph state.

    `total=False` because each turn only supplies `query`; every other key is filled in
    by whichever node runs. Only `messages` accumulates (via the `add_messages` reducer)
    — the rest are plain overwrites.
    """

    # The user's current turn, set by the CLI.
    query: str
    # Which agent the router chose.
    route: Route
    # Q/A conversation memory. Deliberately not touched by the other agents, so the
    # chat history stays a clean dialogue rather than a log of every node.
    messages: Annotated[list[AnyMessage], add_messages]
    # Source text for the summarizer, once resolved from a path or a paste.
    report_text: str
    # Interview answers collected so far, keyed by `schema.FIELD_KEYS`.
    report_data: dict[str, str]
    # What the CLI prints at the end of a turn.
    answer: str
