"""The router: one LLM call, one enum out.

Kept as small a surface as possible on purpose. Open-weight models drift on free-form
output, so the model is asked for a single classification field and nothing else, and any
failure falls back to the Q/A agent — the one route that is always a sane thing to do
with an arbitrary message.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from health_agents.prompts import ROUTER_SYSTEM

__all__ = ["RouteDecision", "make_router"]


class RouteDecision(BaseModel):
    """The router's only output."""

    intent: Literal["summarize", "build_report", "qa"] = Field(
        description="Which agent should handle the message."
    )


def make_router(llm):
    """Build the router node bound to `llm`."""
    classifier = llm.with_structured_output(RouteDecision)

    def router(state):
        query = state.get("query", "")
        try:
            decision = classifier.invoke(
                [SystemMessage(ROUTER_SYSTEM), HumanMessage(query)]
            )
            intent = decision.intent
        except Exception:
            # A router that cannot classify should still answer the user.
            intent = "qa"

        update = {"route": intent}
        # Clear the chosen agent's carried-over state so each request starts clean.
        # Safe to do here: a paused agent sits inside its own node and never re-enters
        # the router, so this only ever runs at the start of a fresh request.
        if intent == "build_report":
            update["report_data"] = {}
        elif intent == "summarize":
            # Without this, "summarize this report" with no path would silently reuse
            # whichever report was summarized earlier in the session.
            update["report_text"] = ""
        return update

    return router
