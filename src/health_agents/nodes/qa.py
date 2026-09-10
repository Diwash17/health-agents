"""The Q/A agent: the only node that reads and writes conversation memory."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from health_agents.prompts import QA_SYSTEM

__all__ = ["make_qa", "HISTORY_TURNS"]

# How many prior messages to send back. Keeps prompts bounded in a long session while
# still resolving follow-ups like "and how is it treated?".
HISTORY_TURNS = 10


def make_qa(llm):
    def qa(state):
        question = HumanMessage(state.get("query", ""))
        history = list(state.get("messages") or [])[-HISTORY_TURNS:]

        response = llm.invoke([SystemMessage(QA_SYSTEM), *history, question])
        answer = response.content

        # `messages` uses the add_messages reducer, so returning both appends both and
        # the next turn sees this exchange as history.
        return {"messages": [question, AIMessage(answer)], "answer": answer}

    return qa
