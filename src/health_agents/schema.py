"""The report-builder's field list and the pure logic that drives the interview.

`next_missing_field` is deliberately the only thing deciding what happens next: the
`report_builder` node calls it to pick a question, and the graph's loop edge calls it to
decide whether to ask again or hand off to `report_writer`. One pure function, so the
interview provably terminates and can be tested without an LLM or a graph.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Field", "REQUIRED_FIELDS", "FIELD_KEYS", "next_missing_field", "question_for"]


@dataclass(frozen=True)
class Field:
    key: str
    question: str
    hint: str


# Order matters: this is the order the user is asked.
REQUIRED_FIELDS: tuple[Field, ...] = (
    Field("patient_name", "Patient's full name?", "e.g. Asha Sharma"),
    Field("age", "Age?", "in years, e.g. 34"),
    Field("sex", "Sex?", "e.g. female / male / other"),
    Field("chief_complaint", "What is the main complaint?", "e.g. persistent headache"),
    Field("symptom_duration", "How long have the symptoms lasted?", "e.g. about 3 weeks"),
    Field("existing_conditions", "Any existing medical conditions?", "'none' is fine"),
    Field("current_medications", "Any medications currently taken?", "'none' is fine"),
    Field("allergies", "Any known allergies?", "'none' is fine"),
)

FIELD_KEYS: tuple[str, ...] = tuple(field.key for field in REQUIRED_FIELDS)

_BY_KEY: dict[str, Field] = {field.key: field for field in REQUIRED_FIELDS}


def next_missing_field(data: dict[str, str] | None) -> str | None:
    """First required field with no usable answer yet, or None when complete.

    A whitespace-only answer counts as missing, so hitting Enter re-asks rather than
    silently leaving a hole in the report.
    """
    data = data or {}
    for key in FIELD_KEYS:
        if not str(data.get(key) or "").strip():
            return key
    return None


def question_for(key: str) -> str:
    """The prompt shown for a field, including its hint. Raises KeyError if unknown."""
    field = _BY_KEY[key]
    return f"{field.question} ({field.hint})"
