import pytest

from health_agents.schema import (
    FIELD_KEYS,
    REQUIRED_FIELDS,
    next_missing_field,
    question_for,
)


def test_field_keys_match_required_fields_order():
    assert FIELD_KEYS == tuple(f.key for f in REQUIRED_FIELDS)


def test_empty_data_asks_for_the_first_field():
    assert next_missing_field({}) == FIELD_KEYS[0]


def test_none_is_treated_as_empty():
    assert next_missing_field(None) == FIELD_KEYS[0]


def test_partially_filled_asks_for_the_first_gap_not_the_next_index():
    # Second field answered, first still blank -> must still ask the first.
    data = {FIELD_KEYS[1]: "34"}
    assert next_missing_field(data) == FIELD_KEYS[0]


def test_sequential_fill_walks_the_field_list_in_order():
    data = {}
    for expected in FIELD_KEYS:
        assert next_missing_field(data) == expected
        data[expected] = "answered"
    assert next_missing_field(data) is None


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t "])
def test_whitespace_only_answers_count_as_missing(blank):
    data = {FIELD_KEYS[0]: blank}
    assert next_missing_field(data) == FIELD_KEYS[0]


def test_complete_data_returns_none():
    data = {key: "answered" for key in FIELD_KEYS}
    assert next_missing_field(data) is None


def test_unknown_keys_do_not_satisfy_required_fields():
    assert next_missing_field({"favourite_colour": "blue"}) == FIELD_KEYS[0]


def test_question_for_returns_a_prompt_for_every_field():
    for key in FIELD_KEYS:
        assert question_for(key).strip()


def test_question_for_rejects_unknown_key():
    with pytest.raises(KeyError):
        question_for("not_a_field")
