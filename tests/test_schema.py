"""Unit tests for schema.py — no network, pure Pydantic validation logic."""

import pytest
from pydantic import ValidationError

from schema import ASAP_PROFILE, YELP_PROFILE, FullReviewAnalysis, OperationalFields

VALID_ASAP = {
    "sentiment": "positive",
    "rating_prediction": "4",
    "aspect_sentiments": {"food_taste": "positive", "bogus_aspect": "positive"},
    "problem_type": "overpriced",
    "action_priority": "low",
    "operator_action": "review_pricing",
}


def test_rating_prediction_coerced_from_string():
    r = FullReviewAnalysis.model_validate(VALID_ASAP, context={"profile": ASAP_PROFILE})
    assert r.rating_prediction == 4


def test_rating_prediction_clamped_when_out_of_range():
    data = dict(VALID_ASAP, rating_prediction=99)
    r = FullReviewAnalysis.model_validate(data, context={"profile": ASAP_PROFILE})
    assert r.rating_prediction == 5


def test_rating_prediction_defaults_to_3_when_unparseable():
    data = dict(VALID_ASAP, rating_prediction="high")
    r = FullReviewAnalysis.model_validate(data, context={"profile": ASAP_PROFILE})
    assert r.rating_prediction == 3


def test_unknown_aspect_is_dropped_not_rejected():
    r = FullReviewAnalysis.model_validate(VALID_ASAP, context={"profile": ASAP_PROFILE})
    assert "bogus_aspect" not in r.aspect_sentiments
    assert r.aspect_sentiments == {"food_taste": "positive"}


def test_aspect_sentiments_accepts_list_form():
    data = dict(VALID_ASAP, aspect_sentiments=["food_taste", "service_attitude"])
    r = FullReviewAnalysis.model_validate(data, context={"profile": ASAP_PROFILE})
    assert r.aspect_sentiments == {"food_taste": "neutral", "service_attitude": "neutral"}


def test_aspect_sentiments_accepts_list_of_dicts():
    """Regression: deepseek-v4-flash returns this shape. Previously raised an
    unhandled TypeError ('unhashable type: dict') that escaped Pydantic entirely."""
    data = dict(VALID_ASAP, aspect_sentiments=[
        {"aspect": "food_taste", "sentiment": "positive"},
        {"aspect": "service_attitude", "sentiment": "negative"},
    ])
    r = FullReviewAnalysis.model_validate(data, context={"profile": ASAP_PROFILE})
    assert r.aspect_sentiments == {"food_taste": "positive", "service_attitude": "negative"}


def test_aspect_sentiments_accepts_list_of_single_key_dicts():
    data = dict(VALID_ASAP, aspect_sentiments=[{"food_taste": "positive"}, {"price_level": "negative"}])
    r = FullReviewAnalysis.model_validate(data, context={"profile": ASAP_PROFILE})
    assert r.aspect_sentiments == {"food_taste": "positive", "price_level": "negative"}


@pytest.mark.parametrize("junk", [
    [{"nested": {"deep": "value"}}],          # dict-valued, unhashable if used as key
    [[1, 2], None, 42],                       # lists/None/ints inside the list
    {"food_taste": {"nested": "dict"}},       # non-str value
    {123: "positive"},                        # non-str key
    "not a container at all",
    None,
])
def test_aspect_sentiments_never_raises_on_junk(junk):
    """The coercion layer must degrade to {} rather than crash — bad model output
    should never propagate an exception to the caller."""
    data = dict(VALID_ASAP, aspect_sentiments=junk)
    r = FullReviewAnalysis.model_validate(data, context={"profile": ASAP_PROFILE})
    assert isinstance(r.aspect_sentiments, dict)


@pytest.mark.parametrize("field,bad_value", [
    ("problem_type", "not_a_real_problem"),
    ("action_priority", "urgent"),
    ("operator_action", "call_the_manager"),
    ("sentiment", "furious"),
])
def test_invalid_enum_value_raises(field, bad_value):
    data = dict(VALID_ASAP, **{field: bad_value})
    with pytest.raises(ValidationError):
        FullReviewAnalysis.model_validate(data, context={"profile": ASAP_PROFILE})


def test_operational_fields_only_validates_three_fields():
    op = OperationalFields.model_validate(
        {"problem_type": "poor_service", "action_priority": "high", "operator_action": "train_service"},
        context={"profile": ASAP_PROFILE},
    )
    assert op.problem_type == "poor_service"


def test_yelp_profile_uses_its_own_vocabulary():
    data = {
        "sentiment": "negative",
        "rating_prediction": 1,
        "aspect_sentiments": {"customer_service": "negative"},
        "problem_type": "fake_product",
        "action_priority": "high",
        "operator_action": "verify_authenticity",
    }
    r = FullReviewAnalysis.model_validate(data, context={"profile": YELP_PROFILE})
    assert r.problem_type == "fake_product"
    # An ASAP-only value should be rejected under the Yelp profile.
    with pytest.raises(ValidationError):
        FullReviewAnalysis.model_validate(dict(data, problem_type="hygiene_issue"), context={"profile": YELP_PROFILE})


def test_no_profile_context_skips_enum_checks():
    # Without a profile in context, validators can't check membership — should not raise.
    data = dict(VALID_ASAP, problem_type="anything_goes")
    r = FullReviewAnalysis.model_validate(data)
    assert r.problem_type == "anything_goes"
