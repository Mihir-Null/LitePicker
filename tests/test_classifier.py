from classifier import classify, TIER_TO_MODEL, SIMPLE_BOUNDARY, MEDIUM_BOUNDARY, COMPLEX_BOUNDARY
from config import settings


def test_simple_factual():
    """Short factual query should route to the cheapest tier."""
    messages = [{"role": "user", "content": "what time is it"}]
    tier, model_alias, confidence = classify(messages)
    assert tier == "simple"


def test_medium_request():
    """Moderate coding request should land in medium tier with confidence [0.30, 0.50)."""
    messages = [{"role": "user", "content": "debug and refactor this async function to use dependency injection, returning a structured JSON response"}]
    tier, model_alias, confidence = classify(messages)
    assert tier == "medium"
    assert SIMPLE_BOUNDARY <= confidence < MEDIUM_BOUNDARY, (
        f"Expected confidence in [{SIMPLE_BOUNDARY}, {MEDIUM_BOUNDARY}), got {confidence:.4f}"
    )


def test_reasoning_hard_override():
    """2+ reasoning markers always force reasoning tier at confidence 0.97."""
    messages = [{"role": "user", "content": "prove by induction that this theorem holds using contradiction"}]
    tier, model_alias, confidence = classify(messages)
    assert tier == "reasoning"
    assert confidence == 0.97


def test_empty_messages():
    """Empty message list returns the configured default tier."""
    tier, model_alias, confidence = classify([])
    assert tier == settings.default_tier


def test_tool_use_not_simple():
    """Requests with tools present should not route to the simple tier."""
    messages = [{"role": "user", "content": "debug and refactor this async function to use dependency injection, returning a structured JSON response"}]
    tools = [{"type": "function", "function": {"name": "foo", "parameters": {}}}]
    tier, model_alias, confidence = classify(messages, tools)
    assert tier != "simple"


def test_bias_correction():
    """Messages scoring in the 'complex' confidence range [0.50, 0.70) are demoted to medium.

    The 'complex' tier is intentionally suppressed in v1. nanobot's large system prompt
    inflates scores. The bias correction threshold (0.80) is always true when tier == 'complex'
    (which only fires for confidence < 0.70 < 0.80), making 'complex' unreachable in v1.
    """
    messages = [{"role": "user", "content": "prove that implementing a neural network algorithm is efficient"}]
    tier, model_alias, confidence = classify(messages)
    # First: confirm the message actually scored in the 'complex' range
    assert MEDIUM_BOUNDARY <= confidence < COMPLEX_BOUNDARY, (
        f"Test input miscalibrated — expected confidence in [{MEDIUM_BOUNDARY}, {COMPLEX_BOUNDARY}), "
        f"got {confidence:.4f}. Adjust the message content to score in this range."
    )
    # Then: confirm bias correction demoted it to medium
    assert tier == "medium"


def test_returns_valid_model_alias():
    """Every returned model_alias must be a value in TIER_TO_MODEL."""
    messages = [{"role": "user", "content": "hello world"}]
    tier, model_alias, confidence = classify(messages)
    assert model_alias in TIER_TO_MODEL.values()


def test_confidence_bounds():
    """Confidence must always be in [0.0, 1.0] regardless of input."""
    inputs = [
        [{"role": "user", "content": "what is 2+2"}],
        [{"role": "user", "content": "refactor this function"}],
        [{"role": "user", "content": "prove by induction theorem contradiction"}],
        [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "help me write code"},
        ],
        [],
    ]
    for messages in inputs:
        tier, model_alias, confidence = classify(messages)
        assert 0.0 <= confidence <= 1.0, (
            f"confidence {confidence:.4f} out of bounds for input: {messages}"
        )
