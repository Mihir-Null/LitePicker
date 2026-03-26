import math
import re

from config import settings

# ── Tier → LiteLLM model alias ─────────────────────────────────────────────
# Must match model_list aliases in your LiteLLM config exactly.
# heartbeat and local tiers exist here for completeness; v1 classify() never
# returns them — reserved for future heartbeat detection and explicit local routing.
TIER_TO_MODEL: dict[str, str] = {
    "heartbeat": "ollama-default",
    "simple":    "deepseek-local",
    "medium":    "haiku-4-5",
    "complex":   "sonnet-4-6",
    "reasoning": "sonnet-reasoning",
    "local":     "r1-local",
}

# ── Dimension weights ───────────────────────────────────────────────────────
# Ported from ClawRouter rules.ts. Weights sum to 0.95 (factual_lookup is -0.05).
# These dicts are intentionally mutable — a contextual bandit can update them
# at runtime once outcome data accumulates.
DIMENSION_WEIGHTS: dict[str, float] = {
    "vocabulary_complexity":   0.20,
    "reasoning_depth":         0.25,
    "code_complexity":         0.15,
    "domain_specificity":      0.10,
    "instruction_complexity":  0.10,
    "context_dependency":      0.05,
    "output_format":           0.05,
    "ambiguity":               0.03,
    "multi_step":              0.03,
    "creative_requirement":    0.01,
    "factual_lookup":         -0.05,  # negative: pulls toward cheaper tier
    "message_length":          0.01,
    "tool_use_required":       0.01,
    "language_mix":            0.01,
}

# ── Tier boundaries ─────────────────────────────────────────────────────────
SIMPLE_BOUNDARY: float = 0.30   # confidence < this → simple
MEDIUM_BOUNDARY: float = 0.50   # confidence < this → medium
COMPLEX_BOUNDARY: float = 0.70  # confidence < this → complex; >= this → reasoning
REASONING_MARKERS_THRESHOLD: int = 2

# ── Sigmoid parameters ──────────────────────────────────────────────────────
SIGMOID_K: float = 8.0
SIGMOID_MIDPOINT: float = 0.45

# ── Keyword sets ─────────────────────────────────────────────────────────────
# Matching is substring-based and case-insensitive.
# IMPORTANT: verify these against /tmp/clawrouter/src/router/rules.ts before
# production — the keyword lists may have been updated upstream.
REASONING_KEYWORDS: set[str] = {
    "prove", "proof", "theorem", "lemma", "derive", "formal verification",
    "type theory", "induction", "contradiction", "axiom", "corollary",
    "if and only if", "necessary and sufficient", "qed",
}
CODE_KEYWORDS: set[str] = {
    "function", "class", "async", "await", "import", "export", "def ",
    "return", "const ", "let ", "var ", "interface", "struct", "trait",
    "implement", "refactor", "debug", "optimize", "algorithm", "complexity",
    "big o", "recursion", "dependency injection", "design pattern",
}
DOMAIN_KEYWORDS: set[str] = {
    "quantum", "thermodynamics", "stochastic", "differential equation",
    "machine learning", "neural network", "transformer", "attention mechanism",
    "distributed system", "consensus", "byzantine", "cryptography", "zk proof",
    "pharmacology", "jurisprudence", "epistemology", "ontology",
}
FACTUAL_KEYWORDS: set[str] = {
    "what is", "who is", "when was", "where is", "how many", "what year",
    "capital of", "population of", "translate", "definition of",
    "convert", "time in", "weather in",
}
MULTI_STEP_KEYWORDS: set[str] = {
    "first", "then", "next", "finally", "step by step", "step-by-step",
    "walk me through", "sequence", "pipeline", "workflow", "plan",
}

# ── Internal helpers ─────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-SIGMOID_K * (x - SIGMOID_MIDPOINT)))


def _keyword_score(text: str, keywords: set[str]) -> float:
    """Fraction of keywords present in text, capped at 1.0. Case-insensitive."""
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return min(hits / max(len(keywords) * 0.1, 1.0), 1.0)


def _avg_word_length(text: str) -> float:
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0.0


def _score_dimensions(text: str, is_user_turn: bool) -> dict[str, float]:
    """Score one message across all 14 dimensions. All values in [0.0, 1.0].

    reasoning_depth is 0.0 for non-user turns — system prompt boilerplate
    (e.g. 'think step by step') must not inflate this score.
    All regex patterns use re.IGNORECASE.
    """
    word_count = len(text.split())
    return {
        "vocabulary_complexity":  min(_avg_word_length(text) / 10.0, 1.0),
        "reasoning_depth":        _keyword_score(text, REASONING_KEYWORDS) if is_user_turn else 0.0,
        "code_complexity":        _keyword_score(text, CODE_KEYWORDS),
        "domain_specificity":     _keyword_score(text, DOMAIN_KEYWORDS),
        "instruction_complexity": min(text.count(",") / 10.0, 1.0),
        "context_dependency":     1.0 if re.search(r"\b(it|this|that|they|them|those)\b", text, re.I) else 0.0,
        "output_format":          1.0 if re.search(r"\b(json|xml|table|markdown|format|structure)\b", text, re.I) else 0.0,
        "ambiguity":              1.0 if "?" in text and word_count < 10 else 0.0,
        "multi_step":             _keyword_score(text, MULTI_STEP_KEYWORDS),
        "creative_requirement":   1.0 if re.search(r"\b(write|compose|create|generate|imagine|story|poem)\b", text, re.I) else 0.0,
        "factual_lookup":         _keyword_score(text, FACTUAL_KEYWORDS),
        "message_length":         min(word_count / 500.0, 1.0),
        "tool_use_required":      0.0,   # set externally in classify() when tools are present
        "language_mix":           0.0,   # placeholder for multilingual support
    }


# ── Public API ───────────────────────────────────────────────────────────────

def classify(messages: list[dict], tools: list | None = None) -> tuple[str, str, float]:
    """
    Classify a message list into a routing tier.

    Args:
        messages: OpenAI-format message list.
        tools:    Top-level 'tools' array from the request body, if present.
                  Passed by main.py via body.get("tools") — NOT a per-message field.

    Returns:
        (tier, model_alias, confidence)
        tier        — "simple" | "medium" | "reasoning" in v1 ("complex" is suppressed)
        model_alias — LiteLLM model_list alias string
        confidence  — sigmoid-calibrated score in [0.0, 1.0]

    Note on 'complex' tier: the bias correction threshold (confidence < 0.80) is
    intentionally always-true for the 'complex' confidence range (0.50–0.70), making
    'complex' unreachable in v1. Lower the threshold to ~0.65 once bandit outcome
    data confirms the classifier's complex-tier accuracy.
    """
    recent = messages[-3:]
    if not recent:
        tier = settings.default_tier
        return tier, TIER_TO_MODEL[tier], 0.5

    combined: dict[str, float] = {dim: 0.0 for dim in DIMENSION_WEIGHTS}
    reasoning_marker_count = 0
    total_weight = 0.0

    for i, msg in enumerate(recent):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # OpenAI content-blocks format: [{"type": "text", "text": "..."}]
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        if not content:
            continue

        # Most-recent message gets 2× weight; earlier messages get 1×
        msg_weight = 2.0 if i == len(recent) - 1 else 1.0
        total_weight += msg_weight

        scores = _score_dimensions(content, is_user_turn=(role == "user"))
        for dim, score in scores.items():
            combined[dim] += score * msg_weight

        if role == "user":
            content_lower = content.lower()
            # Per-message count, accumulated across turns. Same keyword in two
            # different messages counts twice. No cross-message deduplication.
            reasoning_marker_count += sum(
                1 for kw in REASONING_KEYWORDS if kw in content_lower
            )

    # tools is a top-level request body field. Overwrite (not add) — tool_use_required
    # is always 0.0 after the loop above, so the overwrite is safe.
    # After normalisation this becomes 1.0, contributing its weight to raw_score.
    if tools:
        combined["tool_use_required"] = total_weight

    if total_weight == 0:
        tier = settings.default_tier
        return tier, TIER_TO_MODEL[tier], 0.5

    normalised = {k: v / total_weight for k, v in combined.items()}
    raw_score = sum(normalised[dim] * DIMENSION_WEIGHTS[dim] for dim in DIMENSION_WEIGHTS)
    confidence = _sigmoid(raw_score)

    # Hard override: 2+ reasoning markers always forces REASONING regardless of confidence
    if reasoning_marker_count >= REASONING_MARKERS_THRESHOLD:
        return "reasoning", TIER_TO_MODEL["reasoning"], 0.97

    # Map confidence to tier
    if confidence < SIMPLE_BOUNDARY:
        tier = "simple"
    elif confidence < MEDIUM_BOUNDARY:
        tier = "medium"
    elif confidence < COMPLEX_BOUNDARY:
        tier = "complex"
    else:
        tier = "reasoning"

    # Bias correction: suppress 'complex' until bandit data validates classifier accuracy.
    # COMPLEX_BOUNDARY (0.70) < 0.80, so this condition is always true when tier == 'complex'.
    # To re-enable the 'complex' tier, lower this threshold to ~0.65 in a future update.
    if tier == "complex" and confidence < 0.80:
        tier = "medium"

    return tier, TIER_TO_MODEL[tier], confidence
