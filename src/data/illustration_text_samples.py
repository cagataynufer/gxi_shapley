"""
Fixed text samples for illustration purposes.

These sentence pairs are used to illustrate token-level interaction
effects in a pretrained Natural Language Inference (NLI) model.
They are intentionally hard-coded to ensure full reproducibility and
independence from external datasets.
"""

TEXT_SAMPLES = {
    "nli_contradiction_simple": (
        "The movie was good.",
        "The movie failed completely."
    ),

    "nli_contradiction_contextual": (
        "The plan sounded good on paper.",
        "The plan was a complete failure in practice."
    ),

    "nli_conditional_reinterpretation": (
        "The decision was good.",
        "The decision was good only if the goal was to make things worse."
    ),
}
