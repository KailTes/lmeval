"""LongBench aligned scoring — wraps upstream metrics with GPT-OSS channel stripping.

All wrapper functions apply extract_final_channel() to the model response
before delegating to the original LongBench metric functions.

This ensures GPT-OSS analysis channel content doesn't contaminate F1, ROUGE,
classification, code similarity, or other LongBench metrics.
"""

from lm_eval.tasks._gptoss_utils import extract_final_channel
from lm_eval.tasks.longbench.metrics import (
    get_classification_with_score,
    get_code_sim_with_score,
    get_count_with_score,
    get_qa_f1_with_score,
    get_qa_f1_zh_with_score,
    get_retrieval_with_score,
    get_retrieval_zh_with_score,
    get_rouge_with_score,
    get_rouge_zh_with_score,
)


def _strip_channel(results: list[str]) -> list[str]:
    """Apply GPT-OSS channel stripping to the first result."""
    return [extract_final_channel(results[0])] + results[1:]


# ---------------------------------------------------------------------------
# Wrapped metric functions — same signature as originals
# ---------------------------------------------------------------------------

def aligned_qa_f1(doc, results, **kwargs):
    return get_qa_f1_with_score(doc, _strip_channel(results), **kwargs)


def aligned_qa_f1_zh(doc, results, **kwargs):
    return get_qa_f1_zh_with_score(doc, _strip_channel(results), **kwargs)


def aligned_rouge(doc, results, **kwargs):
    return get_rouge_with_score(doc, _strip_channel(results), **kwargs)


def aligned_rouge_zh(doc, results, **kwargs):
    return get_rouge_zh_with_score(doc, _strip_channel(results), **kwargs)


def aligned_classification(doc, results, **kwargs):
    return get_classification_with_score(doc, _strip_channel(results), **kwargs)


def aligned_count(doc, results, **kwargs):
    return get_count_with_score(doc, _strip_channel(results), **kwargs)


def aligned_retrieval(doc, results, **kwargs):
    return get_retrieval_with_score(doc, _strip_channel(results), **kwargs)


def aligned_retrieval_zh(doc, results, **kwargs):
    return get_retrieval_zh_with_score(doc, _strip_channel(results), **kwargs)


def aligned_code_sim(doc, results, **kwargs):
    return get_code_sim_with_score(doc, _strip_channel(results), **kwargs)
