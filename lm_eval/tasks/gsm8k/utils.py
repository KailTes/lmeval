from typing import Dict, List


def process_results_aligned(doc: dict, results: List[str]) -> Dict[str, int]:
    """Evalscope-aligned scoring: cascading extraction + math-aware comparison.

    Used by gsm8k_cot_zeroshot_aligned (CoT + boxed prompt) to match evalscope's scoring.
    Reuses the AIME utils which were ported from evalscope.
    """
    from lm_eval.tasks.aime.utils import extract_answer, math_equal

    response = results[0]
    pred = extract_answer(response)
    # GSM8K gold answers are after "####" in the solution
    target = doc["answer"].split("####")[-1].strip()
    correct = math_equal(pred, target)
    return {"exact_match": int(correct)}
