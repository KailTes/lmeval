from typing import Dict, List


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """Evalscope-aligned scoring: cascading extraction + math-aware comparison."""
    from lm_eval.tasks.aime.utils import extract_answer, math_equal

    response = results[0]
    pred = extract_answer(response)
    target = str(doc["answer"])
    correct = math_equal(pred, target)
    return {"exact_match": int(correct)}
