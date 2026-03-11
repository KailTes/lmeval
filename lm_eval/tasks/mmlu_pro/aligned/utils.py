import re
from functools import partial
from typing import Dict, List


LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def doc_to_text(doc: dict) -> str:
    """Evalscope-aligned prompt for MMLU-Pro with dynamic option count."""
    options = doc["options"]
    n = len(options)
    letter_list = ",".join(LETTERS[:n])

    prompt = (
        "Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of "
        f"{letter_list}. Think step by step before answering.\n\n"
        f"{doc['question'].strip()}\n\n"
    )
    for i, opt in enumerate(options):
        if i >= len(LETTERS):
            break
        prompt += f"{LETTERS[i]}) {opt.strip()}\n"

    return prompt


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """Evalscope-aligned MMLU-Pro scoring with cascading letter extraction."""
    response = results[0]
    n_options = len(doc["options"])
    pred = extract_choice(response, n_options)
    target = doc["answer"]
    correct = pred is not None and pred == target
    return {"exact_match": int(correct)}


def extract_choice(response: str, n_options: int = 10) -> str | None:
    """Extract multiple-choice answer with cascading patterns.

    Supports variable option counts (up to 10, A-J).

    Priority:
    1. "ANSWER: X" format (evalscope style)
    2. "The answer is (X)" / "the answer is X"
    3. "\\boxed{X}"
    4. Last standalone letter in parentheses
    5. Last single letter on its own line
    """
    letter_range = "".join(LETTERS[:n_options])
    pattern = f"[{letter_range}]"

    # Pattern 1: ANSWER: X (case-insensitive)
    m = re.search(rf"(?i)ANSWER\s*:\s*\(?({pattern})\)?", response)
    if m:
        return m.group(1).upper()

    # Pattern 2: The answer is (X)
    m = re.search(rf"[Tt]he answer is\s*\(?({pattern})\)?", response)
    if m:
        return m.group(1).upper()

    # Pattern 3: \boxed{X}
    m = re.search(rf"\\boxed\{{?\(?({pattern})\)?\}}?", response)
    if m:
        return m.group(1).upper()

    # Pattern 4: Last (X) in parentheses
    matches = re.findall(rf"\(({pattern})\)", response)
    if matches:
        return matches[-1].upper()

    # Pattern 5: Last standalone letter on its own line
    matches_line = list(
        re.finditer(rf"(?:^|\n)\s*({pattern})\s*\.?\s*$", response, re.MULTILINE)
    )
    if matches_line:
        return matches_line[-1].group(1).upper()

    return None


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


process_biology = partial(process_docs, subject="biology")
process_business = partial(process_docs, subject="business")
process_chemistry = partial(process_docs, subject="chemistry")
process_computer_science = partial(process_docs, subject="computer science")
process_economics = partial(process_docs, subject="economics")
process_engineering = partial(process_docs, subject="engineering")
process_health = partial(process_docs, subject="health")
process_history = partial(process_docs, subject="history")
process_law = partial(process_docs, subject="law")
process_math = partial(process_docs, subject="math")
process_other = partial(process_docs, subject="other")
process_philosophy = partial(process_docs, subject="philosophy")
process_physics = partial(process_docs, subject="physics")
process_psychology = partial(process_docs, subject="psychology")
