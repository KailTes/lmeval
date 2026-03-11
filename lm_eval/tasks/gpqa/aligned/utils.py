import random
import re
from typing import Dict, List

import datasets

from lm_eval.tasks._gptoss_utils import extract_final_channel


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """Robust multiple-choice answer extraction with cascading patterns."""
    response = extract_final_channel(results[0])
    target = doc["answer"]  # e.g. "(A)"
    pred = extract_choice(response)
    correct = pred is not None and pred == target
    return {"exact_match": int(correct)}


def extract_choice(response: str) -> str | None:
    """Extract multiple-choice answer (A)/(B)/(C)/(D) with cascading patterns.

    Priority order:
    1. "ANSWER: X" format (evalscope style)
    2. "The answer is (X)" format (lm-eval style)
    3. "\\boxed{X}" format
    4. Last standalone (A)/(B)/(C)/(D) occurrence
    """
    # Pattern 1: ANSWER: X (case-insensitive)
    m = re.search(r"(?i)ANSWER\s*:\s*\(?([A-D])\)?", response)
    if m:
        return f"({m.group(1).upper()})"

    # Pattern 2: The answer is (X)
    m = re.search(r"[Tt]he answer is\s*\(?([A-D])\)?", response)
    if m:
        return f"({m.group(1).upper()})"

    # Pattern 3: \boxed{X}
    m = re.search(r"\\boxed\{?\(?([A-D])\)?\}?", response)
    if m:
        return f"({m.group(1).upper()})"

    # Pattern 4: Last standalone (A)/(B)/(C)/(D)
    matches = re.findall(r"\(([A-D])\)", response)
    if matches:
        return f"({matches[-1].upper()})"

    return None
