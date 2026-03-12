"""LongBench v2 generative mode — extract A/B/C/D from generated text."""

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from lm_eval.tasks._gptoss_utils import extract_final_channel


LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def process_results(doc, results):
    """Extract answer choice from generated text, compare to ground truth."""
    raw = results[0]
    text = extract_final_channel(raw).strip()

    pred_letter = extract_choice(text)
    pred_idx = LETTER_TO_IDX.get(pred_letter, -1)
    target = doc["answer"]  # integer: 0, 1, 2, 3

    correct = 1.0 if pred_idx == target else 0.0
    return {"acc": correct}


def extract_choice(text: str) -> str:
    """Extract A/B/C/D from model output. Cascading strategies."""
    text = text.strip()

    # Strategy 1: starts with a single letter (most common for instruct models)
    m = re.match(r"^[\(\[]?\s*([A-D])\s*[\)\]]?", text)
    if m:
        return m.group(1)

    # Strategy 2: "The answer is (X)" pattern
    m = re.search(r"[Aa]nswer\s*(?:is|:)\s*[\(\[]?\s*([A-D])\s*[\)\]]?", text)
    if m:
        return m.group(1)

    # Strategy 3: boxed answer
    m = re.search(r"\\boxed\{([A-D])\}", text)
    if m:
        return m.group(1)

    # Strategy 4: last standalone letter A-D in the text
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1]

    # Fallback: no valid choice found
    return ""
