import re
from typing import Dict, List

from lm_eval.tasks._gptoss_utils import extract_final_channel


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    """Evalscope-aligned BBH scoring with cascading extraction."""
    response = extract_final_channel(results[0])
    pred = extract_answer(response)
    target = doc["target"].strip()
    correct = normalize(pred) == normalize(target)
    return {"exact_match": int(correct)}


def extract_answer(response: str) -> str:
    """Extract answer from BBH response with cascading patterns.

    Priority:
    1. "So the answer is X" (evalscope anchor)
    2. "The answer is X" (lm-eval anchor)
    3. "The final answer is X"
    4. "ANSWER: X"
    5. Last line as fallback
    """
    # Pattern 1: "So the answer is X"
    m = re.search(r"So the answer is\s*(.+?)\.?\s*$", response, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Pattern 2: "The answer is X"
    m = re.search(r"[Tt]he answer is\s*(.+?)\.?\s*$", response, re.MULTILINE)
    if m:
        return m.group(1).strip()

    # Pattern 3: "The final answer is X"
    m = re.search(r"[Tt]he final answer is\s*(.+?)\.?\s*$", response, re.MULTILINE)
    if m:
        return m.group(1).strip()

    # Pattern 4: "ANSWER: X"
    m = re.search(r"(?i)ANSWER\s*:\s*(.+?)\.?\s*$", response, re.MULTILINE)
    if m:
        return m.group(1).strip()

    # Fallback: last non-empty line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return ""


def normalize(s: str) -> str:
    """Normalize answer for comparison: lowercase, strip punctuation/whitespace, strip markdown bold."""
    s = s.strip()
    # Strip markdown bold
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    # Strip trailing period
    s = re.sub(r"\.\s*$", "", s)
    # Strip surrounding quotes
    s = re.sub(r'^["\']|["\']$', "", s)
    # Lowercase
    s = s.lower().strip()
    return s
