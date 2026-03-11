"""Verify GPT-OSS channel stripping works across all aligned benchmarks.

Strategy: craft multi-channel responses where:
- Analysis channel contains a WRONG answer (decoy)
- Final channel contains the CORRECT answer
If channel stripping works, process_results picks up the correct answer.
Also tests that non-GPT-OSS (plain) responses pass through unchanged.
"""

import pytest

# ---------------------------------------------------------------------------
# Shared channel stripping module
# ---------------------------------------------------------------------------

from lm_eval.tasks._gptoss_utils import extract_final_channel


def _wrap_gptoss(analysis: str, final: str) -> str:
    """Build a GPT-OSS multi-channel response."""
    return (
        f"<|channel|>analysis<|message|>{analysis}"
        f"<|end|><|start|>assistant<|channel|>final<|message|>{final}"
    )


class TestExtractFinalChannel:
    def test_extracts_final(self):
        raw = _wrap_gptoss("thinking about B", "ANSWER: A")
        assert extract_final_channel(raw) == "ANSWER: A"

    def test_plain_passthrough(self):
        assert extract_final_channel("ANSWER: A") == "ANSWER: A"

    def test_analysis_only_returns_empty(self):
        raw = "<|channel|>analysis<|message|>still thinking..."
        assert extract_final_channel(raw) == ""

    def test_strips_trailing_special_tokens(self):
        raw = _wrap_gptoss("thinking", "the answer<|end|><|endoftext|>")
        assert extract_final_channel(raw) == "the answer"


# ---------------------------------------------------------------------------
# MCQ benchmarks: GPQA, BBH, MMLU-Redux, MMLU-Pro, C-Eval
# ---------------------------------------------------------------------------

class TestGPQA:
    def test_gptoss_channel(self):
        from lm_eval.tasks.gpqa.aligned.utils import process_results
        doc = {"answer": "(A)"}
        # Analysis says (B), final says (A)
        resp = _wrap_gptoss("I think the answer is (B)", "ANSWER: A")
        result = process_results(doc, [resp])
        assert result["exact_match"] == 1

    def test_gptoss_wrong_decoy(self):
        from lm_eval.tasks.gpqa.aligned.utils import process_results
        doc = {"answer": "(A)"}
        # Analysis says (A) — correct, but final says (B) — wrong
        resp = _wrap_gptoss("I think the answer is (A)", "ANSWER: B")
        result = process_results(doc, [resp])
        assert result["exact_match"] == 0

    def test_plain_response(self):
        from lm_eval.tasks.gpqa.aligned.utils import process_results
        doc = {"answer": "(C)"}
        result = process_results(doc, ["ANSWER: C"])
        assert result["exact_match"] == 1


class TestBBH:
    def test_gptoss_channel(self):
        from lm_eval.tasks.bbh.aligned.utils import process_results
        doc = {"target": "True"}
        resp = _wrap_gptoss(
            "Let me analyze... So the answer is False.",
            "So the answer is True."
        )
        result = process_results(doc, [resp])
        assert result["exact_match"] == 1

    def test_gptoss_decoy(self):
        from lm_eval.tasks.bbh.aligned.utils import process_results
        doc = {"target": "True"}
        resp = _wrap_gptoss(
            "So the answer is True.",
            "So the answer is False."
        )
        result = process_results(doc, [resp])
        assert result["exact_match"] == 0

    def test_plain_response(self):
        from lm_eval.tasks.bbh.aligned.utils import process_results
        doc = {"target": "(A)"}
        result = process_results(doc, ["So the answer is (A)."])
        assert result["exact_match"] == 1


class TestMMLURedux:
    def test_gptoss_channel(self):
        import importlib
        mod = importlib.import_module("lm_eval.tasks.mmlu-redux.aligned.utils")
        doc = {"answer": 0}  # 0 → "A"
        resp = _wrap_gptoss("I think B", "ANSWER: A")
        result = mod.process_results(doc, [resp])
        assert result["exact_match"] == 1

    def test_plain_response(self):
        import importlib
        mod = importlib.import_module("lm_eval.tasks.mmlu-redux.aligned.utils")
        doc = {"answer": 2}  # 2 → "C"
        result = mod.process_results(doc, ["ANSWER: C"])
        assert result["exact_match"] == 1


class TestMMLUPro:
    def test_gptoss_channel(self):
        from lm_eval.tasks.mmlu_pro.aligned.utils import process_results
        doc = {"options": ["opt1", "opt2", "opt3", "opt4"], "answer": "A"}
        resp = _wrap_gptoss("Hmm, probably B", "ANSWER: A")
        result = process_results(doc, [resp])
        assert result["exact_match"] == 1

    def test_plain_response(self):
        from lm_eval.tasks.mmlu_pro.aligned.utils import process_results
        doc = {"options": ["o1", "o2", "o3", "o4", "o5"], "answer": "D"}
        result = process_results(doc, ["ANSWER: D"])
        assert result["exact_match"] == 1


class TestCEval:
    def test_gptoss_channel(self):
        from lm_eval.tasks.ceval.aligned.utils import process_results
        doc = {"answer": "B"}
        resp = _wrap_gptoss("答案是A", "答案是B")
        result = process_results(doc, [resp])
        assert result["exact_match"] == 1

    def test_plain_response(self):
        from lm_eval.tasks.ceval.aligned.utils import process_results
        doc = {"answer": "C"}
        result = process_results(doc, ["ANSWER: C"])
        assert result["exact_match"] == 1


# ---------------------------------------------------------------------------
# Math benchmarks: AIME, GSM8K, HMMT, MATH-500
# ---------------------------------------------------------------------------

class TestAIME:
    def test_gptoss_channel(self):
        from lm_eval.tasks.aime.utils import process_results
        doc = {"Answer": "42"}
        resp = _wrap_gptoss(
            "Let me compute... \\boxed{99}",
            "The answer is \\boxed{42}"
        )
        result = process_results(doc, [resp])
        assert result["exact_match"] == 1

    def test_gptoss_decoy(self):
        from lm_eval.tasks.aime.utils import process_results
        doc = {"Answer": "42"}
        resp = _wrap_gptoss(
            "\\boxed{42}",
            "\\boxed{99}"
        )
        result = process_results(doc, [resp])
        assert result["exact_match"] == 0

    def test_plain_response(self):
        from lm_eval.tasks.aime.utils import process_results
        doc = {"Answer": "7"}
        result = process_results(doc, ["So \\boxed{7}"])
        assert result["exact_match"] == 1


class TestGSM8K:
    def test_gptoss_channel(self):
        from lm_eval.tasks.gsm8k.utils import process_results_aligned
        doc = {"answer": "Some steps\n#### 15"}
        resp = _wrap_gptoss("\\boxed{999}", "\\boxed{15}")
        result = process_results_aligned(doc, [resp])
        assert result["exact_match"] == 1

    def test_plain_response(self):
        from lm_eval.tasks.gsm8k.utils import process_results_aligned
        doc = {"answer": "Reasoning\n#### 200"}
        result = process_results_aligned(doc, ["Therefore \\boxed{200}"])
        assert result["exact_match"] == 1


class TestHMMT:
    def test_gptoss_channel(self):
        from lm_eval.tasks.hmmt.utils import process_results
        doc = {"answer": "5"}
        resp = _wrap_gptoss("\\boxed{3}", "\\boxed{5}")
        result = process_results(doc, [resp])
        assert result["exact_match"] == 1

    def test_plain_response(self):
        from lm_eval.tasks.hmmt.utils import process_results
        doc = {"answer": "12"}
        result = process_results(doc, ["\\boxed{12}"])
        assert result["exact_match"] == 1


class TestMATH500:
    def test_gptoss_channel(self):
        from lm_eval.tasks.hendrycks_math.utils import process_results_aligned
        doc = {"answer": "\\frac{1}{2}"}
        resp = _wrap_gptoss("\\boxed{\\frac{1}{3}}", "\\boxed{\\frac{1}{2}}")
        result = process_results_aligned(doc, [resp])
        assert result["exact_match"] == 1

    def test_plain_response(self):
        from lm_eval.tasks.hendrycks_math.utils import process_results_aligned
        doc = {"answer": "3"}
        result = process_results_aligned(doc, ["\\boxed{3}"])
        assert result["exact_match"] == 1


# ---------------------------------------------------------------------------
# IFEval — channel stripping before instruction-following check
# ---------------------------------------------------------------------------

class TestIFEval:
    def test_gptoss_channel(self):
        from lm_eval.tasks.ifeval.aligned.utils import process_results
        doc = {
            "key": 0,
            "instruction_id_list": ["length_constraints:number_words"],
            "prompt": "Write exactly 3 words.",
            "kwargs": [{"relation": "at least", "num_words": 3}],
        }
        # Analysis is long, final is exactly 3 words
        resp = _wrap_gptoss(
            "Let me think about how to write exactly three words for this prompt...",
            "Hello beautiful world"
        )
        result = process_results(doc, [resp])
        assert result["prompt_level_loose_acc"] is True

    def test_plain_response(self):
        from lm_eval.tasks.ifeval.aligned.utils import process_results
        doc = {
            "key": 1,
            "instruction_id_list": ["length_constraints:number_words"],
            "prompt": "Write at least 3 words.",
            "kwargs": [{"relation": "at least", "num_words": 3}],
        }
        result = process_results(doc, ["Hello beautiful world today"])
        assert result["prompt_level_loose_acc"] is True
