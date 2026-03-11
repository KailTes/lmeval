"""GPT-OSS channel stripping utilities shared across all aligned benchmarks.

GPT-OSS outputs multi-channel responses:
  <|channel|>analysis<|message|>[thinking]<|end|><|start|>assistant<|channel|>final<|message|>[actual response]

When using lm-eval in offline mode (no vLLM chat completions endpoint), the raw
channel tags are present in the output. This module strips the analysis channel
and extracts only the final response.

Compatible with non-GPT-OSS models: if no channel tags are present, the response
is returned unchanged.
"""

import re

_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*)",
    re.DOTALL,
)
_TRAILING_SPECIAL_RE = re.compile(
    r"<\|(end|start|channel|message|return|im_end|endoftext|eot_id)\|>.*$",
    re.DOTALL,
)


def extract_final_channel(response: str) -> str:
    """Extract 'final' channel content from GPT-OSS multi-channel output.

    Returns the original response unchanged for non-GPT-OSS models.
    """
    m = _FINAL_CHANNEL_RE.search(response)
    if m:
        content = m.group(1)
        content = _TRAILING_SPECIAL_RE.sub("", content)
        return content.strip()
    # Analysis-only output (no final channel) — model didn't produce a final answer
    if "<|channel|>analysis" in response:
        return ""
    return response
