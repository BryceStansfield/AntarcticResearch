#!/usr/bin/env python3
"""
Check whether the cached intervention-classification prompt (instructions + few-shot
examples in antarctic_ladder_metrics/final_report_metrics.py) is long enough to actually
get cached by OpenRouter.

Claude Sonnet 4.6 requires a minimum 1,024-token prefix before OpenRouter will write a
cache entry for it - shorter prefixes are silently never cached (cache_creation_input_tokens
stays 0 forever, no error). This script measures the real prefix against the real Claude
tokenizer via Anthropic's /v1/messages/count_tokens endpoint, so the number reflects what
will actually happen, not a chars/4 guess.

Usage:
    python check_intervention_prompt_length.py
"""

import sys

import requests

import secret_management
from antarctic_ladder_metrics.final_report_metrics import (
    FEW_SHOT_EXAMPLES,
    INTERVENTION_CACHED_PREFIX,
    INTERVENTION_INSTRUCTIONS,
)

# Must match the model actually used for classification (antarctic_ladder_metrics/
# final_report_metrics.py: INTERVENTION_MODEL), so the count reflects that model's
# tokenizer/minimum. OpenRouter's slug uses dots ("claude-sonnet-4.6"); the native
# Anthropic count_tokens endpoint wants the first-party ID with hyphens.
ANTHROPIC_MODEL_ID = "claude-sonnet-4-6"
MIN_CACHEABLE_TOKENS = 1024


def count_tokens(text: str) -> int:
    api_key = secret_management.get("ANTHROPIC_API_KEY")
    resp = requests.post(
        "https://api.anthropic.com/v1/messages/count_tokens",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": ANTHROPIC_MODEL_ID,
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["input_tokens"]


def main() -> None:
    instructions_tokens = count_tokens(INTERVENTION_INSTRUCTIONS)
    total_tokens = count_tokens(INTERVENTION_CACHED_PREFIX)
    examples_tokens = total_tokens - instructions_tokens
    num_examples = len(FEW_SHOT_EXAMPLES)

    print(f"Model checked against:      {ANTHROPIC_MODEL_ID}")
    print(f"Instructions only:          {instructions_tokens} tokens")
    print(f"Few-shot examples:          {num_examples} examples, {examples_tokens} tokens "
          f"({examples_tokens / num_examples:.0f} tokens/example avg)" if num_examples else
          "Few-shot examples:          none")
    print(f"Total cached prefix:        {total_tokens} tokens")
    print(f"Cacheable minimum required: {MIN_CACHEABLE_TOKENS} tokens")
    print()

    if total_tokens >= MIN_CACHEABLE_TOKENS:
        print(f"✅ Long enough - {total_tokens} >= {MIN_CACHEABLE_TOKENS}. This prefix will be cached.")
    else:
        shortfall = MIN_CACHEABLE_TOKENS - total_tokens
        avg = examples_tokens / num_examples if num_examples else 100
        more_needed = max(1, round(shortfall / avg))
        print(f"❌ Too short by {shortfall} tokens - this prefix will NOT be cached "
              f"(cache_creation_input_tokens will silently stay 0).")
        print(f"   Add roughly {more_needed} more example(s) of similar length to clear the threshold.")
        sys.exit(1)


if __name__ == "__main__":
    main()
