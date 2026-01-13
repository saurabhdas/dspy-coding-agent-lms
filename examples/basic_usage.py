#!/usr/bin/env python3
"""Basic usage example for dspy-coding-agent-lms.

This example demonstrates the simplest way to use ClaudeCodeLM
with DSPy for basic question answering.

Requirements:
    - Claude Code CLI installed and authenticated
    - dspy-coding-agent-lms installed

Run:
    python examples/basic_usage.py
"""

from __future__ import annotations

import dspy

from dspy_coding_agent_lms import ClaudeCodeLM


def main() -> None:
    """Run basic usage example."""
    # Initialize the LM with default settings
    # - model="sonnet" (default, fast and capable)
    # - permission_mode="plan" (safe mode, allows planning but not execution)
    lm = ClaudeCodeLM(
        model="sonnet",
        permission_mode="plan",
    )

    # Configure DSPy to use our LM
    dspy.configure(lm=lm)

    print("DSPy configured with ClaudeCodeLM")
    print(f"Model: {lm.config.model}")
    print(f"Permission mode: {lm.config.permission_mode}")
    print()

    # Simple prediction
    predict = dspy.Predict("question -> answer")

    questions = [
        "What is the capital of France?",
        "What is 42 * 17?",
        "Who wrote Romeo and Juliet?",
    ]

    for question in questions:
        print(f"Q: {question}")
        result = predict(question=question)
        print(f"A: {result.answer}")
        print()

    # Show transcript summary
    print("=" * 50)
    print("Session Summary:")
    print(f"  Total requests: {len(lm.transcript)}")
    print(f"  Total cost: ${lm.transcript.total_cost_usd():.4f}")

    tokens = lm.transcript.total_tokens()
    print(f"  Input tokens: {tokens['input']:,}")
    print(f"  Output tokens: {tokens['output']:,}")
    print(f"  Cache read: {tokens['cache_read']:,}")

    avg_duration = lm.transcript.average_duration_ms()
    print(f"  Avg response time: {avg_duration:.0f}ms")


if __name__ == "__main__":
    main()
