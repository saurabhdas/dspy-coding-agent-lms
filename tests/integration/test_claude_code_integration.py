"""Integration tests for Claude Code CLI.

These tests require an actual Claude Code CLI installation and valid
authentication. They are skipped by default unless explicitly enabled.

To run integration tests:
    RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/ -v
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

# Skip all tests in this module unless explicitly enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled (set RUN_INTEGRATION_TESTS=1 to enable)",
)


class TestClaudeCodeIntegration:
    """Integration tests with real Claude Code CLI."""

    def test_simple_prompt(self) -> None:
        """Test simple prompt execution."""
        from dspy_coding_agent_lms import ClaudeCodeLM

        lm = ClaudeCodeLM(
            model="sonnet",
            permission_mode="plan",
            timeout_seconds=60.0,
        )

        result = lm.forward(prompt="What is 2+2? Reply with just the number.")

        assert len(result) == 1
        assert "4" in result[0]["text"]

    def test_structured_output(self) -> None:
        """Test structured output with JSON schema."""
        from dspy_coding_agent_lms import ClaudeCodeLM

        lm = ClaudeCodeLM(
            model="sonnet",
            permission_mode="plan",
            timeout_seconds=60.0,
        )

        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "integer"},
                "explanation": {"type": "string"},
            },
            "required": ["answer"],
        }

        result = lm.forward(
            prompt="What is 2+2? Provide the answer as an integer.",
            json_schema=schema,
        )

        # Parse structured output
        data = json.loads(result[0]["text"])
        assert data["answer"] == 4

    def test_transcript_capture(self) -> None:
        """Test transcript is captured."""
        from dspy_coding_agent_lms import ClaudeCodeLM

        lm = ClaudeCodeLM(
            model="sonnet",
            permission_mode="plan",
            timeout_seconds=60.0,
            capture_transcript=True,
        )

        lm.forward(prompt="Say hello")

        assert len(lm.transcript) == 1
        entry = lm.transcript.last
        assert entry is not None
        assert entry.prompt == "Say hello"
        assert entry.session_id is not None

    def test_cost_tracking(self) -> None:
        """Test cost tracking works."""
        from dspy_coding_agent_lms import ClaudeCodeLM

        lm = ClaudeCodeLM(
            model="sonnet",
            permission_mode="plan",
            timeout_seconds=60.0,
        )

        lm.forward(prompt="Say one word")

        cost = lm.get_total_cost()
        assert cost >= 0  # Should have some cost

        tokens = lm.get_total_tokens()
        assert tokens["output"] > 0

    @pytest.mark.asyncio
    async def test_async_execution(self) -> None:
        """Test async execution."""
        from dspy_coding_agent_lms import ClaudeCodeLM

        lm = ClaudeCodeLM(
            model="sonnet",
            permission_mode="plan",
            timeout_seconds=60.0,
        )

        result = await lm.aforward(prompt="Say hello")

        assert len(result) == 1
        assert len(result[0]["text"]) > 0

    @pytest.mark.asyncio
    async def test_concurrent_requests(self) -> None:
        """Test concurrent async requests."""
        import asyncio

        from dspy_coding_agent_lms import ClaudeCodeLM

        lm = ClaudeCodeLM(
            model="sonnet",
            permission_mode="plan",
            timeout_seconds=120.0,
        )

        prompts = [
            "What is 1+1? Reply with just the number.",
            "What is 2+2? Reply with just the number.",
            "What is 3+3? Reply with just the number.",
        ]

        tasks = [lm.aforward(prompt=p) for p in prompts]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        # Check expected answers
        assert "2" in results[0][0]["text"]
        assert "4" in results[1][0]["text"]
        assert "6" in results[2][0]["text"]


class TestDSPyIntegration:
    """Integration tests with DSPy framework."""

    def test_dspy_predict(self) -> None:
        """Test using ClaudeCodeLM with DSPy Predict."""
        try:
            import dspy
        except ImportError:
            pytest.skip("dspy not installed")

        from dspy_coding_agent_lms import ClaudeCodeLM

        lm = ClaudeCodeLM(
            model="sonnet",
            permission_mode="plan",
            timeout_seconds=60.0,
        )
        dspy.configure(lm=lm)

        # Simple predict
        predict = dspy.Predict("question -> answer")
        result = predict(question="What is the capital of France?")

        assert result.answer is not None
        assert "Paris" in result.answer or "paris" in result.answer.lower()

    def test_dspy_chain_of_thought(self) -> None:
        """Test using ClaudeCodeLM with DSPy ChainOfThought."""
        try:
            import dspy
        except ImportError:
            pytest.skip("dspy not installed")

        from dspy_coding_agent_lms import ClaudeCodeLM

        lm = ClaudeCodeLM(
            model="sonnet",
            permission_mode="plan",
            timeout_seconds=90.0,
        )
        dspy.configure(lm=lm)

        # Chain of thought
        cot = dspy.ChainOfThought("question -> answer")
        result = cot(question="If a train travels 60 mph for 2 hours, how far does it go?")

        assert result.answer is not None
        assert "120" in result.answer
