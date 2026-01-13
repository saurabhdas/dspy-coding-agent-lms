"""Tests for lm module."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from dspy_coding_agent_lms import ClaudeCodeLM
from dspy_coding_agent_lms.exceptions import ClaudeCodeError


class TestClaudeCodeLMInit:
    """Tests for ClaudeCodeLM initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        lm = ClaudeCodeLM()

        # Model name is mapped to litellm-compatible format for JSONAdapter
        assert lm.model == "anthropic/claude-sonnet-4-5-20250929"
        assert lm.model_type == "chat"
        assert lm.config.model == "sonnet"  # CLI uses the alias
        assert lm.config.permission_mode == "plan"
        assert lm.cache is True  # BaseLM cache attribute
        assert lm._capture_transcript is True
        assert len(lm.history) == 0

    def test_custom_init(self) -> None:
        """Test custom initialization."""
        lm = ClaudeCodeLM(
            model="opus",
            permission_mode="dontAsk",
            dangerously_skip_permissions=True,
            allowed_tools=["Read", "Glob"],
            working_directory="/tmp/test",
            timeout_seconds=600.0,
            cache=False,
        )

        assert lm.config.model == "opus"
        assert lm.config.permission_mode == "dontAsk"
        assert lm.config.dangerously_skip_permissions is True
        assert "Read" in lm.config.allowed_tools
        assert lm.config.working_directory == "/tmp/test"
        assert lm.config.timeout_seconds == 600.0
        assert lm.cache is False  # BaseLM cache attribute

    def test_init_with_auth(self) -> None:
        """Test initialization with authentication."""
        lm = ClaudeCodeLM(
            anthropic_api_key="sk-test",
            oauth_token="oauth-test",
        )

        assert lm.auth.anthropic_api_key == "sk-test"
        assert lm.auth.oauth_token == "oauth-test"


class TestClaudeCodeLMForward:
    """Tests for forward method."""

    def test_forward_with_prompt(
        self, mocker: Any, mock_cli_response: dict[str, Any]
    ) -> None:
        """Test forward with simple prompt."""
        mock_execute = mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.execute"
        )
        mock_execute.return_value = json.dumps(mock_cli_response)

        lm = ClaudeCodeLM(cache=False)
        result = lm.forward(prompt="Hello")

        # Result is now an LMResponse object
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Test response from Claude Code"
        assert result.choices[0].finish_reason == "stop"
        mock_execute.assert_called_once()

    def test_forward_with_messages(
        self, mocker: Any, mock_cli_response: dict[str, Any]
    ) -> None:
        """Test forward with chat messages."""
        mock_execute = mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.execute"
        )
        mock_execute.return_value = json.dumps(mock_cli_response)

        lm = ClaudeCodeLM(cache=False)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        result = lm.forward(messages=messages)

        # Result is now an LMResponse object
        assert len(result.choices) == 1
        # Check that the prompt was constructed from messages
        call_args = mock_execute.call_args
        prompt = call_args[1]["prompt"] if "prompt" in call_args[1] else call_args[0][0]
        assert "Hello" in prompt
        assert "Hi there" in prompt
        assert "How are you?" in prompt

    def test_forward_with_json_schema(
        self,
        mocker: Any,
        mock_structured_response: dict[str, Any],
        sample_json_schema: dict[str, Any],
    ) -> None:
        """Test forward with JSON schema."""
        mock_execute = mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.execute"
        )
        mock_execute.return_value = json.dumps(mock_structured_response)

        lm = ClaudeCodeLM(cache=False)
        result = lm.forward(prompt="Test", json_schema=sample_json_schema)

        # Check that json_schema was passed to execute
        call_kwargs = mock_execute.call_args[1]
        assert call_kwargs["json_schema"] == sample_json_schema

        # Result should contain structured output (now as LMResponse)
        text = result.choices[0].message.content
        parsed = json.loads(text)
        assert parsed["name"] == "Test"

    def test_forward_missing_prompt(self) -> None:
        """Test forward without prompt or messages raises error."""
        lm = ClaudeCodeLM(cache=False)

        with pytest.raises(ValueError, match="Either prompt or messages"):
            lm.forward()

    def test_forward_updates_history(
        self, mocker: Any, mock_cli_response: dict[str, Any]
    ) -> None:
        """Test forward updates history via transcript."""
        mock_execute = mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.execute"
        )
        mock_execute.return_value = json.dumps(mock_cli_response)

        lm = ClaudeCodeLM(cache=False)

        lm.forward(prompt="Test")

        # ClaudeCodeLM uses transcript for history
        assert len(lm.transcript) == 1
        assert lm.transcript.last.prompt == "Test"

    def test_forward_captures_transcript(
        self, mocker: Any, mock_cli_response: dict[str, Any]
    ) -> None:
        """Test forward captures transcript."""
        mock_execute = mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.execute"
        )
        mock_execute.return_value = json.dumps(mock_cli_response)

        lm = ClaudeCodeLM(cache=False, capture_transcript=True)
        lm.forward(prompt="Test prompt")

        assert len(lm.transcript) == 1
        assert lm.transcript.last.prompt == "Test prompt"
        assert lm.transcript.last.session_id == "test-session-123"

    def test_forward_uses_cache(
        self, mocker: Any, mock_cli_response: dict[str, Any], tmp_path: Any
    ) -> None:
        """Test forward uses cache."""
        mock_execute = mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.execute"
        )
        mock_execute.return_value = json.dumps(mock_cli_response)

        # Use a temporary cache directory to avoid interference from other tests
        cache_dir = str(tmp_path / "cache")
        lm = ClaudeCodeLM(cache=True, cache_dir=cache_dir)

        # First call
        result1 = lm.forward(prompt="Test cache prompt")
        assert mock_execute.call_count == 1

        # Second call with same prompt should use cache
        result2 = lm.forward(prompt="Test cache prompt")
        assert mock_execute.call_count == 1  # Not called again

        # Compare content (objects are different due to deserialization)
        assert result1.choices[0].message.content == result2.choices[0].message.content
        assert result1.model == result2.model

    def test_forward_execution_error(self, mocker: Any) -> None:
        """Test forward handles execution error."""
        mock_execute = mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.execute"
        )
        mock_execute.side_effect = Exception("CLI failed")

        lm = ClaudeCodeLM(cache=False)

        with pytest.raises(ClaudeCodeError, match="CLI execution failed"):
            lm.forward(prompt="Test")


class TestClaudeCodeLMAsync:
    """Tests for async forward method."""

    @pytest.mark.asyncio
    async def test_aforward(
        self, mocker: Any, mock_cli_response: dict[str, Any]
    ) -> None:
        """Test async forward."""
        async def mock_aexecute(*args: Any, **kwargs: Any) -> str:
            return json.dumps(mock_cli_response)

        mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.aexecute",
            side_effect=mock_aexecute,
        )

        lm = ClaudeCodeLM(cache=False)
        result = await lm.aforward(prompt="Test async")

        # Result is now an LMResponse object
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Test response from Claude Code"


class TestClaudeCodeLMHelpers:
    """Tests for helper methods."""

    def test_callable(self, mocker: Any, mock_cli_response: dict[str, Any]) -> None:
        """Test __call__ method."""
        mock_execute = mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.execute"
        )
        mock_execute.return_value = json.dumps(mock_cli_response)

        lm = ClaudeCodeLM(cache=False)
        result = lm(prompt="Test callable")

        # BaseLM.__call__ processes response and returns list of outputs
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "Test response from Claude Code"

    def test_copy(self) -> None:
        """Test copy method."""
        lm = ClaudeCodeLM(model="sonnet", permission_mode="plan")
        lm_copy = lm.copy(model="opus", timeout_seconds=600.0)

        # Original unchanged
        assert lm.config.model == "sonnet"

        # Copy has updates
        assert lm_copy.config.model == "opus"
        assert lm_copy.config.timeout_seconds == 600.0
        # Unchanged values preserved
        assert lm_copy.config.permission_mode == "plan"

    def test_dump_state(self) -> None:
        """Test dump_state method."""
        lm = ClaudeCodeLM(model="opus", permission_mode="dontAsk")
        state = lm.dump_state()

        assert state["model"] == "opus"
        assert state["permission_mode"] == "dontAsk"
        assert state["cache_enabled"] is True
        assert state["history_length"] == 0

    def test_transcript_entries(
        self, mocker: Any, mock_cli_response: dict[str, Any]
    ) -> None:
        """Test transcript captures multiple entries."""
        mock_execute = mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.execute"
        )
        mock_execute.return_value = json.dumps(mock_cli_response)

        lm = ClaudeCodeLM(cache=False)
        lm.forward(prompt="Test 1")
        lm.forward(prompt="Test 2")
        lm.forward(prompt="Test 3")

        # Check transcript has all entries
        assert len(lm.transcript) == 3
        entries = list(lm.transcript)
        assert entries[0].prompt == "Test 1"
        assert entries[1].prompt == "Test 2"
        assert entries[2].prompt == "Test 3"

    def test_clear_transcript(
        self, mocker: Any, mock_cli_response: dict[str, Any]
    ) -> None:
        """Test clear transcript."""
        from dspy_coding_agent_lms.transcript import Transcript

        mock_execute = mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.execute"
        )
        mock_execute.return_value = json.dumps(mock_cli_response)

        lm = ClaudeCodeLM(cache=False)
        lm.forward(prompt="Test")

        assert len(lm.transcript) == 1
        lm._transcript = Transcript()
        assert len(lm.transcript) == 0

    def test_get_total_cost(
        self, mocker: Any, mock_cli_response: dict[str, Any]
    ) -> None:
        """Test get_total_cost method."""
        mock_execute = mocker.patch(
            "dspy_coding_agent_lms.cli.ClaudeCodeCLI.execute"
        )
        mock_execute.return_value = json.dumps(mock_cli_response)

        lm = ClaudeCodeLM(cache=False)
        lm.forward(prompt="Test 1")
        lm.forward(prompt="Test 2")

        cost = lm.get_total_cost()
        assert cost == pytest.approx(0.02)  # 0.01 * 2

    def test_repr(self) -> None:
        """Test __repr__ method."""
        lm = ClaudeCodeLM(model="opus", permission_mode="dontAsk")
        repr_str = repr(lm)

        assert "ClaudeCodeLM" in repr_str
        assert "opus" in repr_str
        assert "dontAsk" in repr_str


class TestClaudeCodeLMPreparePrompt:
    """Tests for _prepare_prompt method."""

    def test_prepare_from_string(self) -> None:
        """Test preparing prompt from string."""
        lm = ClaudeCodeLM()
        prompt = lm._prepare_prompt("Hello world", None)

        assert prompt == "Hello world"

    def test_prepare_from_messages(self) -> None:
        """Test preparing prompt from messages."""
        lm = ClaudeCodeLM()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]
        prompt = lm._prepare_prompt(None, messages)

        assert "User: Hello" in prompt
        assert "Assistant: Hi" in prompt
        assert "User: Bye" in prompt

    def test_prepare_with_system_in_messages(self) -> None:
        """Test system role in messages is handled."""
        lm = ClaudeCodeLM()  # No system_prompt set
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        prompt = lm._prepare_prompt(None, messages)

        # System message should be included if no system_prompt is set
        assert "System: Be helpful" in prompt
        assert "User: Hello" in prompt

    def test_prepare_empty_messages(self) -> None:
        """Test error on empty messages."""
        lm = ClaudeCodeLM()

        with pytest.raises(ValueError, match="Either prompt or messages"):
            lm._prepare_prompt(None, [])

        with pytest.raises(ValueError, match="Either prompt or messages"):
            lm._prepare_prompt(None, None)
