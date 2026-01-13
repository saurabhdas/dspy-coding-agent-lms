"""Tests for response module."""

from __future__ import annotations

import json
from typing import Any

import pytest

from dspy_coding_agent_lms.exceptions import ClaudeCodeParseError
from dspy_coding_agent_lms.response import (
    extract_structured_output,
    extract_text_result,
    format_openai_response,
    get_usage_summary,
    parse_json_response,
)


class TestParseJsonResponse:
    """Tests for parse_json_response function."""

    def test_parse_simple_json(self) -> None:
        """Test parsing simple JSON response."""
        raw = '{"type":"result","result":"Hello"}'
        parsed = parse_json_response(raw)

        assert parsed["type"] == "result"
        assert parsed["result"] == "Hello"

    def test_parse_full_response(self, mock_cli_response: dict[str, Any]) -> None:
        """Test parsing full CLI response."""
        raw = json.dumps(mock_cli_response)
        parsed = parse_json_response(raw)

        assert parsed["type"] == "result"
        assert parsed["subtype"] == "success"
        assert parsed["session_id"] == "test-session-123"
        assert parsed["usage"]["input_tokens"] == 10

    def test_parse_ndjson(self, mock_ndjson_response: str) -> None:
        """Test parsing newline-delimited JSON."""
        parsed = parse_json_response(mock_ndjson_response)

        assert parsed["type"] == "result"
        assert parsed["result"] == "Final result from stream"
        assert parsed["session_id"] == "stream-session"

    def test_parse_ndjson_with_whitespace(self) -> None:
        """Test parsing NDJSON with extra whitespace."""
        raw = '''{"type":"init"}

{"type":"result","subtype":"success","result":"Done"}

'''
        parsed = parse_json_response(raw)

        assert parsed["type"] == "result"
        assert parsed["result"] == "Done"

    def test_parse_empty_response(self) -> None:
        """Test parsing empty response raises error."""
        with pytest.raises(ClaudeCodeParseError, match="Empty response"):
            parse_json_response("")

        with pytest.raises(ClaudeCodeParseError, match="Empty response"):
            parse_json_response("   \n\n   ")

    def test_parse_invalid_json(self) -> None:
        """Test parsing invalid JSON raises error."""
        with pytest.raises(ClaudeCodeParseError, match="Failed to parse"):
            parse_json_response("not json at all")


class TestFormatOpenAIResponse:
    """Tests for format_openai_response function."""

    def test_format_success_response(self, mock_cli_response: dict[str, Any]) -> None:
        """Test formatting successful response."""
        result = format_openai_response(mock_cli_response)

        assert len(result) == 1
        assert result[0]["text"] == "Test response from Claude Code"
        assert result[0]["finish_reason"] == "stop"
        assert result[0]["logprobs"] is None

    def test_format_with_usage(self, mock_cli_response: dict[str, Any]) -> None:
        """Test formatting includes usage info."""
        result = format_openai_response(mock_cli_response)

        assert "usage" in result[0]
        usage = result[0]["usage"]
        # input_tokens + cache_read + cache_creation = 10 + 100 + 50 = 160
        assert usage["prompt_tokens"] == 160
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 165

    def test_format_with_cost(self, mock_cli_response: dict[str, Any]) -> None:
        """Test formatting includes cost."""
        result = format_openai_response(mock_cli_response)

        assert result[0]["cost_usd"] == 0.01

    def test_format_structured_output(self, mock_structured_response: dict[str, Any]) -> None:
        """Test formatting structured output."""
        result = format_openai_response(mock_structured_response)

        # Structured output should be JSON stringified
        text = result[0]["text"]
        parsed = json.loads(text)
        assert parsed["name"] == "Test"
        assert parsed["value"] == 42
        assert parsed["items"] == ["a", "b", "c"]

    def test_format_error_response(self, mock_error_response: dict[str, Any]) -> None:
        """Test formatting error response."""
        result = format_openai_response(mock_error_response)

        assert result[0]["finish_reason"] == "error"
        assert "Error" in result[0]["text"]


class TestExtractStructuredOutput:
    """Tests for extract_structured_output function."""

    def test_extract_present(self, mock_structured_response: dict[str, Any]) -> None:
        """Test extracting when structured output is present."""
        output = extract_structured_output(mock_structured_response)

        assert output is not None
        assert output["name"] == "Test"
        assert output["value"] == 42

    def test_extract_absent(self, mock_cli_response: dict[str, Any]) -> None:
        """Test extracting when structured output is absent."""
        output = extract_structured_output(mock_cli_response)

        assert output is None


class TestExtractTextResult:
    """Tests for extract_text_result function."""

    def test_extract_text(self, mock_cli_response: dict[str, Any]) -> None:
        """Test extracting text result."""
        text = extract_text_result(mock_cli_response)

        assert text == "Test response from Claude Code"

    def test_extract_empty(self) -> None:
        """Test extracting from response without result."""
        text = extract_text_result({})

        assert text == ""


class TestGetUsageSummary:
    """Tests for get_usage_summary function."""

    def test_get_usage(self, mock_cli_response: dict[str, Any]) -> None:
        """Test getting usage summary."""
        summary = get_usage_summary(mock_cli_response)

        assert summary["input_tokens"] == 10
        assert summary["output_tokens"] == 5
        assert summary["cache_read_tokens"] == 100
        assert summary["cache_creation_tokens"] == 50
        assert summary["total_input"] == 160
        assert summary["total_output"] == 5

    def test_get_usage_empty(self) -> None:
        """Test getting usage from response without usage."""
        summary = get_usage_summary({})

        assert summary["input_tokens"] == 0
        assert summary["output_tokens"] == 0
        assert summary["total_input"] == 0
        assert summary["total_output"] == 0
