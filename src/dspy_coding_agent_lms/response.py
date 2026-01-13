"""Response parsing and formatting for Claude Code CLI output."""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from .exceptions import ClaudeCodeParseError

logger = logging.getLogger(__name__)


class UsageInfo(TypedDict, total=False):
    """Token usage information from Claude Code response."""

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class ParsedResponse(TypedDict, total=False):
    """Parsed Claude Code CLI response structure."""

    type: str
    subtype: str
    is_error: bool
    duration_ms: int
    result: str
    session_id: str
    total_cost_usd: float
    usage: UsageInfo
    structured_output: dict[str, Any]
    modelUsage: dict[str, Any]


class OpenAICompatibleResponse(TypedDict, total=False):
    """OpenAI-compatible response format for DSPy."""

    text: str
    logprobs: None
    finish_reason: str
    usage: dict[str, int]
    cost_usd: float


def parse_json_response(raw_response: str) -> ParsedResponse:
    """Parse JSON response from Claude Code CLI.

    Handles both single JSON responses and newline-delimited JSON (NDJSON)
    format used by stream-json output mode.

    Args:
        raw_response: Raw string output from the CLI.

    Returns:
        Parsed response dictionary.

    Raises:
        ClaudeCodeParseError: If the response cannot be parsed.
    """
    if not raw_response.strip():
        raise ClaudeCodeParseError("Empty response from CLI")

    # Try parsing as single JSON first
    try:
        parsed: ParsedResponse = json.loads(raw_response)
        return parsed
    except json.JSONDecodeError:
        pass

    # Handle stream-json format (newline-delimited JSON)
    # Look for the final "result" message
    lines = raw_response.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if parsed.get("type") == "result":
                return parsed
        except json.JSONDecodeError:
            continue

    # If no result found, try to return the last valid JSON
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            return parsed
        except json.JSONDecodeError:
            continue

    raise ClaudeCodeParseError(
        f"Failed to parse CLI response as JSON. Raw response: {raw_response[:500]}"
    )


def format_openai_response(
    parsed: ParsedResponse,
) -> list[OpenAICompatibleResponse]:
    """Format Claude Code response to OpenAI-compatible format.

    DSPy expects responses in a specific format. This function converts
    the Claude Code CLI output to that format.

    Args:
        parsed: Parsed response from Claude Code CLI.

    Returns:
        List of completion results in OpenAI-compatible format.
    """
    # Extract the result text
    if parsed.get("type") == "result":
        # Check for structured output first
        if "structured_output" in parsed and parsed["structured_output"] is not None:
            text = json.dumps(parsed["structured_output"])
        else:
            text = parsed.get("result", "")
    else:
        # Fallback: stringify the entire response
        text = json.dumps(parsed) if isinstance(parsed, dict) else str(parsed)

    # Determine finish reason
    if parsed.get("is_error"):
        finish_reason = "error"
    elif parsed.get("subtype") == "success":
        finish_reason = "stop"
    else:
        finish_reason = "stop"  # Default to stop

    # Build OpenAI-compatible response
    response: OpenAICompatibleResponse = {
        "text": text,
        "logprobs": None,
        "finish_reason": finish_reason,
    }

    # Add usage metadata if available
    usage = parsed.get("usage")
    if usage:
        input_tokens = (
            usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
        )
        output_tokens = usage.get("output_tokens", 0)
        response["usage"] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    # Add cost if available
    if "total_cost_usd" in parsed:
        response["cost_usd"] = parsed["total_cost_usd"]

    return [response]


def extract_structured_output(
    parsed: ParsedResponse,
) -> dict[str, Any] | None:
    """Extract structured output from Claude Code response.

    When using --json-schema, Claude Code returns the validated output
    in the structured_output field.

    Args:
        parsed: Parsed response from Claude Code CLI.

    Returns:
        The structured output dictionary, or None if not present.
    """
    return parsed.get("structured_output")


def extract_text_result(parsed: ParsedResponse) -> str:
    """Extract the text result from Claude Code response.

    Args:
        parsed: Parsed response from Claude Code CLI.

    Returns:
        The text result string.
    """
    return parsed.get("result", "")


def get_usage_summary(parsed: ParsedResponse) -> dict[str, int]:
    """Get a summary of token usage from the response.

    Args:
        parsed: Parsed response from Claude Code CLI.

    Returns:
        Dictionary with token usage breakdown.
    """
    usage = parsed.get("usage", {})
    return {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
        "total_input": (
            usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
        ),
        "total_output": usage.get("output_tokens", 0),
    }
