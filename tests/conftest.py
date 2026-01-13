"""Pytest fixtures for dspy-coding-agent-lms tests."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_cli_response() -> dict[str, Any]:
    """Standard successful CLI response."""
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 1500,
        "result": "Test response from Claude Code",
        "session_id": "test-session-123",
        "total_cost_usd": 0.01,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
        },
        "modelUsage": {
            "claude-sonnet-4-5-20250929": {
                "input_tokens": 10,
                "output_tokens": 5,
            }
        },
    }


@pytest.fixture
def mock_cli_response_json(mock_cli_response: dict[str, Any]) -> str:
    """Standard successful CLI response as JSON string."""
    return json.dumps(mock_cli_response)


@pytest.fixture
def mock_structured_response() -> dict[str, Any]:
    """CLI response with structured output."""
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 2000,
        "result": "",
        "session_id": "test-session-456",
        "total_cost_usd": 0.02,
        "structured_output": {
            "name": "Test",
            "value": 42,
            "items": ["a", "b", "c"],
        },
        "usage": {
            "input_tokens": 20,
            "output_tokens": 10,
        },
    }


@pytest.fixture
def mock_structured_response_json(mock_structured_response: dict[str, Any]) -> str:
    """Structured response as JSON string."""
    return json.dumps(mock_structured_response)


@pytest.fixture
def mock_error_response() -> dict[str, Any]:
    """CLI response indicating an error."""
    return {
        "type": "result",
        "subtype": "error",
        "is_error": True,
        "duration_ms": 500,
        "result": "Error: Something went wrong",
        "session_id": "test-session-error",
        "total_cost_usd": 0.001,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 2,
        },
    }


@pytest.fixture
def mock_ndjson_response() -> str:
    """Newline-delimited JSON (stream-json) response."""
    messages = [
        {"type": "system", "data": "init"},
        {"type": "message", "content": "Processing..."},
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "Final result from stream",
            "session_id": "stream-session",
            "total_cost_usd": 0.015,
            "usage": {"input_tokens": 15, "output_tokens": 8},
        },
    ]
    return "\n".join(json.dumps(m) for m in messages)


@pytest.fixture
def mock_subprocess(mocker: Any) -> MagicMock:
    """Mock subprocess.run for CLI tests."""
    mock = mocker.patch("subprocess.run")
    mock.return_value = MagicMock(
        returncode=0,
        stdout='{"type":"result","subtype":"success","result":"OK"}',
        stderr="",
    )
    return mock


@pytest.fixture
def mock_async_subprocess(mocker: Any) -> MagicMock:
    """Mock asyncio.create_subprocess_exec for async CLI tests."""
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = MagicMock(
        return_value=(
            b'{"type":"result","subtype":"success","result":"OK async"}',
            b"",
        )
    )

    mock = mocker.patch("asyncio.create_subprocess_exec")
    mock.return_value = mock_process
    return mock


@pytest.fixture
def temp_cache_dir(tmp_path: Any) -> str:
    """Temporary directory for cache tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def sample_json_schema() -> dict[str, Any]:
    """Sample JSON schema for structured output tests."""
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "integer"},
            "explanation": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["answer"],
    }
