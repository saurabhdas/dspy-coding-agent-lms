"""Tests for transcript module."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from dspy_coding_agent_lms.transcript import Transcript, TranscriptEntry


class TestTranscriptEntry:
    """Tests for TranscriptEntry dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic entry creation."""
        entry = TranscriptEntry(
            prompt="Test prompt",
            response={"type": "result", "result": "Test"},
            raw_response='{"type":"result","result":"Test"}',
        )

        assert entry.prompt == "Test prompt"
        assert entry.response["result"] == "Test"
        assert isinstance(entry.timestamp, datetime)

    def test_full_creation(self, mock_cli_response: dict[str, Any]) -> None:
        """Test entry creation with all fields."""
        entry = TranscriptEntry(
            prompt="Test prompt",
            response=mock_cli_response,
            raw_response=json.dumps(mock_cli_response),
            system_prompt="System prompt",
            append_system_prompt="Additional prompt",
            usage=mock_cli_response["usage"],
            models=["claude-sonnet-4-5-20250929"],
            session_id="test-session",
            duration_ms=1500,
            cost_usd=0.01,
        )

        assert entry.system_prompt == "System prompt"
        assert entry.append_system_prompt == "Additional prompt"
        assert entry.models == ["claude-sonnet-4-5-20250929"]
        assert entry.duration_ms == 1500
        assert entry.cost_usd == 0.01

    def test_to_dict(self) -> None:
        """Test converting entry to dictionary."""
        entry = TranscriptEntry(
            prompt="Test",
            response={"result": "OK"},
            raw_response="{}",
            session_id="test-123",
        )

        data = entry.to_dict()

        assert data["prompt"] == "Test"
        assert data["session_id"] == "test-123"
        assert "timestamp" in data

    def test_from_dict(self) -> None:
        """Test creating entry from dictionary."""
        data = {
            "prompt": "Test prompt",
            "response": {"result": "OK"},
            "raw_response": "{}",
            "session_id": "test-456",
            "timestamp": "2024-01-15T10:30:00",
        }

        entry = TranscriptEntry.from_dict(data)

        assert entry.prompt == "Test prompt"
        assert entry.session_id == "test-456"
        assert entry.timestamp.year == 2024


class TestTranscript:
    """Tests for Transcript class."""

    def test_empty_transcript(self) -> None:
        """Test empty transcript."""
        transcript = Transcript()

        assert len(transcript) == 0
        assert transcript.last is None
        assert transcript.first is None

    def test_add_entry(self) -> None:
        """Test adding entries."""
        transcript = Transcript()
        entry = TranscriptEntry(
            prompt="Test",
            response={"result": "OK"},
            raw_response="{}",
        )

        transcript.add_entry(entry)

        assert len(transcript) == 1
        assert transcript.last == entry
        assert transcript.first == entry

    def test_max_entries_limit(self) -> None:
        """Test max entries limit."""
        transcript = Transcript(max_entries=3)

        for i in range(5):
            entry = TranscriptEntry(
                prompt=f"Prompt {i}",
                response={"result": f"Result {i}"},
                raw_response="{}",
            )
            transcript.add_entry(entry)

        assert len(transcript) == 3
        # Oldest entries should be removed
        assert transcript.first.prompt == "Prompt 2"
        assert transcript.last.prompt == "Prompt 4"

    def test_iteration(self) -> None:
        """Test iterating over entries."""
        transcript = Transcript()
        for i in range(3):
            entry = TranscriptEntry(
                prompt=f"Prompt {i}",
                response={},
                raw_response="{}",
            )
            transcript.add_entry(entry)

        prompts = [e.prompt for e in transcript]
        assert prompts == ["Prompt 0", "Prompt 1", "Prompt 2"]

    def test_indexing(self) -> None:
        """Test index access."""
        transcript = Transcript()
        for i in range(3):
            entry = TranscriptEntry(
                prompt=f"Prompt {i}",
                response={},
                raw_response="{}",
            )
            transcript.add_entry(entry)

        assert transcript[0].prompt == "Prompt 0"
        assert transcript[1].prompt == "Prompt 1"
        assert transcript[-1].prompt == "Prompt 2"

    def test_total_cost_usd(self) -> None:
        """Test calculating total cost."""
        transcript = Transcript()
        for cost in [0.01, 0.02, 0.03]:
            entry = TranscriptEntry(
                prompt="Test",
                response={},
                raw_response="{}",
                cost_usd=cost,
            )
            transcript.add_entry(entry)

        assert transcript.total_cost_usd() == pytest.approx(0.06)

    def test_total_cost_from_response(self) -> None:
        """Test calculating total cost from response field."""
        transcript = Transcript()
        entry = TranscriptEntry(
            prompt="Test",
            response={"total_cost_usd": 0.05},
            raw_response="{}",
        )
        transcript.add_entry(entry)

        assert transcript.total_cost_usd() == pytest.approx(0.05)

    def test_total_tokens(self) -> None:
        """Test calculating total tokens."""
        transcript = Transcript()
        entry1 = TranscriptEntry(
            prompt="Test 1",
            response={},
            raw_response="{}",
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 100,
            },
        )
        entry2 = TranscriptEntry(
            prompt="Test 2",
            response={},
            raw_response="{}",
            usage={
                "input_tokens": 20,
                "output_tokens": 10,
                "cache_creation_input_tokens": 50,
            },
        )
        transcript.add_entry(entry1)
        transcript.add_entry(entry2)

        totals = transcript.total_tokens()
        assert totals["input"] == 30
        assert totals["output"] == 15
        assert totals["cache_read"] == 100
        assert totals["cache_creation"] == 50

    def test_total_duration(self) -> None:
        """Test calculating total duration."""
        transcript = Transcript()
        for duration in [100, 200, 300]:
            entry = TranscriptEntry(
                prompt="Test",
                response={},
                raw_response="{}",
                duration_ms=duration,
            )
            transcript.add_entry(entry)

        assert transcript.total_duration_ms() == 600
        assert transcript.average_duration_ms() == 200.0

    def test_to_json(self) -> None:
        """Test exporting to JSON."""
        transcript = Transcript()
        entry = TranscriptEntry(
            prompt="Test",
            response={"result": "OK"},
            raw_response="{}",
            session_id="test-json",
        )
        transcript.add_entry(entry)

        json_str = transcript.to_json()
        data = json.loads(json_str)

        assert len(data) == 1
        assert data[0]["prompt"] == "Test"
        assert data[0]["session_id"] == "test-json"

    def test_from_json(self) -> None:
        """Test creating from JSON."""
        json_str = json.dumps([
            {
                "prompt": "Test 1",
                "response": {"result": "OK"},
                "raw_response": "{}",
                "timestamp": "2024-01-15T10:00:00",
            },
            {
                "prompt": "Test 2",
                "response": {"result": "OK 2"},
                "raw_response": "{}",
                "timestamp": "2024-01-15T10:01:00",
            },
        ])

        transcript = Transcript.from_json(json_str)

        assert len(transcript) == 2
        assert transcript[0].prompt == "Test 1"
        assert transcript[1].prompt == "Test 2"

    def test_save_and_load(self, tmp_path: Any) -> None:
        """Test saving and loading from file."""
        filepath = tmp_path / "transcript.json"

        # Save
        transcript = Transcript()
        entry = TranscriptEntry(
            prompt="File test",
            response={"result": "Saved"},
            raw_response="{}",
        )
        transcript.add_entry(entry)
        transcript.save_to_file(str(filepath))

        # Load
        loaded = Transcript.load_from_file(str(filepath))

        assert len(loaded) == 1
        assert loaded[0].prompt == "File test"

    def test_clear(self) -> None:
        """Test clearing transcript."""
        transcript = Transcript()
        for i in range(3):
            entry = TranscriptEntry(
                prompt=f"Test {i}",
                response={},
                raw_response="{}",
            )
            transcript.add_entry(entry)

        assert len(transcript) == 3
        transcript.clear()
        assert len(transcript) == 0

    def test_get_by_session_id(self) -> None:
        """Test filtering by session ID."""
        transcript = Transcript()
        for i in range(3):
            entry = TranscriptEntry(
                prompt=f"Test {i}",
                response={},
                raw_response="{}",
                session_id="session-a" if i % 2 == 0 else "session-b",
            )
            transcript.add_entry(entry)

        session_a = transcript.get_by_session_id("session-a")
        assert len(session_a) == 2
        assert all(e.session_id == "session-a" for e in session_a)
