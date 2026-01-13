"""Transcript capture and storage for Claude Code interactions."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TranscriptEntry:
    """A single interaction entry in the transcript.

    Captures all relevant information about a single interaction with
    Claude Code, including the prompt, response, and metadata.

    Attributes:
        prompt: The user prompt sent to Claude Code.
        response: The parsed response from Claude Code.
        raw_response: The raw string response from the CLI.
        system_prompt: The system prompt used (if custom).
        append_system_prompt: Additional system prompt appended (if any).
        usage: Token usage information.
        models: List of models used in the response.
        session_id: The session ID from Claude Code.
        duration_ms: Response duration in milliseconds.
        cost_usd: Cost of the interaction in USD.
        timestamp: When the interaction occurred.
    """

    prompt: str
    response: dict[str, Any]
    raw_response: str
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    usage: dict[str, Any] | None = None
    models: list[str] | None = None
    session_id: str | None = None
    duration_ms: int | None = None
    cost_usd: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the entry.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
            "append_system_prompt": self.append_system_prompt,
            "response": self.response,
            "raw_response": self.raw_response,
            "usage": self.usage,
            "models": self.models,
            "session_id": self.session_id,
            "duration_ms": self.duration_ms,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptEntry:
        """Create entry from dictionary.

        Args:
            data: Dictionary representation of the entry.

        Returns:
            TranscriptEntry instance.
        """
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            prompt=data.get("prompt", ""),
            response=data.get("response", {}),
            raw_response=data.get("raw_response", ""),
            system_prompt=data.get("system_prompt"),
            append_system_prompt=data.get("append_system_prompt"),
            usage=data.get("usage"),
            models=data.get("models"),
            session_id=data.get("session_id"),
            duration_ms=data.get("duration_ms"),
            cost_usd=data.get("cost_usd"),
            timestamp=timestamp,
        )


@dataclass
class Transcript:
    """Complete transcript of all interactions.

    Provides a record of all interactions with Claude Code during a session,
    with methods for aggregating statistics and exporting data.

    Attributes:
        entries: List of transcript entries.
        max_entries: Maximum number of entries to keep (oldest removed first).
    """

    entries: list[TranscriptEntry] = field(default_factory=list)
    max_entries: int = 1000

    def add_entry(self, entry: TranscriptEntry) -> None:
        """Add an entry to the transcript.

        If the transcript exceeds max_entries, the oldest entries are removed.

        Args:
            entry: The transcript entry to add.
        """
        self.entries.append(entry)

        # Trim old entries if over limit
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def __iter__(self) -> Iterator[TranscriptEntry]:
        """Iterate over transcript entries."""
        return iter(self.entries)

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self.entries)

    def __getitem__(self, index: int) -> TranscriptEntry:
        """Get entry by index."""
        return self.entries[index]

    @property
    def last(self) -> TranscriptEntry | None:
        """Get the most recent entry.

        Returns:
            The last entry, or None if transcript is empty.
        """
        return self.entries[-1] if self.entries else None

    @property
    def first(self) -> TranscriptEntry | None:
        """Get the first entry.

        Returns:
            The first entry, or None if transcript is empty.
        """
        return self.entries[0] if self.entries else None

    def total_cost_usd(self) -> float:
        """Calculate total cost across all entries.

        Returns:
            Total cost in USD.
        """
        total = 0.0
        for entry in self.entries:
            if entry.cost_usd is not None:
                total += entry.cost_usd
            elif entry.response and "total_cost_usd" in entry.response:
                total += entry.response["total_cost_usd"]
        return total

    def total_tokens(self) -> dict[str, int]:
        """Calculate total token usage across all entries.

        Returns:
            Dictionary with token usage breakdown.
        """
        totals = {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_creation": 0,
        }
        for entry in self.entries:
            if entry.usage:
                totals["input"] += entry.usage.get("input_tokens", 0)
                totals["output"] += entry.usage.get("output_tokens", 0)
                totals["cache_read"] += entry.usage.get("cache_read_input_tokens", 0)
                totals["cache_creation"] += entry.usage.get(
                    "cache_creation_input_tokens", 0
                )
        return totals

    def total_duration_ms(self) -> int:
        """Calculate total duration across all entries.

        Returns:
            Total duration in milliseconds.
        """
        total = 0
        for entry in self.entries:
            if entry.duration_ms is not None:
                total += entry.duration_ms
        return total

    def average_duration_ms(self) -> float:
        """Calculate average duration per entry.

        Returns:
            Average duration in milliseconds, or 0 if no entries.
        """
        entries_with_duration = [e for e in self.entries if e.duration_ms is not None]
        if not entries_with_duration:
            return 0.0
        return sum(e.duration_ms or 0 for e in entries_with_duration) / len(
            entries_with_duration
        )

    def to_json(self, indent: int = 2) -> str:
        """Export transcript as JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string representation of the transcript.
        """
        return json.dumps(
            [e.to_dict() for e in self.entries],
            indent=indent,
            default=str,
        )

    @classmethod
    def from_json(cls, json_string: str, max_entries: int = 1000) -> Transcript:
        """Create transcript from JSON string.

        Args:
            json_string: JSON string representation of transcript.
            max_entries: Maximum entries to keep.

        Returns:
            Transcript instance.
        """
        data = json.loads(json_string)
        entries = [TranscriptEntry.from_dict(d) for d in data]
        return cls(entries=entries, max_entries=max_entries)

    def save_to_file(self, filepath: str) -> None:
        """Save transcript to a JSON file.

        Args:
            filepath: Path to save the transcript.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load_from_file(cls, filepath: str, max_entries: int = 1000) -> Transcript:
        """Load transcript from a JSON file.

        Args:
            filepath: Path to load the transcript from.
            max_entries: Maximum entries to keep.

        Returns:
            Transcript instance.
        """
        with open(filepath, encoding="utf-8") as f:
            return cls.from_json(f.read(), max_entries=max_entries)

    def clear(self) -> None:
        """Clear all entries from the transcript."""
        self.entries.clear()

    def get_by_session_id(self, session_id: str) -> list[TranscriptEntry]:
        """Get all entries for a specific session ID.

        Args:
            session_id: The session ID to filter by.

        Returns:
            List of entries matching the session ID.
        """
        return [e for e in self.entries if e.session_id == session_id]
