"""Configuration dataclasses for Claude Code LM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

PermissionMode = Literal[
    "acceptEdits",
    "bypassPermissions",
    "default",
    "delegate",
    "dontAsk",
    "plan",
]

OutputFormat = Literal["json", "stream-json", "text"]


@dataclass(frozen=True)
class ClaudeCodeConfig:
    """Configuration for Claude Code CLI invocation.

    This dataclass holds all configuration options that control how
    the Claude Code CLI is invoked. It is immutable (frozen) to ensure
    configuration consistency throughout a session.

    Attributes:
        model: Model to use (sonnet, opus, haiku, or full model name).
        system_prompt: Completely replaces Claude Code's default system prompt.
        append_system_prompt: Adds to Claude Code's default system prompt.
        permission_mode: Controls how Claude Code handles permission requests.
        dangerously_skip_permissions: Bypasses all permission prompts.
        allowed_tools: List of tool patterns that can execute without prompts.
        disallowed_tools: List of tool patterns that are blocked.
        working_directory: Working directory for Claude Code execution.
        add_dirs: Additional directories to allow access.
        timeout_seconds: Maximum execution time in seconds.
        verbose: Enable verbose output.
        output_format: Output format for CLI response.
        max_budget_usd: Maximum budget in USD for the request.
    """

    model: str = "sonnet"
    system_prompt: str | None = None
    append_system_prompt: str | None = None
    permission_mode: PermissionMode = "plan"
    dangerously_skip_permissions: bool = False
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    disallowed_tools: tuple[str, ...] = field(default_factory=tuple)
    working_directory: str | None = None
    add_dirs: tuple[str, ...] = field(default_factory=tuple)
    timeout_seconds: float = 300.0
    verbose: bool = False
    output_format: OutputFormat = "json"
    max_budget_usd: float | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        if self.max_budget_usd is not None and self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive if specified")

    def with_updates(self, **kwargs: object) -> ClaudeCodeConfig:
        """Create a new config with updated values.

        Since ClaudeCodeConfig is frozen, this method creates a new instance
        with the specified updates applied.

        Args:
            **kwargs: Configuration values to update.

        Returns:
            A new ClaudeCodeConfig instance with updates applied.
        """
        from dataclasses import asdict

        current = asdict(self)
        current.update(kwargs)
        return ClaudeCodeConfig(**current)


@dataclass(frozen=True)
class StructuredOutputConfig:
    """Configuration for JSON schema-based structured output.

    When using structured output, Claude Code will validate responses
    against the provided JSON schema and return the result in the
    `structured_output` field of the response.

    Attributes:
        schema: JSON schema defining the expected output structure.
        strict: Whether to enforce strict schema validation.
    """

    schema: dict[str, object]
    strict: bool = True

    def to_json_string(self) -> str:
        """Convert schema to JSON string for CLI argument.

        Returns:
            JSON string representation of the schema.
        """
        return json.dumps(self.schema)

    @classmethod
    def from_json_string(cls, json_string: str, strict: bool = True) -> StructuredOutputConfig:
        """Create config from JSON string.

        Args:
            json_string: JSON string representation of the schema.
            strict: Whether to enforce strict schema validation.

        Returns:
            StructuredOutputConfig instance.
        """
        schema = json.loads(json_string)
        return cls(schema=schema, strict=strict)
