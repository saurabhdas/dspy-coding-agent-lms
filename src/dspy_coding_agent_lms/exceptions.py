"""Custom exceptions for dspy-coding-agent-lms."""

from __future__ import annotations


class ClaudeCodeError(Exception):
    """Base exception for Claude Code LM errors.

    All exceptions raised by this library inherit from this class,
    making it easy to catch all library-specific errors.
    """


class ClaudeCodeTimeoutError(ClaudeCodeError):
    """Raised when CLI execution times out.

    This occurs when the Claude Code CLI does not respond within
    the configured timeout period.
    """


class ClaudeCodeAuthError(ClaudeCodeError):
    """Raised when authentication fails.

    This can occur when:
    - No authentication credentials are provided
    - API key or OAuth token is invalid
    - Authentication is rejected by Claude Code
    """


class ClaudeCodeParseError(ClaudeCodeError):
    """Raised when response parsing fails.

    This occurs when the CLI output cannot be parsed as valid JSON
    or does not match the expected response structure.
    """


class ClaudeCodeValidationError(ClaudeCodeError):
    """Raised when structured output validation fails.

    This occurs when using --json-schema and the response does not
    conform to the specified schema.
    """


class ClaudeCodeExecutionError(ClaudeCodeError):
    """Raised when the CLI execution fails with a non-zero exit code.

    Attributes:
        exit_code: The exit code returned by the CLI
        stderr: The stderr output from the CLI
    """

    def __init__(self, message: str, exit_code: int, stderr: str = "") -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr = stderr
