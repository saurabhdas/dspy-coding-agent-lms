"""DSPy Coding Agent LMs - Claude Code CLI integration for DSPy.

This library provides a DSPy-compatible language model that routes
requests through the Claude Code CLI, enabling agentic capabilities
within DSPy programs.

Features:
- Full DSPy LM compatibility
- Structured output via JSON schema (using Claude Code's --json-schema)
- Tool restriction support (allowed/disallowed tools)
- Async execution support
- Response caching (memory + disk)
- Transcript capture with cost tracking
- Support for both API key and OAuth authentication

RECOMMENDED: Use with dspy.JSONAdapter() for native JSON schema output.

Example:
    >>> from dspy_coding_agent_lms import ClaudeCodeLM
    >>> import dspy
    >>>
    >>> # Initialize the LM
    >>> lm = ClaudeCodeLM(model="sonnet", permission_mode="plan")
    >>>
    >>> # Configure with JSONAdapter for native JSON schema output (RECOMMENDED)
    >>> dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
    >>>
    >>> # Use DSPy as normal - outputs use Claude Code's --json-schema
    >>> predict = dspy.Predict("question -> answer")
    >>> result = predict(question="What is 2+2?")
    >>> print(result.answer)
    >>>
    >>> # Check transcript and cost
    >>> print(f"Total cost: ${lm.transcript.total_cost_usd():.4f}")

For more examples, see the examples/ directory.
"""

from .auth import AuthConfig, resolve_authentication
from .cache import ResponseCache, compute_cache_key
from .config import ClaudeCodeConfig, PermissionMode, StructuredOutputConfig
from .exceptions import (
    ClaudeCodeAuthError,
    ClaudeCodeError,
    ClaudeCodeExecutionError,
    ClaudeCodeParseError,
    ClaudeCodeTimeoutError,
    ClaudeCodeValidationError,
)
from .lm import ClaudeCodeLM
from .transcript import Transcript, TranscriptEntry

__version__ = "0.1.0"

__all__ = [  # noqa: RUF022
    # Version
    "__version__",
    # Main class
    "ClaudeCodeLM",
    # Configuration
    "ClaudeCodeConfig",
    "StructuredOutputConfig",
    "PermissionMode",
    # Transcript
    "Transcript",
    "TranscriptEntry",
    # Auth
    "AuthConfig",
    "resolve_authentication",
    # Cache
    "ResponseCache",
    "compute_cache_key",
    # Exceptions
    "ClaudeCodeError",
    "ClaudeCodeTimeoutError",
    "ClaudeCodeAuthError",
    "ClaudeCodeParseError",
    "ClaudeCodeValidationError",
    "ClaudeCodeExecutionError",
]
