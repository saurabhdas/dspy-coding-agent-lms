"""Main ClaudeCodeLM class - DSPy LM integration for Claude Code CLI."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Literal, cast

import dspy  # type: ignore[import-untyped]

from .auth import resolve_authentication
from .cache import ResponseCache, compute_cache_key
from .cli import ClaudeCodeCLI
from .config import ClaudeCodeConfig, PermissionMode
from .exceptions import ClaudeCodeError
from .response import ParsedResponse, parse_json_response
from .transcript import Transcript, TranscriptEntry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Usage:
    """Token usage information compatible with DSPy's expectations."""

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        # Store any additional usage data
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __iter__(self) -> Iterator[tuple[str, int]]:
        """Allow dict(usage) to work."""
        yield "prompt_tokens", self.prompt_tokens
        yield "completion_tokens", self.completion_tokens
        yield "total_tokens", self.total_tokens


class Message:
    """Message object compatible with OpenAI response format."""

    def __init__(self, content: str, role: str = "assistant") -> None:
        self.content = content
        self.role = role
        self.tool_calls = None


class Choice:
    """Choice object compatible with OpenAI response format."""

    def __init__(self, message: Message, index: int = 0, finish_reason: str = "stop") -> None:
        self.message = message
        self.index = index
        self.finish_reason = finish_reason


class LMResponse:
    """Response object compatible with DSPy's BaseLM expectations.

    This class mimics the OpenAI API response format that DSPy expects.
    """

    def __init__(
        self,
        text: str,
        model: str,
        usage: Usage,
        finish_reason: str = "stop",
    ) -> None:
        self.choices = [Choice(Message(text), finish_reason=finish_reason)]
        self.model = model
        self.usage = usage
        self._hidden_params: dict[str, Any] = {}

    def set_cost(self, cost: float) -> None:
        """Set the response cost for logging."""
        self._hidden_params["response_cost"] = cost

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for caching."""
        return {
            "text": self.choices[0].message.content,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "finish_reason": self.choices[0].finish_reason,
            "hidden_params": self._hidden_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LMResponse:
        """Deserialize from dict."""
        usage = Usage(
            prompt_tokens=data["usage"]["prompt_tokens"],
            completion_tokens=data["usage"]["completion_tokens"],
            total_tokens=data["usage"]["total_tokens"],
        )
        response = cls(
            text=data["text"],
            model=data["model"],
            usage=usage,
            finish_reason=data.get("finish_reason", "stop"),
        )
        response._hidden_params = data.get("hidden_params", {})
        return response


class ClaudeCodeLM(dspy.BaseLM):  # type: ignore[misc]
    """A DSPy-compatible LM that routes requests through Claude Code CLI.

    This class provides a DSPy-compatible interface for the Claude Code CLI,
    enabling the use of Claude Code's agentic capabilities (file editing,
    bash execution, tool use) within DSPy programs.

    Unlike traditional API-based LMs, ClaudeCodeLM invokes the local Claude
    Code CLI for each request, allowing access to the full range of Claude
    Code features including permission management, tool restrictions, and
    structured output via JSON schemas.

    RECOMMENDED: Use with dspy.JSONAdapter() for native JSON schema output.
    This uses Claude Code's --json-schema flag for reliable structured output
    instead of string parsing with [[ ## field ## ]] markers.

    Example:
        >>> from dspy_coding_agent_lms import ClaudeCodeLM
        >>> import dspy
        >>>
        >>> lm = ClaudeCodeLM(model="sonnet", permission_mode="plan")
        >>> # Use JSONAdapter for native JSON schema output (recommended)
        >>> dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
        >>>
        >>> # Now DSPy programs will use Claude Code CLI with JSON schema
        >>> predict = dspy.Predict("question -> answer")
        >>> result = predict(question="What is 2+2?")

    Attributes:
        config: Configuration for Claude Code CLI invocation.
        auth: Authentication configuration.
        cli: CLI executor instance.
        transcript: Transcript of all interactions.
        history: History of all calls (DSPy requirement).
    """

    def __init__(
        self,
        model: str = "sonnet",
        model_type: Literal["chat"] = "chat",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        cache: bool = True,
        cache_dir: str | None = None,
        cache_ttl: float | None = None,
        # Claude Code specific options
        system_prompt: str | None = None,
        append_system_prompt: str | None = None,
        permission_mode: PermissionMode = "plan",
        dangerously_skip_permissions: bool = False,
        allowed_tools: list[str] | tuple[str, ...] | None = None,
        disallowed_tools: list[str] | tuple[str, ...] | None = None,
        working_directory: str | None = None,
        add_dirs: list[str] | tuple[str, ...] | None = None,
        timeout_seconds: float = 300.0,
        max_budget_usd: float | None = None,
        # Authentication
        anthropic_api_key: str | None = None,
        oauth_token: str | None = None,
        # Advanced options
        claude_binary: str = "claude",
        verbose: bool = False,
        capture_transcript: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize Claude Code LM.

        Args:
            model: Model alias (sonnet, opus, haiku) or full model name.
            model_type: Always "chat" for Claude Code.
            temperature: Sampling temperature (passed to underlying model).
            max_tokens: Maximum output tokens.
            cache: Enable response caching.
            cache_dir: Directory for cache storage.
            cache_ttl: Time-to-live for cache entries in seconds.
            system_prompt: Override Claude Code's default system prompt.
            append_system_prompt: Append to default system prompt.
            permission_mode: Permission mode for tool use.
            dangerously_skip_permissions: Bypass all permission checks.
            allowed_tools: List of allowed tool patterns.
            disallowed_tools: List of disallowed tool patterns.
            working_directory: Working directory for Claude Code.
            add_dirs: Additional directories to allow access.
            timeout_seconds: Command timeout in seconds.
            max_budget_usd: Maximum budget in USD for requests.
            anthropic_api_key: Explicit API key.
            oauth_token: OAuth token for Pro/Max subscribers.
            claude_binary: Path to claude binary.
            verbose: Enable verbose output.
            capture_transcript: Capture full interaction transcript.
            **kwargs: Additional parameters.
        """
        # Map model aliases to full model names for litellm compatibility
        # This ensures DSPy's JSONAdapter recognizes our model as supporting response_format
        model_name_map = {
            "sonnet": "anthropic/claude-sonnet-4-5-20250929",
            "opus": "anthropic/claude-opus-4-20250514",
            "haiku": "anthropic/claude-haiku-4-5-20251001",
        }
        litellm_model = model_name_map.get(model, f"anthropic/{model}")

        # Call parent BaseLM init with litellm-compatible model name
        super().__init__(
            model=litellm_model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **kwargs,
        )

        # Store the raw model name/alias for CLI
        self._raw_model = model

        # Build configuration
        self.config = ClaudeCodeConfig(
            model=model,
            system_prompt=system_prompt,
            append_system_prompt=append_system_prompt,
            permission_mode=permission_mode,
            dangerously_skip_permissions=dangerously_skip_permissions,
            allowed_tools=tuple(allowed_tools) if allowed_tools else (),
            disallowed_tools=tuple(disallowed_tools) if disallowed_tools else (),
            working_directory=working_directory,
            add_dirs=tuple(add_dirs) if add_dirs else (),
            timeout_seconds=timeout_seconds,
            verbose=verbose,
            max_budget_usd=max_budget_usd,
        )

        # Authentication
        self.auth = resolve_authentication(
            anthropic_api_key=anthropic_api_key,
            oauth_token=oauth_token,
        )

        # CLI executor
        self.cli = ClaudeCodeCLI(
            binary=claude_binary,
            config=self.config,
            auth=self.auth,
        )

        # Our own caching layer (separate from DSPy's cache)
        self._cache_dir = cache_dir
        self._cache_ttl = cache_ttl
        self._our_cache = ResponseCache(cache_dir, default_ttl=cache_ttl) if cache else None

        # Transcript capture
        self._capture_transcript = capture_transcript
        self._transcript = Transcript()

    @property
    def transcript(self) -> Transcript:
        """Access the interaction transcript.

        Returns:
            The transcript containing all interactions.
        """
        return self._transcript

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LMResponse:
        """Execute a request through Claude Code CLI (synchronous).

        This is the main method for executing requests. It handles prompt
        preparation, caching, CLI execution, response parsing, and
        transcript capture.

        Args:
            prompt: Simple text prompt.
            messages: Chat messages in OpenAI format.
            json_schema: JSON schema for structured output.
            **kwargs: Additional CLI options including response_format.

        Returns:
            LMResponse object compatible with DSPy's expectations.

        Raises:
            ClaudeCodeError: If CLI execution fails.
            ValueError: If neither prompt nor messages is provided.
        """
        # Handle response_format from DSPy's JSONAdapter
        json_schema = self._extract_json_schema(json_schema, kwargs)

        # Convert messages to prompt if needed
        effective_prompt = self._prepare_prompt(prompt, messages)

        # Check our cache
        config_dict = asdict(self.config)
        cache_key = compute_cache_key(effective_prompt, json_schema, config_dict, **kwargs)

        if self._our_cache:
            cached_data = self._our_cache.get(cache_key)
            if cached_data is not None:
                logger.debug("Cache hit for prompt")
                return LMResponse.from_dict(cached_data)

        # Execute CLI
        try:
            raw_response = self.cli.execute(
                prompt=effective_prompt,
                json_schema=json_schema,
                **kwargs,
            )
        except Exception as e:
            raise ClaudeCodeError(f"CLI execution failed: {e}") from e

        # Parse response
        parsed = parse_json_response(raw_response)

        # Capture transcript
        if self._capture_transcript:
            self._add_transcript_entry(
                prompt=effective_prompt,
                parsed=parsed,
                raw_response=raw_response,
            )

        # Build LMResponse for DSPy
        response = self._build_lm_response(parsed)

        # Cache result (serialize to dict for JSON compatibility)
        if self._our_cache:
            self._our_cache.set(cache_key, response.to_dict())

        return response

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LMResponse:
        """Execute a request through Claude Code CLI (asynchronous).

        Uses asyncio.subprocess for non-blocking execution, enabling
        concurrent requests.

        Args:
            prompt: Simple text prompt.
            messages: Chat messages in OpenAI format.
            json_schema: JSON schema for structured output.
            **kwargs: Additional CLI options including response_format.

        Returns:
            LMResponse object compatible with DSPy's expectations.

        Raises:
            ClaudeCodeError: If CLI execution fails.
            ClaudeCodeTimeoutError: If the command times out.
        """
        # Handle response_format from DSPy's JSONAdapter
        json_schema = self._extract_json_schema(json_schema, kwargs)

        effective_prompt = self._prepare_prompt(prompt, messages)

        # Check our cache
        config_dict = asdict(self.config)
        cache_key = compute_cache_key(effective_prompt, json_schema, config_dict, **kwargs)

        if self._our_cache:
            cached_data = self._our_cache.get(cache_key)
            if cached_data is not None:
                logger.debug("Cache hit for prompt (async)")
                return LMResponse.from_dict(cached_data)

        # Execute CLI asynchronously
        try:
            raw_response = await self.cli.aexecute(
                prompt=effective_prompt,
                json_schema=json_schema,
                **kwargs,
            )
        except Exception as e:
            raise ClaudeCodeError(f"CLI execution failed: {e}") from e

        # Parse response
        parsed = parse_json_response(raw_response)

        # Capture transcript
        if self._capture_transcript:
            self._add_transcript_entry(
                prompt=effective_prompt,
                parsed=parsed,
                raw_response=raw_response,
            )

        # Build LMResponse for DSPy
        response = self._build_lm_response(parsed)

        # Cache result (serialize to dict for JSON compatibility)
        if self._our_cache:
            self._our_cache.set(cache_key, response.to_dict())

        return response

    def _build_lm_response(self, parsed: ParsedResponse) -> LMResponse:
        """Build an LMResponse from parsed CLI output.

        Args:
            parsed: Parsed response from Claude Code CLI.

        Returns:
            LMResponse object for DSPy.
        """
        import json

        # Extract the result text
        if parsed.get("structured_output") is not None:
            text = json.dumps(parsed["structured_output"])
        else:
            text = parsed.get("result", "")

        # Determine finish reason
        if parsed.get("is_error"):
            finish_reason = "error"
        elif parsed.get("subtype") == "success":
            finish_reason = "stop"
        else:
            finish_reason = "stop"

        # Build usage info
        usage_data = parsed.get("usage", {})
        input_tokens = (
            usage_data.get("input_tokens", 0)
            + usage_data.get("cache_read_input_tokens", 0)
            + usage_data.get("cache_creation_input_tokens", 0)
        )
        output_tokens = usage_data.get("output_tokens", 0)

        usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        # Get model name from response
        model_usage = parsed.get("modelUsage", {})
        model = next(iter(model_usage.keys())) if model_usage else self.model

        response = LMResponse(
            text=text,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
        )

        # Set cost if available
        if "total_cost_usd" in parsed:
            response.set_cost(parsed["total_cost_usd"])

        return response

    def _extract_json_schema(
        self,
        json_schema: dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Extract JSON schema from response_format or explicit json_schema.

        DSPy's JSONAdapter passes a response_format kwarg that can be:
        - A Pydantic model class (for structured outputs)
        - {"type": "json_object"} (for JSON mode)

        This method converts these to a JSON schema for Claude Code's
        --json-schema flag.

        Args:
            json_schema: Explicit JSON schema if provided.
            kwargs: Additional kwargs that may contain response_format.

        Returns:
            JSON schema dict or None.
        """
        # If explicit json_schema provided, use it
        if json_schema is not None:
            return json_schema

        # Check for response_format from DSPy's JSONAdapter
        response_format = kwargs.pop("response_format", None)
        if response_format is None:
            return None

        # Handle {"type": "json_object"} - no specific schema
        if isinstance(response_format, dict):
            if response_format.get("type") == "json_object":
                # Basic JSON mode without schema - return minimal schema
                return {"type": "object"}
            # If it's already a schema dict, use it
            return response_format

        # Handle Pydantic model - extract JSON schema
        try:
            import pydantic

            is_pydantic = isinstance(response_format, type) and issubclass(
                response_format, pydantic.BaseModel
            )
            if is_pydantic:
                schema: dict[str, Any] = response_format.model_json_schema()
                # Clean up the schema for Claude Code
                # Remove $defs if present and inline them
                if "$defs" in schema:
                    schema = self._inline_schema_defs(schema)
                return schema
        except ImportError:
            logger.warning("Pydantic not available for response_format conversion")
        except Exception as e:
            logger.warning(f"Failed to extract schema from response_format: {e}")

        return None

    def _inline_schema_defs(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Inline $defs references in a JSON schema.

        Claude Code's --json-schema works best with fully inlined schemas
        rather than schemas with $ref references.

        Args:
            schema: JSON schema with potential $defs.

        Returns:
            Schema with $refs resolved.
        """
        import copy

        defs = schema.pop("$defs", {})
        if not defs:
            return schema

        def resolve_refs(obj: Any) -> Any:
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_path = obj["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        def_name = ref_path.split("/")[-1]
                        if def_name in defs:
                            return resolve_refs(copy.deepcopy(defs[def_name]))
                    return obj
                return {k: resolve_refs(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_refs(item) for item in obj]
            return obj

        return cast(dict[str, Any], resolve_refs(schema))

    def _prepare_prompt(
        self,
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
    ) -> str:
        """Convert messages to a single prompt string for CLI.

        Args:
            prompt: Simple text prompt.
            messages: Chat messages in OpenAI format.

        Returns:
            The effective prompt string.

        Raises:
            ValueError: If neither prompt nor messages is provided.
        """
        if prompt is not None:
            return prompt

        if messages is None or len(messages) == 0:
            raise ValueError("Either prompt or messages must be provided")

        # Convert chat messages to prompt
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # System messages are handled via --system-prompt
                # but we can include them in context if no system_prompt is set
                if not self.config.system_prompt:
                    parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:  # user
                parts.append(f"User: {content}")

        return "\n\n".join(parts)

    def _add_transcript_entry(
        self,
        prompt: str,
        parsed: ParsedResponse,
        raw_response: str,
    ) -> None:
        """Add an entry to the transcript.

        Args:
            prompt: The prompt that was sent.
            parsed: The parsed response.
            raw_response: The raw CLI output.
        """
        # Extract models from modelUsage if available
        models = None
        if "modelUsage" in parsed:
            models = list(parsed["modelUsage"].keys())

        usage_data = parsed.get("usage")
        entry = TranscriptEntry(
            prompt=prompt,
            response=cast(dict[str, Any], parsed),
            raw_response=raw_response,
            system_prompt=self.config.system_prompt,
            append_system_prompt=self.config.append_system_prompt,
            usage=cast(dict[str, Any], usage_data) if usage_data else None,
            models=models,
            session_id=parsed.get("session_id"),
            duration_ms=parsed.get("duration_ms"),
            cost_usd=parsed.get("total_cost_usd"),
        )
        self._transcript.add_entry(entry)

    def copy(self, **kwargs: Any) -> ClaudeCodeLM:
        """Create a copy with modified parameters.

        This method is required for DSPy compatibility, allowing creation
        of modified LM instances (e.g., with different temperature).

        Args:
            **kwargs: Parameters to update.

        Returns:
            A new ClaudeCodeLM instance with updates applied.
        """
        # Get current config as dict
        config_dict = asdict(self.config)

        # Map config fields to init params
        init_params = {
            "model": config_dict["model"],
            "system_prompt": config_dict["system_prompt"],
            "append_system_prompt": config_dict["append_system_prompt"],
            "permission_mode": config_dict["permission_mode"],
            "dangerously_skip_permissions": config_dict["dangerously_skip_permissions"],
            "allowed_tools": list(config_dict["allowed_tools"]),
            "disallowed_tools": list(config_dict["disallowed_tools"]),
            "working_directory": config_dict["working_directory"],
            "add_dirs": list(config_dict["add_dirs"]),
            "timeout_seconds": config_dict["timeout_seconds"],
            "verbose": config_dict["verbose"],
            "max_budget_usd": config_dict["max_budget_usd"],
            # Non-config params from BaseLM
            "model_type": self.model_type,
            "temperature": self.kwargs.get("temperature", 0.0),
            "max_tokens": self.kwargs.get("max_tokens", 4096),
            "cache": self.cache,
            "cache_dir": self._cache_dir,
            "cache_ttl": self._cache_ttl,
            "capture_transcript": self._capture_transcript,
            "claude_binary": self.cli.binary,
            "anthropic_api_key": self.auth.anthropic_api_key,
            "oauth_token": self.auth.oauth_token,
        }

        # Apply updates
        init_params.update(kwargs)

        return ClaudeCodeLM(**init_params)

    def dump_state(self) -> dict[str, Any]:
        """Export configuration (excluding secrets).

        Returns:
            Dictionary with configuration state.
        """
        return {
            "model": self.config.model,
            "model_type": self.model_type,
            "permission_mode": self.config.permission_mode,
            "cache_enabled": self.cache,
            "history_length": len(self.history),
            "transcript_length": len(self._transcript),
            "auth_method": self.auth.auth_method,
        }

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self._our_cache:
            self._our_cache.clear()

    def get_transcript_json(self) -> str:
        """Get the transcript as a JSON string.

        Returns:
            JSON representation of the transcript.
        """
        return self._transcript.to_json()

    def get_total_cost(self) -> float:
        """Get the total cost of all interactions.

        Returns:
            Total cost in USD.
        """
        return self._transcript.total_cost_usd()

    def get_total_tokens(self) -> dict[str, int]:
        """Get the total token usage.

        Returns:
            Dictionary with token usage breakdown.
        """
        return self._transcript.total_tokens()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ClaudeCodeLM(model={self.config.model!r}, "
            f"permission_mode={self.config.permission_mode!r}, "
            f"cache={self.cache})"
        )
