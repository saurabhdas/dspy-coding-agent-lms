"""Authentication handling for Claude Code LM."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

AuthMethod = Literal["api_key", "oauth", None]


@dataclass(frozen=True)
class AuthConfig:
    """Authentication configuration for Claude Code.

    Claude Code supports two authentication methods:
    1. Anthropic API Key (ANTHROPIC_API_KEY)
    2. OAuth Token for Pro/Max subscribers (CLAUDE_CODE_OAUTH_TOKEN)

    If both are configured, the library will use the first available method.

    Attributes:
        anthropic_api_key: Anthropic API key for authentication.
        oauth_token: OAuth token for Pro/Max subscribers.
    """

    anthropic_api_key: str | None = None
    oauth_token: str | None = None

    @property
    def is_authenticated(self) -> bool:
        """Check if any authentication is configured.

        Returns:
            True if at least one authentication method is configured.
        """
        return bool(self.anthropic_api_key or self.oauth_token)

    @property
    def auth_method(self) -> AuthMethod:
        """Return the authentication method being used.

        OAuth token takes precedence over API key if both are configured.

        Returns:
            The authentication method being used, or None if not authenticated.
        """
        if self.oauth_token:
            return "oauth"
        if self.anthropic_api_key:
            return "api_key"
        return None

    def to_environment(self) -> dict[str, str]:
        """Convert auth config to environment variables.

        Returns:
            Dictionary of environment variables for authentication.
        """
        env: dict[str, str] = {}
        if self.anthropic_api_key:
            env["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        if self.oauth_token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = self.oauth_token
        return env


def resolve_authentication(
    anthropic_api_key: str | None = None,
    oauth_token: str | None = None,
) -> AuthConfig:
    """Resolve authentication from explicit args or environment.

    This function attempts to resolve authentication credentials in the
    following priority order:
    1. Explicit arguments passed to this function
    2. Environment variables (ANTHROPIC_API_KEY, CLAUDE_CODE_OAUTH_TOKEN)

    Args:
        anthropic_api_key: Explicit API key (takes priority over env var).
        oauth_token: Explicit OAuth token (takes priority over env var).

    Returns:
        AuthConfig with resolved credentials.

    Example:
        >>> auth = resolve_authentication()  # From environment
        >>> auth = resolve_authentication(anthropic_api_key="sk-...")  # Explicit
    """
    # Explicit arguments take priority over environment
    key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    token = oauth_token or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")

    return AuthConfig(
        anthropic_api_key=key,
        oauth_token=token,
    )
