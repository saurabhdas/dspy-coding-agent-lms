"""Tests for cli module."""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import MagicMock

import pytest

from dspy_coding_agent_lms.auth import AuthConfig
from dspy_coding_agent_lms.cli import ClaudeCodeCLI
from dspy_coding_agent_lms.config import ClaudeCodeConfig
from dspy_coding_agent_lms.exceptions import (
    ClaudeCodeError,
    ClaudeCodeExecutionError,
    ClaudeCodeTimeoutError,
)


class TestClaudeCodeCLI:
    """Tests for ClaudeCodeCLI class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        cli = ClaudeCodeCLI()

        assert cli.binary == "claude"
        assert cli.config.model == "sonnet"
        assert cli.auth is None

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        config = ClaudeCodeConfig(model="opus", permission_mode="dontAsk")
        auth = AuthConfig(anthropic_api_key="sk-test")
        cli = ClaudeCodeCLI(binary="/custom/claude", config=config, auth=auth)

        assert cli.binary == "/custom/claude"
        assert cli.config.model == "opus"
        assert cli.auth is not None
        assert cli.auth.anthropic_api_key == "sk-test"


class TestBuildCommand:
    """Tests for build_command method."""

    def test_basic_command(self) -> None:
        """Test basic command building."""
        cli = ClaudeCodeCLI()
        cmd = cli.build_command("Hello")

        assert cmd[0] == "claude"
        assert "--print" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--model" in cmd
        assert "sonnet" in cmd
        assert "--permission-mode" in cmd
        assert "plan" in cmd
        # Prompt is passed via stdin, not as positional argument
        assert "Hello" not in cmd

    def test_with_json_schema(self) -> None:
        """Test command with JSON schema."""
        cli = ClaudeCodeCLI()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        cmd = cli.build_command("Test", json_schema=schema)

        assert "--json-schema" in cmd
        schema_idx = cmd.index("--json-schema")
        assert '{"type": "object"' in cmd[schema_idx + 1] or '"type":"object"' in cmd[schema_idx + 1]

    def test_with_system_prompt(self) -> None:
        """Test command with system prompt."""
        config = ClaudeCodeConfig(system_prompt="Be helpful")
        cli = ClaudeCodeCLI(config=config)
        cmd = cli.build_command("Test")

        assert "--system-prompt" in cmd
        assert "Be helpful" in cmd

    def test_with_append_system_prompt(self) -> None:
        """Test command with append system prompt."""
        config = ClaudeCodeConfig(append_system_prompt="Additional instructions")
        cli = ClaudeCodeCLI(config=config)
        cmd = cli.build_command("Test")

        assert "--append-system-prompt" in cmd
        assert "Additional instructions" in cmd

    def test_with_tools(self) -> None:
        """Test command with tool restrictions."""
        config = ClaudeCodeConfig(
            allowed_tools=("Read", "Bash(git:*)"),
            disallowed_tools=("Edit",),
        )
        cli = ClaudeCodeCLI(config=config)
        cmd = cli.build_command("Test")

        # Check allowed tools
        assert "--allowedTools" in cmd
        assert "Read" in cmd
        assert "Bash(git:*)" in cmd

        # Check disallowed tools
        assert "--disallowedTools" in cmd
        assert "Edit" in cmd

    def test_with_dangerous_skip(self) -> None:
        """Test command with dangerous skip permissions."""
        config = ClaudeCodeConfig(dangerously_skip_permissions=True)
        cli = ClaudeCodeCLI(config=config)
        cmd = cli.build_command("Test")

        assert "--dangerously-skip-permissions" in cmd

    def test_with_budget(self) -> None:
        """Test command with budget limit."""
        config = ClaudeCodeConfig(max_budget_usd=1.0)
        cli = ClaudeCodeCLI(config=config)
        cmd = cli.build_command("Test")

        assert "--max-budget-usd" in cmd
        assert "1.0" in cmd

    def test_with_add_dirs(self) -> None:
        """Test command with additional directories."""
        config = ClaudeCodeConfig(add_dirs=("/path/one", "/path/two"))
        cli = ClaudeCodeCLI(config=config)
        cmd = cli.build_command("Test")

        add_dir_count = cmd.count("--add-dir")
        assert add_dir_count == 2
        assert "/path/one" in cmd
        assert "/path/two" in cmd

    def test_kwargs_override_config(self) -> None:
        """Test that kwargs override config values."""
        config = ClaudeCodeConfig(model="sonnet")
        cli = ClaudeCodeCLI(config=config)
        cmd = cli.build_command("Test", model="opus")

        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "opus"


class TestBuildEnvironment:
    """Tests for build_environment method."""

    def test_no_auth(self) -> None:
        """Test environment without auth."""
        cli = ClaudeCodeCLI()
        env = cli.build_environment()

        # Should have inherited environment
        assert isinstance(env, dict)

    def test_with_api_key(self) -> None:
        """Test environment with API key."""
        auth = AuthConfig(anthropic_api_key="sk-test-key")
        cli = ClaudeCodeCLI(auth=auth)
        env = cli.build_environment()

        assert env.get("ANTHROPIC_API_KEY") == "sk-test-key"

    def test_with_oauth_token(self) -> None:
        """Test environment with OAuth token."""
        auth = AuthConfig(oauth_token="oauth-test-token")
        cli = ClaudeCodeCLI(auth=auth)
        env = cli.build_environment()

        assert env.get("CLAUDE_CODE_OAUTH_TOKEN") == "oauth-test-token"


class TestExecute:
    """Tests for execute method."""

    def test_execute_success(self, mocker: Any) -> None:
        """Test successful execution."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"type":"result","result":"Hello"}',
            stderr="",
        )

        cli = ClaudeCodeCLI()
        result = cli.execute("Test prompt")

        assert '"result":"Hello"' in result
        mock_run.assert_called_once()

    def test_execute_timeout(self, mocker: Any) -> None:
        """Test execution timeout."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=10)

        cli = ClaudeCodeCLI()
        with pytest.raises(ClaudeCodeTimeoutError):
            cli.execute("Test")

    def test_execute_not_found(self, mocker: Any) -> None:
        """Test binary not found."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.side_effect = FileNotFoundError()

        cli = ClaudeCodeCLI()
        with pytest.raises(ClaudeCodeError, match="not found"):
            cli.execute("Test")

    def test_execute_non_zero_exit(self, mocker: Any) -> None:
        """Test non-zero exit code."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error message",
        )

        cli = ClaudeCodeCLI()
        with pytest.raises(ClaudeCodeExecutionError) as exc_info:
            cli.execute("Test")

        assert exc_info.value.exit_code == 1
        assert "Error message" in exc_info.value.stderr


class TestAsyncExecute:
    """Tests for aexecute method."""

    @pytest.mark.asyncio
    async def test_aexecute_success(self, mocker: Any) -> None:
        """Test successful async execution."""
        mock_process = MagicMock()
        mock_process.returncode = 0

        async def mock_communicate(input: bytes | None = None) -> tuple[bytes, bytes]:
            return (b'{"type":"result","result":"Async OK"}', b"")

        mock_process.communicate = mock_communicate

        mock_create = mocker.patch("asyncio.create_subprocess_exec")
        mock_create.return_value = mock_process

        cli = ClaudeCodeCLI()
        result = await cli.aexecute("Test prompt")

        assert '"result":"Async OK"' in result

    @pytest.mark.asyncio
    async def test_aexecute_timeout(self, mocker: Any) -> None:
        """Test async execution timeout."""
        import asyncio

        mock_process = MagicMock()
        mock_process.kill = MagicMock()
        mock_process.wait = MagicMock(return_value=None)

        async def mock_communicate(input: bytes | None = None) -> tuple[bytes, bytes]:
            raise asyncio.TimeoutError()

        mock_process.communicate = mock_communicate

        # Make wait awaitable
        async def mock_wait() -> None:
            pass

        mock_process.wait = mock_wait

        mock_create = mocker.patch("asyncio.create_subprocess_exec")
        mock_create.return_value = mock_process

        mocker.patch("asyncio.wait_for", side_effect=asyncio.TimeoutError())

        cli = ClaudeCodeCLI()
        with pytest.raises(ClaudeCodeTimeoutError):
            await cli.aexecute("Test")
