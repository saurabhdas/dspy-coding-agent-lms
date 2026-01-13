"""CLI command builder and executor for Claude Code."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
from typing import Any

from .auth import AuthConfig
from .config import ClaudeCodeConfig
from .exceptions import ClaudeCodeError, ClaudeCodeExecutionError, ClaudeCodeTimeoutError

logger = logging.getLogger(__name__)


class ClaudeCodeCLI:
    """Builds and executes Claude Code CLI commands.

    This class handles the construction of CLI arguments and execution
    of the Claude Code CLI, both synchronously and asynchronously.

    Attributes:
        binary: Path to the claude binary.
        config: Configuration for CLI invocation.
        auth: Authentication configuration.
    """

    def __init__(
        self,
        binary: str = "claude",
        config: ClaudeCodeConfig | None = None,
        auth: AuthConfig | None = None,
    ) -> None:
        """Initialize the CLI executor.

        Args:
            binary: Path to the claude binary (default: "claude").
            config: Configuration for CLI invocation.
            auth: Authentication configuration.
        """
        self.binary = binary
        self.config = config or ClaudeCodeConfig()
        self.auth = auth

    def build_command(
        self,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Build the CLI command as a list of arguments.

        Args:
            prompt: The prompt to send to Claude Code.
            json_schema: Optional JSON schema for structured output.
            **kwargs: Additional CLI options to override config.

        Returns:
            List of command arguments for subprocess execution.
        """
        cmd = [self.binary]

        # Always use print mode for non-interactive execution
        cmd.append("--print")

        # Output format (always JSON for programmatic use)
        output_format = kwargs.get("output_format", self.config.output_format)
        cmd.extend(["--output-format", output_format])

        # Model selection
        model = kwargs.get("model", self.config.model)
        cmd.extend(["--model", model])

        # Permission mode
        permission_mode = kwargs.get("permission_mode", self.config.permission_mode)
        cmd.extend(["--permission-mode", permission_mode])

        # System prompt options
        system_prompt = kwargs.get("system_prompt", self.config.system_prompt)
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        append_system_prompt = kwargs.get(
            "append_system_prompt", self.config.append_system_prompt
        )
        if append_system_prompt:
            cmd.extend(["--append-system-prompt", append_system_prompt])

        # Dangerous permission bypass
        dangerously_skip = kwargs.get(
            "dangerously_skip_permissions", self.config.dangerously_skip_permissions
        )
        if dangerously_skip:
            cmd.append("--dangerously-skip-permissions")

        # Tool restrictions
        allowed_tools = kwargs.get("allowed_tools", self.config.allowed_tools)
        if allowed_tools:
            # Convert tuple to list if needed
            tools_list = list(allowed_tools) if isinstance(allowed_tools, tuple) else allowed_tools
            for tool in tools_list:
                cmd.extend(["--allowedTools", tool])

        disallowed_tools = kwargs.get("disallowed_tools", self.config.disallowed_tools)
        if disallowed_tools:
            tools_list = (
                list(disallowed_tools) if isinstance(disallowed_tools, tuple) else disallowed_tools
            )
            for tool in tools_list:
                cmd.extend(["--disallowedTools", tool])

        # Additional directories
        add_dirs = kwargs.get("add_dirs", self.config.add_dirs)
        if add_dirs:
            dirs_list = list(add_dirs) if isinstance(add_dirs, tuple) else add_dirs
            for dir_path in dirs_list:
                cmd.extend(["--add-dir", dir_path])

        # JSON schema for structured output
        if json_schema:
            cmd.extend(["--json-schema", json.dumps(json_schema)])

        # Budget limit
        max_budget = kwargs.get("max_budget_usd", self.config.max_budget_usd)
        if max_budget is not None:
            cmd.extend(["--max-budget-usd", str(max_budget)])

        # Verbose mode
        verbose = kwargs.get("verbose", self.config.verbose)
        if verbose:
            cmd.append("--verbose")

        # NOTE: We pass the prompt via stdin instead of as a positional argument
        # because the CLI has issues parsing positional args when multiple
        # --allowedTools or --disallowedTools flags are used.

        return cmd

    def build_environment(self) -> dict[str, str]:
        """Build environment variables for CLI execution.

        Returns:
            Dictionary of environment variables including auth credentials.
        """
        env = os.environ.copy()

        if self.auth:
            auth_env = self.auth.to_environment()
            env.update(auth_env)

        return env

    def execute(
        self,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute CLI command synchronously.

        Args:
            prompt: The prompt to send to Claude Code.
            json_schema: Optional JSON schema for structured output.
            **kwargs: Additional CLI options.

        Returns:
            The stdout output from the CLI.

        Raises:
            ClaudeCodeTimeoutError: If the command times out.
            ClaudeCodeError: If the claude binary is not found.
            ClaudeCodeExecutionError: If the CLI exits with non-zero code.
        """
        cmd = self.build_command(prompt, json_schema, **kwargs)
        env = self.build_environment()
        cwd = kwargs.get("working_directory", self.config.working_directory)
        timeout = kwargs.get("timeout_seconds", self.config.timeout_seconds)

        logger.debug("Executing: %s", shlex.join(cmd))
        logger.debug("Working directory: %s", cwd or os.getcwd())

        try:
            result = subprocess.run(
                cmd,
                input=prompt,  # Pass prompt via stdin
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=cwd,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise ClaudeCodeTimeoutError(
                f"Command timed out after {timeout}s"
            ) from e
        except FileNotFoundError as e:
            raise ClaudeCodeError(
                f"Claude binary not found: {self.binary}. "
                "Ensure Claude Code CLI is installed and in PATH."
            ) from e

        if result.returncode != 0:
            logger.error("CLI stderr: %s", result.stderr)
            raise ClaudeCodeExecutionError(
                f"CLI exited with code {result.returncode}: {result.stderr}",
                exit_code=result.returncode,
                stderr=result.stderr,
            )

        logger.debug("CLI stdout length: %d bytes", len(result.stdout))
        return result.stdout

    async def aexecute(
        self,
        prompt: str,
        json_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Execute CLI command asynchronously.

        Uses asyncio.subprocess for non-blocking execution, allowing
        concurrent requests and better integration with async frameworks.

        Args:
            prompt: The prompt to send to Claude Code.
            json_schema: Optional JSON schema for structured output.
            **kwargs: Additional CLI options.

        Returns:
            The stdout output from the CLI.

        Raises:
            ClaudeCodeTimeoutError: If the command times out.
            ClaudeCodeError: If the claude binary is not found.
            ClaudeCodeExecutionError: If the CLI exits with non-zero code.
        """
        cmd = self.build_command(prompt, json_schema, **kwargs)
        env = self.build_environment()
        cwd = kwargs.get("working_directory", self.config.working_directory)
        timeout = kwargs.get("timeout_seconds", self.config.timeout_seconds)

        logger.debug("Async executing: %s", shlex.join(cmd))
        logger.debug("Working directory: %s", cwd or os.getcwd())

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=prompt.encode("utf-8")),  # Pass prompt via stdin
                timeout=timeout,
            )
        except TimeoutError as e:
            # Kill the process if it times out
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass  # Process already terminated
            raise ClaudeCodeTimeoutError(
                f"Command timed out after {timeout}s"
            ) from e
        except FileNotFoundError as e:
            raise ClaudeCodeError(
                f"Claude binary not found: {self.binary}. "
                "Ensure Claude Code CLI is installed and in PATH."
            ) from e

        stdout = stdout_bytes.decode("utf-8")
        stderr = stderr_bytes.decode("utf-8")

        if process.returncode != 0:
            logger.error("CLI stderr: %s", stderr)
            raise ClaudeCodeExecutionError(
                f"CLI exited with code {process.returncode}: {stderr}",
                exit_code=process.returncode or 1,
                stderr=stderr,
            )

        logger.debug("CLI stdout length: %d bytes", len(stdout))
        return stdout
