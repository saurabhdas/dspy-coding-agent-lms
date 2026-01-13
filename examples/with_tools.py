#!/usr/bin/env python3
"""Example with tool restrictions for controlled execution.

This example demonstrates how to configure tool access for different
use cases, from read-only code review to full agentic execution.

Requirements:
    - Claude Code CLI installed and authenticated
    - dspy-coding-agent-lms installed

Run:
    python examples/with_tools.py
"""

from __future__ import annotations

from dspy_coding_agent_lms import ClaudeCodeLM


def readonly_code_analysis() -> None:
    """Example: Read-only code analysis mode."""
    print("Example 1: Read-Only Code Analysis")
    print("=" * 50)

    # Configure for code review - read-only access
    lm = ClaudeCodeLM(
        model="sonnet",
        permission_mode="plan",
        # Only allow read-only tools
        allowed_tools=[
            "Read",           # Read file contents
            "Glob",           # Find files by pattern
            "Grep",           # Search file contents
            "WebSearch",      # Search the web
        ],
        # Explicitly disallow modification tools
        disallowed_tools=[
            "Bash",           # No shell commands
            "Edit",           # No file editing
            "Write",          # No file writing
        ],
        append_system_prompt=(
            "You are analyzing code in read-only mode. "
            "Describe what you find but do not modify anything."
        ),
    )

    print(f"Allowed tools: {list(lm.config.allowed_tools)}")
    print(f"Disallowed tools: {list(lm.config.disallowed_tools)}")
    print()

    result = lm.forward(
        prompt="Describe what a typical Python project structure looks like."
    )
    print("Response:", result.choices[0].message.content[:200], "...")
    print()


def git_only_mode() -> None:
    """Example: Git operations only mode."""
    print("Example 2: Git Operations Only")
    print("=" * 50)

    # Only allow git-related commands
    lm = ClaudeCodeLM(
        model="sonnet",
        permission_mode="plan",
        allowed_tools=[
            "Bash(git status)",
            "Bash(git log*)",
            "Bash(git diff*)",
            "Bash(git branch*)",
            "Read",
        ],
        disallowed_tools=[
            "Bash(git push*)",  # No pushing
            "Bash(git reset*)",  # No resets
            "Edit",
            "Write",
        ],
        append_system_prompt=(
            "You can only examine git history and status. "
            "Do not make any changes to the repository."
        ),
    )

    print(f"Allowed: {list(lm.config.allowed_tools)}")
    print()

    result = lm.forward(
        prompt="What git commands could you run to show recent history?"
    )
    print("Response:", result.choices[0].message.content[:200], "...")
    print()


def full_agent_mode() -> None:
    """Example: Full agentic mode (use with caution!)."""
    print("Example 3: Full Agent Mode (Demo Only)")
    print("=" * 50)
    print("WARNING: This mode grants extensive permissions!")
    print()

    # Full agentic mode - use with extreme caution
    # This is for demonstration purposes only
    lm = ClaudeCodeLM(
        model="opus",  # Use more capable model for complex tasks
        permission_mode="acceptEdits",  # Auto-accept edits
        # Allow all standard tools
        allowed_tools=[
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "Bash(npm *)",
            "Bash(python *)",
            "Bash(pip *)",
        ],
        # Budget limit for safety
        max_budget_usd=0.50,
        timeout_seconds=120.0,
    )

    print("Configuration:")
    print(f"  Model: {lm.config.model}")
    print(f"  Permission mode: {lm.config.permission_mode}")
    print(f"  Budget limit: ${lm.config.max_budget_usd}")
    print()

    # In a real scenario, you would have the agent perform tasks
    # Here we just show the configuration
    print("Full agent mode configured (not executing for safety)")


def custom_working_directory() -> None:
    """Example: Working in a specific directory."""
    print("Example 4: Custom Working Directory")
    print("=" * 50)

    # Work in a specific project directory
    lm = ClaudeCodeLM(
        model="sonnet",
        permission_mode="plan",
        working_directory="/tmp",  # Work in /tmp for this example
        add_dirs=["/var/log"],     # Also allow access to logs
        allowed_tools=["Read", "Glob", "Bash(ls*)"],
    )

    print(f"Working directory: {lm.config.working_directory}")
    print(f"Additional directories: {list(lm.config.add_dirs)}")
    print()

    result = lm.forward(
        prompt="What files might exist in /tmp on a typical system?"
    )
    print("Response:", result.choices[0].message.content[:200], "...")
    print()


def main() -> None:
    """Run all tool restriction examples."""
    readonly_code_analysis()
    git_only_mode()
    full_agent_mode()
    custom_working_directory()

    print("=" * 50)
    print("Tool restriction examples complete!")


if __name__ == "__main__":
    main()
