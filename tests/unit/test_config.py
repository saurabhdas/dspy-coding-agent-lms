"""Tests for config module."""

from __future__ import annotations

import pytest

from dspy_coding_agent_lms.config import ClaudeCodeConfig, StructuredOutputConfig


class TestClaudeCodeConfig:
    """Tests for ClaudeCodeConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ClaudeCodeConfig()

        assert config.model == "sonnet"
        assert config.permission_mode == "plan"
        assert config.dangerously_skip_permissions is False
        assert config.timeout_seconds == 300.0
        assert config.output_format == "json"
        assert config.system_prompt is None
        assert config.append_system_prompt is None
        assert config.working_directory is None
        assert config.allowed_tools == ()
        assert config.disallowed_tools == ()

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = ClaudeCodeConfig(
            model="opus",
            permission_mode="bypassPermissions",
            dangerously_skip_permissions=True,
            timeout_seconds=600.0,
            system_prompt="Custom system prompt",
            allowed_tools=("Read", "Glob"),
            working_directory="/tmp/test",
        )

        assert config.model == "opus"
        assert config.permission_mode == "bypassPermissions"
        assert config.dangerously_skip_permissions is True
        assert config.timeout_seconds == 600.0
        assert config.system_prompt == "Custom system prompt"
        assert config.allowed_tools == ("Read", "Glob")
        assert config.working_directory == "/tmp/test"

    def test_immutability(self) -> None:
        """Test that config is frozen (immutable)."""
        config = ClaudeCodeConfig()

        with pytest.raises(AttributeError):
            config.model = "opus"  # type: ignore[misc]

    def test_invalid_timeout(self) -> None:
        """Test validation rejects invalid timeout."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ClaudeCodeConfig(timeout_seconds=0)

        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ClaudeCodeConfig(timeout_seconds=-1)

    def test_invalid_budget(self) -> None:
        """Test validation rejects invalid budget."""
        with pytest.raises(ValueError, match="max_budget_usd must be positive"):
            ClaudeCodeConfig(max_budget_usd=0)

        with pytest.raises(ValueError, match="max_budget_usd must be positive"):
            ClaudeCodeConfig(max_budget_usd=-5.0)

    def test_with_updates(self) -> None:
        """Test creating new config with updates."""
        config = ClaudeCodeConfig(model="sonnet")
        updated = config.with_updates(model="opus", timeout_seconds=600.0)

        assert config.model == "sonnet"  # Original unchanged
        assert updated.model == "opus"
        assert updated.timeout_seconds == 600.0


class TestStructuredOutputConfig:
    """Tests for StructuredOutputConfig dataclass."""

    def test_basic_schema(self) -> None:
        """Test creating config with a schema."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        config = StructuredOutputConfig(schema=schema)

        assert config.schema == schema
        assert config.strict is True

    def test_to_json_string(self) -> None:
        """Test converting schema to JSON string."""
        schema = {"type": "object", "properties": {"value": {"type": "integer"}}}
        config = StructuredOutputConfig(schema=schema)

        json_str = config.to_json_string()
        assert '"type": "object"' in json_str or '"type":"object"' in json_str

    def test_from_json_string(self) -> None:
        """Test creating config from JSON string."""
        json_str = '{"type": "object", "properties": {"x": {"type": "number"}}}'
        config = StructuredOutputConfig.from_json_string(json_str)

        assert config.schema["type"] == "object"
        assert "properties" in config.schema
