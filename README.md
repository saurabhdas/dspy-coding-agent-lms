# dspy-coding-agent-lms

DSPy Language Model integration for Claude Code CLI - route DSPy requests through your local Claude Code agent instead of using Anthropic APIs directly.

## Overview

`dspy-coding-agent-lms` provides a DSPy-compatible language model that invokes the Claude Code CLI for each request, enabling:

- **Agentic capabilities**: Access Claude Code's file editing, bash execution, and tool use within DSPy programs
- **Structured output**: Use Claude Code's built-in JSON schema validation instead of string parsing
- **Permission control**: Fine-grained control over which tools Claude can use
- **Cost tracking**: Automatic transcript capture with usage and cost metrics
- **Local execution**: Run Claude Code locally with your existing authentication

## Installation

### Prerequisites

1. **Claude Code CLI** - Install and authenticate:
   ```bash
   npm install -g @anthropic-ai/claude-code
   claude login  # or use ANTHROPIC_API_KEY
   ```

2. **Python 3.11+**

### Install the package

```bash
# Using uv (recommended)
uv add dspy-coding-agent-lms

# Using pip
pip install dspy-coding-agent-lms
```

## Quick Start

```python
import dspy
from dspy_coding_agent_lms import ClaudeCodeLM

# Initialize the LM
lm = ClaudeCodeLM(model="sonnet", permission_mode="plan")

# Configure DSPy with JSONAdapter (RECOMMENDED)
# JSONAdapter uses Claude Code's native --json-schema for reliable structured output
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

# Use DSPy as normal
predict = dspy.Predict("question -> answer")
result = predict(question="What is 2+2?")
print(result.answer)

# Check cost and usage
print(f"Cost: ${lm.transcript.total_cost_usd():.4f}")
```

> **Recommended**: Always use `dspy.JSONAdapter()` with ClaudeCodeLM. This enables native JSON schema output via Claude Code's `--json-schema` flag, providing more reliable structured output than string parsing.

## Features

### DSPy Adapters: JSONAdapter vs ChatAdapter

When using DSPy Signatures, you can choose between two adapters:

#### JSONAdapter (Recommended)

Uses Claude Code's native `--json-schema` flag for structured output. The response is pure JSON in the `structured_output` field.

```python
import dspy
from dspy_coding_agent_lms import ClaudeCodeLM

lm = ClaudeCodeLM(model="sonnet")
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())  # Recommended

class SentimentSignature(dspy.Signature):
    """Analyze sentiment of text."""
    text: str = dspy.InputField()
    sentiment: str = dspy.OutputField()
    confidence: float = dspy.OutputField()

result = dspy.Predict(SentimentSignature)(text="I love this!")
# Output format: {"sentiment": "positive", "confidence": 0.95}
```

#### ChatAdapter (Default)

Uses string parsing with `[[ ## field ## ]]` markers. This is DSPy's default but less reliable.

```python
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())  # Default, not recommended

# Output format:
# [[ ## sentiment ## ]]
# positive
# [[ ## confidence ## ]]
# 0.95
```

**Why JSONAdapter is recommended:**
- More reliable parsing (no regex-based extraction)
- Native JSON schema validation by Claude Code
- Cleaner API responses with `structured_output` field
- Better type safety for complex nested structures

### Direct JSON Schema

Use Claude Code's built-in JSON schema validation for reliable structured outputs:

```python
from dspy_coding_agent_lms import ClaudeCodeLM
import json

lm = ClaudeCodeLM(model="sonnet")

schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
        "explanation": {"type": "string"}
    },
    "required": ["answer"]
}

result = lm.forward(
    prompt="What is 15% of 200?",
    json_schema=schema
)

data = json.loads(result[0]["text"])
print(f"Answer: {data['answer']}")  # Answer: 30
```

### Tool Restrictions

Control which tools Claude can access:

```python
# Read-only code analysis
lm = ClaudeCodeLM(
    model="sonnet",
    permission_mode="plan",
    allowed_tools=["Read", "Glob", "Grep"],
    disallowed_tools=["Bash", "Edit", "Write"],
)

# Git operations only
lm = ClaudeCodeLM(
    allowed_tools=[
        "Bash(git status)",
        "Bash(git log*)",
        "Bash(git diff*)",
    ],
)
```

### Async Support

Execute concurrent requests for better throughput:

```python
import asyncio
from dspy_coding_agent_lms import ClaudeCodeLM

async def main():
    lm = ClaudeCodeLM(model="sonnet")

    prompts = ["Question 1", "Question 2", "Question 3"]
    tasks = [lm.aforward(prompt=p) for p in prompts]
    results = await asyncio.gather(*tasks)

asyncio.run(main())
```

### Transcript and Cost Tracking

Access complete interaction history with usage metrics:

```python
lm = ClaudeCodeLM(model="sonnet")

# Make some requests...
lm.forward(prompt="Hello")
lm.forward(prompt="World")

# Access transcript
for entry in lm.transcript:
    print(f"Prompt: {entry.prompt}")
    print(f"Duration: {entry.duration_ms}ms")
    print(f"Cost: ${entry.cost_usd:.4f}")

# Aggregated stats
print(f"Total cost: ${lm.transcript.total_cost_usd():.4f}")
print(f"Total tokens: {lm.transcript.total_tokens()}")

# Export transcript
lm.transcript.save_to_file("transcript.json")
```

## Configuration Options

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "sonnet" | Model to use (sonnet, opus, haiku) |
| `permission_mode` | str | "plan" | Permission mode (plan, dontAsk, acceptEdits, etc.) |
| `system_prompt` | str | None | Override default system prompt |
| `append_system_prompt` | str | None | Append to default system prompt |
| `dangerously_skip_permissions` | bool | False | Bypass all permission checks |
| `allowed_tools` | list | None | Tools that can execute without prompts |
| `disallowed_tools` | list | None | Tools that are blocked |
| `working_directory` | str | None | Working directory for execution |
| `timeout_seconds` | float | 300 | Command timeout |
| `max_budget_usd` | float | None | Maximum budget limit |
| `cache` | bool | True | Enable response caching |
| `capture_transcript` | bool | True | Capture interaction transcript |
| `anthropic_api_key` | str | None | Explicit API key |
| `oauth_token` | str | None | OAuth token for Pro/Max subscribers |

### Permission Modes

| Mode | Description |
|------|-------------|
| `plan` | Allow planning but require approval for execution |
| `dontAsk` | Don't ask for permissions (use with caution) |
| `acceptEdits` | Auto-accept file edits |
| `default` | Default Claude Code behavior |
| `delegate` | Delegate permission decisions |
| `bypassPermissions` | Bypass all permissions (dangerous) |

## Authentication

The library supports two authentication methods:

1. **API Key**: Set `ANTHROPIC_API_KEY` environment variable or pass `anthropic_api_key` parameter
2. **OAuth Token**: Set `CLAUDE_CODE_OAUTH_TOKEN` or pass `oauth_token` parameter (for Pro/Max subscribers)

```python
# Using environment variables (recommended)
lm = ClaudeCodeLM(model="sonnet")  # Uses ANTHROPIC_API_KEY

# Explicit credentials
lm = ClaudeCodeLM(
    model="sonnet",
    anthropic_api_key="sk-...",
)
```

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py` - Simple DSPy integration
- `structured_output.py` - JSON schema structured outputs
- `with_tools.py` - Tool restriction configurations
- `async_usage.py` - Concurrent async requests

## Testing

```bash
# Run unit tests
uv run pytest tests/unit/ -v

# Run with coverage
uv run pytest tests/unit/ -v --cov=dspy_coding_agent_lms

# Run integration tests (requires Claude Code CLI)
RUN_INTEGRATION_TESTS=1 uv run pytest tests/integration/ -v
```

## API Reference

### ClaudeCodeLM

Main class for DSPy integration.

```python
class ClaudeCodeLM:
    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        json_schema: dict | None = None,
        **kwargs
    ) -> list[dict]:
        """Execute synchronous request."""

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        json_schema: dict | None = None,
        **kwargs
    ) -> list[dict]:
        """Execute asynchronous request."""

    @property
    def transcript(self) -> Transcript:
        """Access interaction transcript."""

    def copy(self, **kwargs) -> ClaudeCodeLM:
        """Create copy with modified parameters."""
```

### Transcript

Interaction history with aggregation methods.

```python
class Transcript:
    def total_cost_usd(self) -> float: ...
    def total_tokens(self) -> dict[str, int]: ...
    def total_duration_ms(self) -> int: ...
    def to_json(self) -> str: ...
    def save_to_file(self, path: str) -> None: ...
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- [DSPy](https://github.com/stanfordnlp/dspy) - The declarative programming framework for LMs
- [Claude Code](https://claude.ai/code) - Anthropic's agentic coding assistant
