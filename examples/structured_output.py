#!/usr/bin/env python3
"""Structured output example using JSON schema and DSPy Signatures.

This example demonstrates two approaches to structured outputs:
1. Direct JSON schema via Claude Code's --json-schema flag
2. DSPy Signatures with typed fields

Requirements:
    - Claude Code CLI installed and authenticated
    - dspy-coding-agent-lms installed

Run:
    python examples/structured_output.py
"""

from __future__ import annotations

import json
from typing import Literal

import dspy

from dspy_coding_agent_lms import ClaudeCodeLM

# =============================================================================
# DSPy Signature Definitions
# =============================================================================


class MathProblemSignature(dspy.Signature):
    """Solve a math problem and explain the solution."""

    problem: str = dspy.InputField(desc="The math problem to solve")
    answer: float = dspy.OutputField(desc="The numerical answer")
    explanation: str = dspy.OutputField(desc="Step-by-step explanation of the solution")


class SentimentAnalysisSignature(dspy.Signature):
    """Analyze the sentiment of a given text."""

    text: str = dspy.InputField(desc="The text to analyze")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = dspy.OutputField(
        desc="Overall sentiment classification"
    )
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")
    keywords: list[str] = dspy.OutputField(
        desc="Key words/phrases that indicate the sentiment"
    )


class EntityExtractionSignature(dspy.Signature):
    """Extract named entities from text."""

    text: str = dspy.InputField(desc="The text to extract entities from")
    people: list[str] = dspy.OutputField(desc="Names of people mentioned")
    organizations: list[str] = dspy.OutputField(desc="Organization names mentioned")
    locations: list[str] = dspy.OutputField(desc="Location names mentioned")
    dates: list[str] = dspy.OutputField(desc="Dates or time references mentioned")


# =============================================================================
# Example Functions
# =============================================================================


def example_math_problem_direct(lm: ClaudeCodeLM) -> None:
    """Example 1a: Math problem with direct JSON schema."""
    print("Example 1a: Math Problem (Direct JSON Schema)")
    print("-" * 40)

    math_schema = {
        "type": "object",
        "properties": {
            "problem": {"type": "string", "description": "The math problem"},
            "answer": {"type": "number", "description": "The numerical answer"},
            "explanation": {"type": "string", "description": "Step-by-step explanation"},
        },
        "required": ["problem", "answer"],
    }

    result = lm.forward(
        prompt="Solve: What is 15% of 240?",
        json_schema=math_schema,
    )

    data = json.loads(result.choices[0].message.content)
    print(f"Problem: {data.get('problem', 'N/A')}")
    print(f"Answer: {data['answer']}")
    if "explanation" in data:
        print(f"Explanation: {data['explanation']}")
    print()


def example_math_problem_signature(lm: ClaudeCodeLM) -> None:
    """Example 1b: Math problem with DSPy Signature."""
    print("Example 1b: Math Problem (DSPy Signature)")
    print("-" * 40)

    # Configure DSPy with our LM
    dspy.configure(lm=lm)

    # Create a predictor from the signature
    solve_math = dspy.Predict(MathProblemSignature)

    # Run the prediction
    result = solve_math(problem="What is 15% of 240?")

    print(f"Answer: {result.answer}")
    print(f"Explanation: {result.explanation}")
    print()


def example_sentiment_direct(lm: ClaudeCodeLM) -> None:
    """Example 2a: Sentiment analysis with direct JSON schema."""
    print("Example 2a: Sentiment Analysis (Direct JSON Schema)")
    print("-" * 40)

    sentiment_schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral", "mixed"],
                "description": "Overall sentiment",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence score (0-1)",
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key words/phrases that indicate sentiment",
            },
        },
        "required": ["sentiment", "confidence"],
    }

    text = "I absolutely love this product! It exceeded my expectations."
    result = lm.forward(
        prompt=f"Analyze the sentiment of this text: '{text}'",
        json_schema=sentiment_schema,
    )

    data = json.loads(result.choices[0].message.content)
    print(f"Text: {text}")
    print(f"Sentiment: {data['sentiment']}")
    print(f"Confidence: {data['confidence']:.1%}")
    if "keywords" in data:
        print(f"Keywords: {', '.join(data['keywords'])}")
    print()


def example_sentiment_signature(lm: ClaudeCodeLM) -> None:
    """Example 2b: Sentiment analysis with DSPy Signature."""
    print("Example 2b: Sentiment Analysis (DSPy Signature)")
    print("-" * 40)

    dspy.configure(lm=lm)

    analyze_sentiment = dspy.Predict(SentimentAnalysisSignature)

    text = "I absolutely love this product! It exceeded my expectations."
    result = analyze_sentiment(text=text)

    print(f"Text: {text}")
    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Keywords: {', '.join(result.keywords)}")
    print()


def example_entity_extraction_direct(lm: ClaudeCodeLM) -> None:
    """Example 3a: Entity extraction with direct JSON schema."""
    print("Example 3a: Entity Extraction (Direct JSON Schema)")
    print("-" * 40)

    entity_schema = {
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of people mentioned",
            },
            "organizations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Organization names mentioned",
            },
            "locations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Location names mentioned",
            },
            "dates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Dates or time references mentioned",
            },
        },
        "required": ["people", "organizations", "locations"],
    }

    article = """
    Apple CEO Tim Cook announced yesterday that the company will open
    a new headquarters in Austin, Texas. The facility, expected to open
    in 2025, will employ over 5,000 workers. Cook met with Governor
    Greg Abbott to discuss the plans.
    """

    result = lm.forward(
        prompt=f"Extract entities from this text:\n{article}",
        json_schema=entity_schema,
    )

    data = json.loads(result.choices[0].message.content)
    print("Text:", article.strip()[:80], "...")
    print(f"People: {data.get('people', [])}")
    print(f"Organizations: {data.get('organizations', [])}")
    print(f"Locations: {data.get('locations', [])}")
    if "dates" in data:
        print(f"Dates: {data['dates']}")
    print()


def example_entity_extraction_signature(lm: ClaudeCodeLM) -> None:
    """Example 3b: Entity extraction with DSPy Signature."""
    print("Example 3b: Entity Extraction (DSPy Signature)")
    print("-" * 40)

    dspy.configure(lm=lm)

    extract_entities = dspy.Predict(EntityExtractionSignature)

    article = """
    Apple CEO Tim Cook announced yesterday that the company will open
    a new headquarters in Austin, Texas. The facility, expected to open
    in 2025, will employ over 5,000 workers. Cook met with Governor
    Greg Abbott to discuss the plans.
    """

    result = extract_entities(text=article)

    print("Text:", article.strip()[:80], "...")
    print(f"People: {result.people}")
    print(f"Organizations: {result.organizations}")
    print(f"Locations: {result.locations}")
    print(f"Dates: {result.dates}")
    print()


def example_chain_of_thought(lm: ClaudeCodeLM) -> None:
    """Example 4: Using ChainOfThought for reasoning with structured output."""
    print("Example 4: Chain of Thought with Structured Output")
    print("-" * 40)

    dspy.configure(lm=lm)

    # ChainOfThought adds reasoning before the answer
    solve_with_reasoning = dspy.ChainOfThought(MathProblemSignature)

    result = solve_with_reasoning(
        problem="A store has a 20% off sale. If an item originally costs $85, "
        "and you have a $10 coupon to use after the discount, what do you pay?"
    )

    print(f"Reasoning: {result.reasoning}")
    print(f"Answer: ${result.answer:.2f}")
    print(f"Explanation: {result.explanation}")
    print()


def example_custom_signature_inline() -> None:
    """Example 5: Defining signatures inline (shorthand syntax)."""
    print("Example 5: Inline Signature Definition")
    print("-" * 40)

    lm = ClaudeCodeLM(model="sonnet")
    dspy.configure(lm=lm)

    # Shorthand signature syntax: "input_field -> output_field"
    translate = dspy.Predict("english: str -> french: str, formal: bool")

    result = translate(english="Hello, how are you?")
    print("English: Hello, how are you?")
    print(f"French: {result.french}")
    print(f"Formal: {result.formal}")
    print()


def example_json_adapter() -> None:
    """Example 6: Using JSONAdapter for native JSON schema output.

    By default, DSPy uses ChatAdapter which formats outputs as:
        [[ ## field ## ]]
        value

    With JSONAdapter, DSPy uses native JSON schema output via
    Claude Code's --json-schema flag, returning structured JSON directly.
    """
    print("Example 6: JSONAdapter (Native JSON Schema)")
    print("-" * 40)
    print("Comparison: ChatAdapter vs JSONAdapter")
    print()

    text = "The weather is beautiful today, but traffic was terrible."

    # ChatAdapter (default) - uses string parsing
    lm1 = ClaudeCodeLM(model="sonnet")
    dspy.configure(lm=lm1, adapter=dspy.ChatAdapter())
    result1 = dspy.Predict(SentimentAnalysisSignature)(text=text)
    print("ChatAdapter (string parsing):")
    print(f"  Sentiment: {result1.sentiment}")
    print(f"  Output format: [[ ## sentiment ## ]]\\n{result1.sentiment}")
    print()

    # JSONAdapter - uses native JSON schema
    lm2 = ClaudeCodeLM(model="sonnet")
    dspy.configure(lm=lm2, adapter=dspy.JSONAdapter())
    result2 = dspy.Predict(SentimentAnalysisSignature)(text=text)
    print("JSONAdapter (native JSON schema):")
    print(f"  Sentiment: {result2.sentiment}")
    print(f"  Output format: {{\"sentiment\": \"{result2.sentiment}\", ...}}")
    print()

    # Show the raw structured_output from Claude Code
    if lm2.transcript.last.response.get("structured_output"):
        print("Raw structured_output from Claude Code:")
        print(f"  {lm2.transcript.last.response['structured_output']}")
    print()


def main() -> None:
    """Run all structured output examples."""
    # Initialize LM (shared across examples)
    lm = ClaudeCodeLM(model="sonnet")

    print("=" * 60)
    print("Structured Output Examples: JSON Schema vs DSPy Signatures")
    print("=" * 60)
    print()
    print("This example shows three approaches to structured outputs:")
    print("  1. Direct JSON Schema: Uses Claude Code's --json-schema flag directly")
    print("  2. DSPy Signatures + ChatAdapter: Uses string parsing with [[ ## field ## ]]")
    print("  3. DSPy Signatures + JSONAdapter: Uses native JSON schema via --json-schema")
    print()

    # Run all examples
    example_math_problem_direct(lm)
    example_math_problem_signature(lm)

    example_sentiment_direct(lm)
    example_sentiment_signature(lm)

    example_entity_extraction_direct(lm)
    example_entity_extraction_signature(lm)

    example_chain_of_thought(lm)
    example_custom_signature_inline()

    # New: JSONAdapter example
    example_json_adapter()

    # Session summary
    print("=" * 60)
    print("Session Summary:")
    print(f"  Total requests: {len(lm.transcript)}")
    print(f"  Total cost: ${lm.transcript.total_cost_usd():.4f}")

    tokens = lm.transcript.total_tokens()
    print(f"  Input tokens: {tokens['input']:,}")
    print(f"  Output tokens: {tokens['output']:,}")
    print(f"  Cache tokens: {tokens['cache_read']:,}")


if __name__ == "__main__":
    main()
