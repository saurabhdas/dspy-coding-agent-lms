#!/usr/bin/env python3
"""Async usage example for concurrent requests.

This example demonstrates how to use the async interface for
making concurrent requests to Claude Code, improving throughput
for batch operations.

Requirements:
    - Claude Code CLI installed and authenticated
    - dspy-coding-agent-lms installed

Run:
    python examples/async_usage.py
"""

from __future__ import annotations

import asyncio
import time

from dspy_coding_agent_lms import ClaudeCodeLM


async def single_async_request() -> None:
    """Example: Single async request."""
    print("Example 1: Single Async Request")
    print("-" * 40)

    lm = ClaudeCodeLM(model="sonnet", cache=False)

    result = await lm.aforward(prompt="What is the square root of 144?")
    print(f"Result: {result.choices[0].message.content}")
    print()


async def concurrent_requests() -> None:
    """Example: Multiple concurrent requests."""
    print("Example 2: Concurrent Requests")
    print("-" * 40)

    lm = ClaudeCodeLM(model="sonnet", cache=False)

    prompts = [
        "What is Python primarily used for?",
        "What is JavaScript primarily used for?",
        "What is Rust primarily used for?",
        "What is Go primarily used for?",
    ]

    print(f"Sending {len(prompts)} requests concurrently...")
    start_time = time.time()

    # Execute all requests concurrently
    tasks = [lm.aforward(prompt=p) for p in prompts]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    print()

    for prompt, result in zip(prompts, results, strict=True):
        answer = result.choices[0].message.content[:100]
        print(f"Q: {prompt}")
        print(f"A: {answer}...")
        print()

    print(f"Total cost: ${lm.transcript.total_cost_usd():.4f}")
    print()


async def sequential_vs_concurrent() -> None:
    """Example: Compare sequential vs concurrent performance."""
    print("Example 3: Sequential vs Concurrent Comparison")
    print("-" * 40)

    lm = ClaudeCodeLM(model="sonnet", cache=False)

    prompts = [
        "Count to 3",
        "Count to 3",
        "Count to 3",
    ]

    # Sequential execution
    print("Sequential execution...")
    start_seq = time.time()
    seq_results = []
    for p in prompts:
        result = await lm.aforward(prompt=p)
        seq_results.append(result)
    seq_time = time.time() - start_seq
    print(f"Sequential time: {seq_time:.2f}s")

    # Clear cache for fair comparison
    lm.clear_cache()
    lm._transcript.clear()

    # Concurrent execution
    print("Concurrent execution...")
    start_conc = time.time()
    tasks = [lm.aforward(prompt=p) for p in prompts]
    await asyncio.gather(*tasks)
    conc_time = time.time() - start_conc
    print(f"Concurrent time: {conc_time:.2f}s")

    # Comparison
    speedup = seq_time / conc_time if conc_time > 0 else 0
    print()
    print(f"Speedup: {speedup:.2f}x faster with concurrent execution")
    print()


async def batch_processing() -> None:
    """Example: Batch processing with controlled concurrency."""
    print("Example 4: Batch Processing with Semaphore")
    print("-" * 40)

    lm = ClaudeCodeLM(model="sonnet", cache=False)

    # Large batch of items to process
    items = [f"Item {i}" for i in range(10)]

    # Limit concurrent requests to avoid overwhelming the system
    max_concurrent = 3
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(item: str) -> dict:
        """Process item with concurrency limit."""
        async with semaphore:
            result = await lm.aforward(
                prompt=f"Describe '{item}' in one sentence."
            )
            return {"item": item, "result": result.choices[0].message.content}

    print(f"Processing {len(items)} items with max {max_concurrent} concurrent...")
    start_time = time.time()

    tasks = [process_with_limit(item) for item in items]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    print()

    # Show first few results
    for r in results[:3]:
        print(f"  {r['item']}: {r['result'][:50]}...")

    print(f"  ... and {len(results) - 3} more")
    print()
    print(f"Total cost: ${lm.transcript.total_cost_usd():.4f}")


async def main() -> None:
    """Run all async examples."""
    print("Async Usage Examples")
    print("=" * 50)
    print()

    await single_async_request()
    await concurrent_requests()
    await sequential_vs_concurrent()
    await batch_processing()

    print("=" * 50)
    print("Async examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
