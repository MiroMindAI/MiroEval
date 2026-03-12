#!/usr/bin/env python3
"""
Run a single factual evaluation task for debugging.

Usage:
    # By file path (most common)
    uv run scripts/run_single_factual_eval_task.py --file data/factual-eval/gemini_2.5_pro/deep_research_1_20250819_222419.json

    # By task index (0-based, from loaded tasks)
    uv run scripts/run_single_factual_eval_task.py --task-index 0

    # With custom config
    uv run scripts/run_single_factual_eval_task.py --file data.json --config config/benchmark_factual-eval_mirothinker.yaml
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import dotenv
from omegaconf import OmegaConf

from config import load_config
from miroflow.agents import build_agent, build_agent_from_config
from miroflow.benchmark.eval_utils import Task
from miroflow.benchmark.factual_eval_task_runner import run_factual_eval_task
from miroflow.benchmark.run_factual_eval import (
    _extract_query_and_response,
    _load_single_json_task,
    load_factual_eval_tasks,
)
from miroflow.logging.task_tracer import set_tracer, get_tracer


async def run_single(cfg, task):
    """Run a single factual eval task and print results."""
    print("=" * 80)
    print("Factual Eval - Single Task Debug")
    print("=" * 80)
    print(f"Task ID:      {task.task_id}")
    print(f"User Query:   {task.metadata.get('user_query', '')[:120]}...")
    print(f"Report:       {len(task.task_question)} chars")
    print(f"Model:        {task.metadata.get('input_model', 'N/A')}")
    print("=" * 80)

    # Build agent
    print("\nBuilding agent...")
    agent = build_agent_from_config(cfg=cfg)
    print(f"Agent: {agent.__class__.__name__}")

    # Build segment processor (test it separately first)
    print("\nBuilding segment processor...")
    seg_processor = build_agent(cfg["segment_processor"])
    print(f"Segment processor: {seg_processor.__class__.__name__}")

    # Run segmentation first to verify
    print("\nSegmenting report...")
    seg_result = await seg_processor.run({"task_description": task.task_question})
    segments = seg_result.get("segments", [])
    print(f"Segmented into {len(segments)} parts:")
    for i, seg in enumerate(segments):
        print(f"  [{i}] {seg[:80]}{'...' if len(seg) > 80 else ''}")

    # Run full factual eval
    print("\n" + "-" * 80)
    print("Running factual evaluation on all segments...")
    print("-" * 80)

    execution_cfg = cfg.benchmark.execution
    result = await run_factual_eval_task(
        cfg=cfg,
        agent=agent,
        task=task,
        max_concurrent_chunks=execution_cfg.get("max_concurrent_chunks", 20),
        max_retries=execution_cfg.get("max_retries", 5),
    )

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Status: {result.status}")

    if result.model_response:
        try:
            core_state = json.loads(result.model_response)
            print(f"Verified statements: {len(core_state)}")
            for i, stmt in enumerate(core_state):
                verification = stmt.get("verification", "?")
                statement = stmt.get("statement", "")[:80]
                print(f"  [{verification:>7}] {statement}...")
        except json.JSONDecodeError:
            print(f"Raw response: {result.model_response[:500]}")

    if result.error_message:
        print(f"Error: {result.error_message}")

    # Output location
    output_path = Path(cfg.output_dir) / f"{task.task_id}.json"
    if output_path.exists():
        print(f"\nOutput: {output_path}")

    print("=" * 80)
    return result


def main():
    parser = argparse.ArgumentParser(description="Debug a single factual eval task")
    parser.add_argument(
        "--config",
        type=str,
        default="config/benchmark_factual-eval_gpt5-mini.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single JSON data file to evaluate",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        help="Index of task from loaded data (0-based)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: logs/factual-eval/debug_{timestamp})",
    )

    args = parser.parse_args()
    dotenv.load_dotenv()

    # Set output dir
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"logs/factual-eval/debug_{timestamp}"
    else:
        output_dir = args.output_dir

    cfg = load_config(args.config, f"output_dir={output_dir}")
    set_tracer(cfg.output_dir)

    # Determine task
    task = None

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: file not found: {file_path}")
            sys.exit(1)
        task = _load_single_json_task(file_path, model_name=file_path.parent.name)
        if task is None:
            print(f"Error: could not parse task from {file_path}")
            sys.exit(1)

    elif args.task_index is not None:
        tasks = load_factual_eval_tasks(cfg)
        if 0 <= args.task_index < len(tasks):
            task = tasks[args.task_index]
        else:
            print(f"Error: index {args.task_index} out of range (0-{len(tasks)-1})")
            sys.exit(1)

    else:
        print("Error: must provide --file or --task-index")
        parser.print_help()
        sys.exit(1)

    result = asyncio.run(run_single(cfg, task))
    sys.exit(0 if result.status == "completed" else 1)


if __name__ == "__main__":
    main()
