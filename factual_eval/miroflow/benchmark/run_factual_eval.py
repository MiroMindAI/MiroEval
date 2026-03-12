# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
Entry point for running factual evaluation benchmark.

Usage:
    python -m miroflow.benchmark.run_factual_eval --config-path config/benchmark_factual-eval_mirothinker.yaml
"""

import argparse
import asyncio
import json
import os
import signal
from pathlib import Path

import dotenv
from omegaconf import DictConfig, OmegaConf

from config import load_config
from miroflow.benchmark.eval_utils import Task, STATUS_FAILED
from miroflow.benchmark.factual_eval_task_runner import run_factual_eval_tasks
from miroflow.logging.task_tracer import set_tracer


def _signal_handler(signum, frame):
    signal_name = signal.Signals(signum).name
    print(f"\nReceived {signal_name}, exiting...")
    os._exit(128 + signum)


def _extract_query_and_response(data: dict) -> tuple[str, str, dict]:
    """
    Extract user_query and response from a single JSON data object.

    Supports multiple formats:
    - Deep Research format: entries[0].query / entries[0].response
    - Flat format: query/response or user_query/response

    Returns:
        (user_query, response, extra_metadata)
    """
    user_query = ""
    response = ""
    extra_metadata = {}

    if "entries" in data and len(data["entries"]) > 0:
        entry = data["entries"][0]
        user_query = entry.get("query", "")
        response = entry.get("response", "")
        # Preserve useful metadata from Deep Research format
        if "query_id" in data:
            extra_metadata["query_id"] = data["query_id"]
        if "topic" in data:
            extra_metadata["topic"] = data["topic"]
        if "language" in data:
            extra_metadata["language"] = data["language"]
    elif "query" in data and "response" in data:
        user_query = data["query"]
        response = data["response"]
    elif "user_query" in data and "response" in data:
        user_query = data["user_query"]
        response = data["response"]

    return user_query, response, extra_metadata


def load_factual_eval_tasks(cfg: DictConfig) -> list[Task]:
    """
    Load factual eval tasks from the data directory.

    Supports four data layouts:
    0. source_file mode: Load a specific JSON array file from data_dir
       (shared MiroEval data format: [{id, query, response, ...}, ...])
    1. standardized_data.jsonl in data_dir: Each line is a JSON object with
       task_id, task_question, ground_truth, metadata.user_query
    2. data_dir/{model_name}/*.json: Sub-directories by model, each containing
       raw JSON files (e.g., Deep Research format with entries[0].query/response)
    3. data_dir/*.json: Flat directory with raw JSON files
    """
    data_cfg = cfg.benchmark.data
    data_dir = Path(data_cfg.get("data_dir", "data/factual-eval"))

    # Mode 0: source_file — load a specific JSON array file from the shared data directory
    source_file = data_cfg.get("source_file", None)
    if source_file:
        source_path = data_dir / source_file
        if source_path.exists():
            tasks = _load_json_array_tasks(source_path)
            if tasks:
                print(f"Loaded {len(tasks)} tasks from {source_path}")
                return tasks
        else:
            print(f"source_file not found: {source_path}")

    # Try JSONL format first
    metadata_file = data_dir / data_cfg.get("metadata_file", "standardized_data.jsonl")
    if metadata_file.exists():
        tasks = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                tasks.append(
                    Task(
                        task_id=data["task_id"],
                        task_question=data["task_question"],
                        ground_truth=data.get("ground_truth", ""),
                        file_path=data.get("file_path"),
                        metadata=data.get("metadata", {}),
                    )
                )
        print(f"Loaded {len(tasks)} tasks from {metadata_file}")
        return tasks

    # Check for sub-directory structure: data_dir/{model_name}/*.json
    subdirs = sorted(
        [d for d in data_dir.iterdir() if d.is_dir()]
    ) if data_dir.exists() else []

    if subdirs:
        tasks = []
        for model_dir in subdirs:
            model_name = model_dir.name
            json_files = sorted(model_dir.glob("*.json"))
            for json_file in json_files:
                task = _load_single_json_task(json_file, model_name)
                if task:
                    tasks.append(task)
            print(f"  Loaded {len(json_files)} files from {model_name}/")
        print(f"Loaded {len(tasks)} tasks total from {len(subdirs)} model(s)")
        return tasks

    # Flat directory: data_dir/*.json (supports both single-object and array files)
    json_files = sorted(data_dir.glob("*.json")) if data_dir.exists() else []
    if not json_files:
        print(f"No data found in {data_dir}")
        return []

    tasks = []
    for json_file in json_files:
        # Try loading as JSON array first
        array_tasks = _load_json_array_tasks(json_file)
        if array_tasks:
            tasks.extend(array_tasks)
        else:
            task = _load_single_json_task(json_file)
            if task:
                tasks.append(task)

    print(f"Loaded {len(tasks)} tasks from {data_dir}")
    return tasks


def _load_json_array_tasks(json_file: Path) -> list[Task]:
    """Load tasks from a JSON array file (shared MiroEval data format).

    Each entry has: {id, query, rewritten_query, response, process, files, annotation, ...}
    The response field is fact-checked as the task_question.
    Returns empty list if the file is not a JSON array or entries lack required fields.
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        model_name = json_file.stem  # e.g., "mirothinker_v17_text_100"
        tasks = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            response = entry.get("response", "")
            query = entry.get("rewritten_query") or entry.get("query", "")
            if not response:
                continue

            entry_id = entry.get("id", "")
            task_id = f"{model_name}/{entry_id}"
            metadata = {
                "user_query": query,
                "source_file": str(json_file),
                "entry_id": entry_id,
            }
            annotation = entry.get("annotation", {})
            if annotation:
                metadata["annotation"] = annotation

            file_paths = entry.get("files")
            tasks.append(
                Task(
                    task_id=task_id,
                    task_question=response,
                    ground_truth="",
                    file_path=file_paths if file_paths else None,
                    metadata=metadata,
                )
            )
        return tasks
    except Exception as e:
        print(f"Error loading JSON array from {json_file}: {e}")
        return []


def _load_single_json_task(
    json_file: Path, model_name: str = ""
) -> Task | None:
    """Load a single JSON file into a Task object."""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        user_query, response, extra_metadata = _extract_query_and_response(data)

        if not user_query or not response:
            print(f"Skipping {json_file.name}: missing query or response")
            return None

        # Build task_id: {model_name}/{file_stem} or just {file_stem}
        if model_name:
            task_id = f"{model_name}/{json_file.stem}"
        else:
            task_id = json_file.stem

        metadata = {
            "user_query": user_query,
            "source_file": str(json_file),
            **extra_metadata,
        }
        if model_name:
            metadata["input_model"] = model_name

        return Task(
            task_id=task_id,
            task_question=response,
            ground_truth="",
            file_path=data.get("file_path") or data.get("file_paths") or data.get("attachments"),
            metadata=metadata,
        )
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


async def run_factual_eval(cfg: DictConfig):
    """Main entry point for factual evaluation."""
    print("Factual eval configuration:\n", OmegaConf.to_yaml(cfg, resolve=True))

    # Load tasks
    tasks = load_factual_eval_tasks(cfg)
    if not tasks:
        print("No tasks loaded. Exiting.")
        return

    # Apply whitelist filter if configured
    whitelist = cfg.benchmark.data.get("whitelist", [])
    if whitelist:
        whitelist_set = set(whitelist)
        tasks = [t for t in tasks if t.task_id in whitelist_set]
        print(f"After whitelist filter: {len(tasks)} tasks")

    # Apply max_tasks limit
    max_tasks = cfg.benchmark.execution.get("max_tasks", None)
    if max_tasks and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]
        print(f"Limited to {max_tasks} tasks")

    execution_cfg = cfg.benchmark.execution
    results = run_factual_eval_tasks(
        cfg=cfg,
        tasks=tasks,
        max_concurrent=execution_cfg.get("max_concurrent", 3),
        max_concurrent_chunks=execution_cfg.get("max_concurrent_chunks", 20),
        max_retries=execution_cfg.get("max_retries", 5),
        chunk_timeout=execution_cfg.get("chunk_timeout", None),
        chunk_tracing=execution_cfg.get("chunk_tracing", True),
    )

    # Summary
    completed = sum(1 for r in results if r.status == "completed")
    failed = sum(1 for r in results if r.status == STATUS_FAILED)
    print(f"\nFactual eval completed: {completed} succeeded, {failed} failed")

    # Save summary
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "factual_eval_summary.jsonl"
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    parser = argparse.ArgumentParser(description="Run factual evaluation")
    parser.add_argument(
        "--config-path", type=str, default="", help="Configuration file path"
    )
    parser.add_argument("overrides", nargs="*", help="Configuration overrides")
    args = parser.parse_args()

    dotenv.load_dotenv()

    cfg = load_config(args.config_path, *args.overrides)
    set_tracer(cfg.output_dir)

    asyncio.run(run_factual_eval(cfg))
