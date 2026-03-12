#!/usr/bin/env python3
"""
Factual Evaluation Progress Checker

Analyzes factual-eval logs to show progress, accuracy, and time estimates.

Log structure:
  - mirothinker_N.json: final result files (array of statement verifications)
  - task_mirothinker_N_attempt_A_retry_R_chunk_C.json: per-chunk task files

Usage:
    python check_progress_factual_eval.py <log_folder> --total <N>

Example:
    python check_progress_factual_eval.py logs/factual-eval/20260306_1450 --total 100
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROGRESS_BAR_WIDTH = 20

CHUNK_FILE_PATTERN = re.compile(
    r"task_([a-zA-Z][a-zA-Z0-9_-]*)_(\d+)_attempt_(\d+)_retry_(\d+)_chunk_(\d+)\.json$"
)
# Result files must NOT start with "task_" to avoid matching chunk files
RESULT_FILE_PATTERN = re.compile(r"(?!task_)([a-zA-Z][a-zA-Z0-9_-]*)_(\d+)\.json$")


def create_progress_bar(percentage: float, width: int = PROGRESS_BAR_WIDTH) -> str:
    filled = int(width * percentage / 100)
    bar = "█" * filled + "░" * (width - filled)
    if percentage >= 80:
        color = "\033[92m"
    elif percentage >= 60:
        color = "\033[93m"
    elif percentage >= 40:
        color = "\033[33m"
    else:
        color = "\033[91m"
    return f"{color}[{bar}] {percentage:.1f}%\033[0m"


def format_duration(minutes: float) -> str:
    if minutes < 60:
        return f"{int(minutes)} minutes"
    elif minutes < 1440:
        return f"{minutes / 60:.1f} hours"
    else:
        return f"{minutes / 1440:.1f} days"


def parse_timestamp(time_str: str) -> Optional[datetime]:
    if not time_str:
        return None
    try:
        if time_str.endswith("Z"):
            time_str = time_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(time_str)
        return dt.replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def load_task_meta_fast(file_path: Path) -> Optional[dict]:
    """Load only task_meta from the beginning of the JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chunk = f.read(8192)

        start = chunk.find('"task_meta"')
        if start == -1:
            return None

        brace_start = chunk.find("{", start)
        if brace_start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False
        for i in range(brace_start, len(chunk)):
            c = chunk[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\" and in_string:
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(chunk[brace_start : i + 1])
        return None
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def analyze_result_file(file_path: Path) -> Dict[str, int]:
    """Analyze a mirothinker_N.json result file and count verdicts."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"total": 0, "right": 0, "wrong": 0, "conflict": 0, "unknown": 0}

    if not isinstance(data, list):
        return {"total": 0, "right": 0, "wrong": 0, "conflict": 0, "unknown": 0}

    right = sum(1 for d in data if d.get("verification", "").strip() == "Right")
    wrong = sum(1 for d in data if d.get("verification", "").strip() == "Wrong")
    conflict = sum(1 for d in data if d.get("verification", "").strip() == "Conflict")
    total = len(data)
    unknown = total - right - wrong - conflict
    return {"total": total, "right": right, "wrong": wrong, "conflict": conflict, "unknown": unknown}


def scan_log_folder(log_folder: Path) -> Tuple[
    str,
    Dict[int, Path],
    Dict[int, List[Path]],
]:
    """Scan log folder and return model name, result files and chunk files grouped by task_id."""
    result_files: Dict[int, Path] = {}
    chunk_files: Dict[int, List[Path]] = defaultdict(list)
    model_name = None

    for f in log_folder.iterdir():
        if not f.is_file() or not f.suffix == ".json":
            continue

        m = CHUNK_FILE_PATTERN.match(f.name)
        if m:
            if model_name is None:
                model_name = m.group(1)
            task_id = int(m.group(2))
            chunk_files[task_id].append(f)
            continue

        m = RESULT_FILE_PATTERN.match(f.name)
        if m:
            if model_name is None:
                model_name = m.group(1)
            result_files[int(m.group(2))] = f

    return model_name or "task", result_files, chunk_files


def main():
    parser = argparse.ArgumentParser(
        description="Factual Evaluation Progress Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("log_folder", help="Path to the log folder")
    parser.add_argument(
        "--total", "-n", type=int, required=True,
        help="Total number of tasks expected",
    )
    args = parser.parse_args()

    log_folder = Path(args.log_folder)
    total_expected = args.total

    if not log_folder.exists():
        print(f"Error: Log folder not found: {log_folder}")
        return 1

    print(f"Analyzing: {log_folder}")
    model_name, result_files, chunk_files = scan_log_folder(log_folder)

    # Completed tasks: those with a result file
    completed_ids = set(result_files.keys())
    # In-progress tasks: have chunk files but no result file
    all_chunk_ids = set(chunk_files.keys())
    in_progress_ids = all_chunk_ids - completed_ids

    # Analyze chunk statuses for in-progress tasks
    in_progress_details = {}
    for task_id in sorted(in_progress_ids):
        total_chunks = len(chunk_files[task_id])
        completed_chunks = 0
        running_chunks = 0
        for chunk_path in chunk_files[task_id]:
            meta = load_task_meta_fast(chunk_path)
            if meta:
                status = meta.get("status", "").lower()
                if status == "completed":
                    completed_chunks += 1
                elif status == "running":
                    running_chunks += 1
        in_progress_details[task_id] = {
            "total": total_chunks,
            "completed": completed_chunks,
            "running": running_chunks,
        }

    # Analyze result files for completed tasks
    task_verdicts = {}
    agg_right = 0
    agg_wrong = 0
    agg_conflict = 0
    agg_unknown = 0
    agg_total_statements = 0
    for task_id in sorted(completed_ids):
        verdicts = analyze_result_file(result_files[task_id])
        task_verdicts[task_id] = verdicts
        agg_right += verdicts["right"]
        agg_wrong += verdicts["wrong"]
        agg_conflict += verdicts["conflict"]
        agg_unknown += verdicts["unknown"]
        agg_total_statements += verdicts["total"]

    # Time estimation from chunk files (all tasks)
    earliest_start = None
    latest_end = None
    for task_id in all_chunk_ids:
        for chunk_path in chunk_files[task_id]:
            meta = load_task_meta_fast(chunk_path)
            if not meta:
                continue
            st = parse_timestamp(meta.get("start_time", ""))
            et = parse_timestamp(meta.get("end_time", ""))
            if st and (earliest_start is None or st < earliest_start):
                earliest_start = st
            if et and (latest_end is None or et > latest_end):
                latest_end = et

    num_completed = len(completed_ids)
    num_in_progress = len(in_progress_ids)
    num_remaining = total_expected - num_completed - num_in_progress

    # === Display ===
    print()
    print("=" * 80)
    print(f"FACTUAL-EVAL PROGRESS SUMMARY")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Progress
    completion_pct = num_completed / total_expected * 100 if total_expected > 0 else 0
    progress_bar = create_progress_bar(completion_pct)
    print()
    print("PROGRESS:")
    print(f"  Total Expected:     {total_expected}")
    print(f"  Completed:          {num_completed}")
    print(f"  In Progress:        {num_in_progress}")
    print(f"  Not Started:        {max(0, num_remaining)}")
    print(f"  Completion:         {num_completed}/{total_expected} {progress_bar}")

    # In-progress details
    if in_progress_details:
        print()
        print("IN-PROGRESS TASKS:")
        for task_id, detail in sorted(in_progress_details.items()):
            chunk_pct = (
                detail["completed"] / detail["total"] * 100
                if detail["total"] > 0 else 0
            )
            print(
                f"  {model_name}_{task_id}: "
                f"{detail['completed']}/{detail['total']} chunks done, "
                f"{detail['running']} running "
                f"({chunk_pct:.0f}%)"
            )

    # Statement-level accuracy
    if agg_total_statements > 0:
        accuracy = agg_right / agg_total_statements * 100
        accuracy_bar = create_progress_bar(accuracy)
        print()
        print("STATEMENT-LEVEL ACCURACY (completed tasks):")
        print(f"  Total Statements:   {agg_total_statements}")
        print(f"  Right:              {agg_right}")
        print(f"  Wrong:              {agg_wrong}")
        print(f"  Conflict:           {agg_conflict}")
        print(f"  Unknown:            {agg_unknown}")
        print(f"  Accuracy:           {agg_right}/{agg_total_statements} {accuracy_bar}")

    # Task-level average right ratio
    if task_verdicts:
        task_ratios = [
            v["right"] / v["total"] for v in task_verdicts.values() if v["total"] > 0
        ]
        if task_ratios:
            task_avg = sum(task_ratios) / len(task_ratios) * 100
            task_avg_bar = create_progress_bar(task_avg)
            print()
            print("TASK-LEVEL AVERAGE RIGHT RATIO:")
            print(f"  Tasks with results: {len(task_ratios)}")
            print(f"  Avg Right Ratio:    {task_avg_bar}")

    # Per-task breakdown
    if task_verdicts:
        print()
        print("PER-TASK BREAKDOWN:")
        col_w = max(len(f"{model_name}_{tid}") for tid in task_verdicts) + 2
        print(f"  {'Task':<{col_w}} {'Stmts':>6} {'Right':>6} {'Wrong':>6} {'Conflict':>8} {'Unknown':>8} {'Acc':>8}")
        print(f"  {'-'*col_w} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
        for task_id in sorted(task_verdicts.keys()):
            v = task_verdicts[task_id]
            acc = v["right"] / v["total"] * 100 if v["total"] > 0 else 0
            label = f"{model_name}_{task_id}"
            print(
                f"  {label:<{col_w}} {v['total']:>6} {v['right']:>6} "
                f"{v['wrong']:>6} {v['conflict']:>8} {v['unknown']:>8} {acc:>7.1f}%"
            )

    # Time estimation
    print()
    print("TIME ESTIMATION:")
    if earliest_start and latest_end and num_completed > 0:
        elapsed = latest_end - earliest_start
        elapsed_minutes = elapsed.total_seconds() / 60
        avg_minutes_per_task = elapsed_minutes / num_completed
        tasks_per_minute = num_completed / elapsed_minutes if elapsed_minutes > 0 else 0

        print(f"  Elapsed Time:       {format_duration(elapsed_minutes)}")
        print(f"  Completion Rate:    {tasks_per_minute:.2f} tasks/min")
        print(f"  Avg Time/Task:      {avg_minutes_per_task:.1f} min")

        remaining_count = max(0, num_remaining) + num_in_progress
        if remaining_count > 0:
            est_remaining = remaining_count * avg_minutes_per_task
            print(f"  Est. Remaining:     ~{format_duration(est_remaining)}")
        else:
            print("  Est. Remaining:     All tasks completed!")
    else:
        print("  Cannot estimate (no completed tasks with timing data)")

    print()
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
