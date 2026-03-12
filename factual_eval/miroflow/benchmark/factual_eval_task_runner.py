# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
Factual evaluation task runner.

Implements the segment-parallel-merge pattern:
1. Segment a report into logical parts using LLM
2. Run agent on each part in parallel for factual verification
3. Merge results from all parts into a single output
"""

import asyncio
import gc
import json
import os
import re
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

from miroflow.agents import BaseAgent, build_agent, build_agent_from_config
from miroflow.benchmark.eval_utils import Task, TaskResult, STATUS_FAILED
from miroflow.logging.task_tracer import (
    get_tracer,
    TaskContextVar,
    set_current_task_context_var,
    reset_current_task_context_var,
)
from miroflow.utils.parsing_utils import robust_json_loads

logger = get_tracer()


def build_prompt(user_query: str, long_report: str) -> str:
    """Build the prompt for a single chunk with user_query and part delimiters."""
    return (
        "\n[user_query start]>>>\n"
        + user_query
        + "\n[user_query end]>>>\n"
        + "\n[part_start]>>>\n"
        + long_report
        + "\n[part_end]>>>\n"
    )


def extract_json_block(s: str) -> str:
    """Extract JSON object from a string, handling markdown code fences."""
    s = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.DOTALL
    )
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object braces found in input.")
    return s[start : end + 1]


async def run_factual_eval_task(
    cfg: DictConfig,
    agent: BaseAgent,
    task: Task,
    max_concurrent_chunks: int = 20,
    max_retries: int = 5,
    chunk_timeout: Optional[float] = None,
    chunk_tracing: bool = True,
) -> TaskResult:
    """
    Run factual evaluation on a single task.

    Segments the report, runs verification on each chunk in parallel,
    and merges the results.
    """
    result = TaskResult(task=task)
    report_text = task.task_question
    user_query = task.metadata.get("user_query", "")

    # Step 1: Segment the report
    segment_processor = build_agent(cfg["segment_processor"])
    try:
        seg_result = await segment_processor.run({"task_description": report_text})
    except Exception as e:
        result.status = STATUS_FAILED
        result.error_message = f"Segmentation failed: {e}"
        print(f"Task {task.task_id}: segmentation failed: {e}")
        return result

    segments = seg_result.get("segments", [])
    if not segments:
        result.status = STATUS_FAILED
        result.error_message = "Segmentation returned empty segments"
        return result

    print(f"Task {task.task_id}: segmented into {len(segments)} parts")

    # Step 2: Process each chunk in parallel
    semaphore = asyncio.Semaphore(max_concurrent_chunks)
    tracer = get_tracer()

    async def process_chunk(idx: int, chunk_text: str):
        task_desc = build_prompt(user_query, chunk_text)

        # Set up tracing context for this chunk (only when chunk_tracing enabled)
        token = None
        if chunk_tracing:
            task_context_var = TaskContextVar(
                task_id=task.task_id,
                attempt_id=1,
                retry_id=0,
                suffix=f"chunk_{idx}",
            )
            token = set_current_task_context_var(task_context_var)
            tracer.update_task_meta(
                patch={
                    "task_id": task.task_id,
                    "chunk_idx": idx,
                    "chunk_text_preview": chunk_text[:200],
                    "user_query": user_query[:200],
                }
            )
            tracer.start()

        try:
            async with semaphore:
                # Compute attachment file paths (backward-compatible: None → [])
                file_paths = task.file_path if isinstance(task.file_path, list) else (
                    [task.file_path] if task.file_path else []
                )

                for attempt in range(max_retries):
                    try:
                        coro = agent.run(
                            {
                                "task_description": task_desc,
                                "task_file_name": file_paths[0] if file_paths else "",
                                "attachment_file_paths": file_paths,
                                "is_final_retry": True,
                            }
                        )
                        if chunk_timeout:
                            response = await asyncio.wait_for(coro, timeout=chunk_timeout)
                        else:
                            response = await coro
                        summary = response.get("summary", "")
                        if summary and summary != "No final answer generated.":
                            if chunk_tracing:
                                tracer.finish(status="completed")
                            return {"idx": idx, "ok": True, "summary": summary}
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Task {task.task_id} chunk {idx} timed out after {chunk_timeout}s"
                        )
                        if chunk_tracing:
                            tracer.finish(status="failed", error=f"Chunk timed out after {chunk_timeout}s")
                        return {"idx": idx, "ok": False, "error": f"Timed out after {chunk_timeout}s"}
                    except Exception as e:
                        logger.warning(
                            f"Task {task.task_id} chunk {idx} attempt {attempt + 1} failed: {e}"
                        )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                if chunk_tracing:
                    tracer.finish(status="failed", error="All attempts failed")
                return {"idx": idx, "ok": False, "error": "All attempts failed"}
        except Exception as e:
            if chunk_tracing:
                tracer.finish(status="failed", error=str(e))
            return {"idx": idx, "ok": False, "error": str(e)}
        finally:
            if token is not None:
                reset_current_task_context_var(token)

    chunk_results = await asyncio.gather(
        *[process_chunk(i, seg) for i, seg in enumerate(segments)],
        return_exceptions=False,
    )

    # Step 3: Merge results
    chunk_results = sorted(chunk_results, key=lambda x: x["idx"])
    merged_core_state = []
    for r in chunk_results:
        if r["ok"]:
            try:
                parsed = robust_json_loads(extract_json_block(r["summary"]))
                if "core_state" in parsed and isinstance(parsed["core_state"], list):
                    merged_core_state.extend(parsed["core_state"])
                    logger.info(f"Task {task.task_id} chunk {r['idx']} success")
                else:
                    merged_core_state.append(parsed)
            except Exception as e:
                logger.error(
                    f"Task {task.task_id} chunk {r['idx']} parse error: {e}\n"
                    f"Raw: {r['summary'][:200]}"
                )
        else:
            logger.error(
                f"Task {task.task_id} chunk {r['idx']} failed: {r.get('error', 'unknown')}"
            )

    # Step 4: Save output (task_id may contain '/' for model sub-dirs)
    output_path = Path(cfg.output_dir) / f"{task.task_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_core_state, f, ensure_ascii=False, indent=2)

    print(
        f"Task {task.task_id}: merged {len(merged_core_state)} statements, "
        f"saved to {output_path}"
    )

    # Build TaskResult
    result.model_response = json.dumps(merged_core_state, ensure_ascii=False)
    result.status = "completed"
    return result


# ---- Process-level parallelism for multiple tasks ----


def _worker_signal_handler(signum, frame):
    sys.exit(128 + signum)


def _factual_eval_task_worker(task_dict, cfg_dict, max_concurrent_chunks, max_retries, chunk_timeout=None, chunk_tracing=True):
    """Worker function for ProcessPoolExecutor. Runs a single factual eval task."""
    from miroflow.agents import build_agent_from_config
    from miroflow.logging.task_tracer import set_tracer
    from miroflow.benchmark.eval_utils import Task

    signal.signal(signal.SIGTERM, _worker_signal_handler)
    signal.signal(signal.SIGINT, _worker_signal_handler)

    cfg = OmegaConf.create(cfg_dict)
    task = Task.from_dict(task_dict)

    set_tracer(cfg.output_dir)

    agent = build_agent_from_config(cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(lambda _loop, _context: None)

    try:
        result = loop.run_until_complete(
            run_factual_eval_task(
                cfg=cfg,
                agent=agent,
                task=task,
                max_concurrent_chunks=max_concurrent_chunks,
                max_retries=max_retries,
                chunk_timeout=chunk_timeout,
                chunk_tracing=chunk_tracing,
            )
        )
        return result.to_dict()
    finally:
        loop.close()
        gc.collect()


def run_factual_eval_tasks(
    cfg: DictConfig,
    tasks: List[Task],
    max_concurrent: int = 3,
    max_concurrent_chunks: int = 20,
    max_retries: int = 5,
    chunk_timeout: Optional[float] = None,
    chunk_tracing: bool = True,
) -> List[TaskResult]:
    """
    Run factual eval on multiple tasks in parallel using ProcessPoolExecutor.

    Args:
        cfg: Full configuration object.
        tasks: List of tasks to process.
        max_concurrent: Number of tasks to process in parallel (process-level).
        max_concurrent_chunks: Number of chunks to process in parallel within a task.
        max_retries: Max retries per chunk.

    Returns:
        List of TaskResult objects.
    """
    print(
        f"Running factual eval on {len(tasks)} tasks with "
        f"max_concurrent={max_concurrent}, max_concurrent_chunks={max_concurrent_chunks}"
    )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    results_dict = {}
    executor = None

    # Resume: skip tasks whose output files already exist
    pending_tasks = []
    for task in tasks:
        output_path = Path(cfg.output_dir) / f"{task.task_id}.json"
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                result = TaskResult(task=task)
                result.model_response = json.dumps(existing_data, ensure_ascii=False)
                result.status = "completed"
                results_dict[task.task_id] = result
                print(f"Task {task.task_id}: skipped (output already exists)")
                continue
            except Exception as e:
                print(f"Task {task.task_id}: existing output unreadable ({e}), will re-run")
        pending_tasks.append(task)

    skipped = len(tasks) - len(pending_tasks)
    print(f"Resume: {skipped} tasks skipped, {len(pending_tasks)} tasks to run")

    try:
        mp_context = get_context("forkserver") if sys.platform == "linux" else None
        executor = ProcessPoolExecutor(
            max_workers=max_concurrent, mp_context=mp_context
        )

        future_to_task_id = {
            executor.submit(
                _factual_eval_task_worker,
                task.to_dict(),
                cfg_dict,
                max_concurrent_chunks,
                max_retries,
                chunk_timeout,
                chunk_tracing,
            ): task.task_id
            for task in pending_tasks
        }

        for future in as_completed(future_to_task_id):
            task_id = future_to_task_id[future]
            try:
                result_dict = future.result()
                result = TaskResult.from_dict(result_dict)
                results_dict[task_id] = result
                print(
                    f"Progress: {len(results_dict)}/{len(tasks)} tasks completed"
                )
            except Exception as e:
                print(f"Exception in task {task_id}: {e}")
                task_dict = next(
                    t.to_dict()
                    for t in tasks
                    if t.task_id == task_id
                )
                error_result = TaskResult(task=Task.from_dict(task_dict))
                error_result.status = STATUS_FAILED
                error_result.error_message = str(e)
                results_dict[task_id] = error_result

    except KeyboardInterrupt:
        print("\nReceived interrupt, terminating workers...")
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if executor:
            try:
                executor.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass

    # Sort by original task order
    task_id_to_index = {task.task_id: i for i, task in enumerate(tasks)}
    results = [results_dict[t.task_id] for t in tasks if t.task_id in results_dict]
    results.sort(key=lambda r: task_id_to_index.get(r.task.task_id, len(tasks)))

    return results
