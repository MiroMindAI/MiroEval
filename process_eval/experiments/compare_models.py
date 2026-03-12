#!/usr/bin/env python3
"""
A/B experiment: compare gpt-5-mini vs gpt-5.2 on structuring + evaluation.

Runs 5 entries × 3 source models (chatgpt, glm, mirothinker) through
both LLM models and compares the outputs.
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process_evaluator.preprocessors import get_preprocessor
from process_evaluator.structuring.prompts import STRUCTURING_PROMPT
from process_evaluator.evaluation.prompts import INTRINSIC_EVAL_PROMPT, ALIGNMENT_EVAL_PROMPT
from process_evaluator.utils.llm_client import LLMClient, extract_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Experiment config
ENTRY_IDS = [1, 3, 5, 7, 10]
SOURCE_MODELS = ["chatgpt", "glm", "mirothinker"]  # plain_text, json_array, block_text
LLM_MODELS = ["openai/gpt-5-mini", "openai/gpt-5.2"]
DATA_DIR = "../data/method_results/mirobench-text-refined-v2/mirobench-text-refined"
MAX_CHARS_OVERRIDE = {"chatgpt": 50000}
DEFAULT_MAX_CHARS = 30000

INTRINSIC_DIMS = ["search_breadth", "analytical_depth", "progressive_refinement", "critical_thinking", "efficiency"]
ALIGNMENT_DIMS = ["findings_to_report", "report_to_process", "contradiction"]


def load_entries():
    """Load target entries from data files."""
    entries = {}
    for model in SOURCE_MODELS:
        fn = os.path.join(DATA_DIR, f"{model}_text_100.json")
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
        for e in data:
            if e.get("id") in ENTRY_IDS:
                entries[(model, e["id"])] = e
    return entries


def run_single(llm: LLMClient, model_name: str, entry: dict, max_chars: int) -> dict:
    """Run full pipeline for one entry: preprocess -> structure -> intrinsic eval -> alignment eval."""
    entry_id = entry.get("id", "?")
    query = entry.get("query", entry.get("rewritten_query", ""))
    process_text = entry.get("process", "")
    report = entry.get("response", "")

    result = {"model": model_name, "entry_id": entry_id, "llm": llm.model}

    if not process_text.strip():
        result["error"] = "empty process"
        return result

    # Phase 1: Preprocess
    pp = get_preprocessor(max_chars=max_chars)
    preprocessed = pp.preprocess(process_text)
    result["preprocessed_len"] = len(preprocessed)

    # Phase 1: Structure
    prompt = STRUCTURING_PROMPT.format(query=query, process_text=preprocessed)
    messages = [{"role": "user", "content": prompt}]
    structured = llm.generate_json(messages)
    if structured is None:
        result["error"] = "structuring failed"
        return result

    result["structured_process"] = structured
    result["num_steps"] = len(structured.get("steps", []))
    result["num_global_findings"] = len(structured.get("global_findings", []))

    # Phase 2: Intrinsic eval
    intrinsic_prompt = INTRINSIC_EVAL_PROMPT.format(
        query=query,
        structured_process=json.dumps(structured, ensure_ascii=False, indent=2),
    )
    intrinsic = llm.generate_json([{"role": "user", "content": intrinsic_prompt}])
    if intrinsic:
        result["intrinsic_scores"] = {
            dim: intrinsic[dim]["score"]
            for dim in INTRINSIC_DIMS
            if dim in intrinsic and isinstance(intrinsic[dim], dict) and "score" in intrinsic[dim]
        }

    # Phase 2: Alignment eval
    global_findings = structured.get("global_findings", [])
    findings_text = json.dumps(global_findings, ensure_ascii=False, indent=2)
    truncated_report = report[:30000]
    if len(report) > 30000:
        truncated_report += "\n\n[... report truncated ...]"

    alignment_prompt = ALIGNMENT_EVAL_PROMPT.format(
        query=query,
        global_findings=findings_text,
        report=truncated_report,
    )
    alignment = llm.generate_json([{"role": "user", "content": alignment_prompt}])
    if alignment:
        result["alignment_scores"] = {
            dim: alignment[dim]["score"]
            for dim in ALIGNMENT_DIMS
            if dim in alignment and isinstance(alignment[dim], dict) and "score" in alignment[dim]
        }

    return result


def main():
    load_dotenv()

    logger.info("Loading entries...")
    entries = load_entries()
    logger.info(f"Loaded {len(entries)} entries")

    # Build tasks: (llm_model, source_model, entry)
    tasks = []
    for llm_model in LLM_MODELS:
        for (source_model, entry_id), entry in entries.items():
            tasks.append((llm_model, source_model, entry_id, entry))

    logger.info(f"Total tasks: {len(tasks)} ({len(LLM_MODELS)} LLMs × {len(entries)} entries)")

    # Create LLM clients
    clients = {}
    for llm_model in LLM_MODELS:
        clients[llm_model] = LLMClient(
            model=llm_model,
            api_type="openrouter",
            max_tokens=8192,
            temperature=0.1,
            retry_count=3,
        )

    # Run all tasks with parallelism (but limited to avoid rate limits)
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_info = {}
        for llm_model, source_model, entry_id, entry in tasks:
            max_chars = MAX_CHARS_OVERRIDE.get(source_model, DEFAULT_MAX_CHARS)
            future = executor.submit(
                run_single, clients[llm_model], source_model, entry, max_chars
            )
            future_to_info[future] = (llm_model, source_model, entry_id)

        for future in as_completed(future_to_info):
            llm_model, source_model, entry_id = future_to_info[future]
            try:
                result = future.result()
                results.append(result)
                scores = result.get("intrinsic_scores", {})
                score_str = ", ".join(f"{k}={v}" for k, v in scores.items())
                logger.info(f"Done: {llm_model} | {source_model}_{entry_id} | {score_str}")
            except Exception as e:
                logger.error(f"Failed: {llm_model} | {source_model}_{entry_id} | {e}")

    # Save raw results
    os.makedirs("experiments/outputs", exist_ok=True)
    out_file = f"experiments/outputs/compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Raw results saved to {out_file}")

    # Print comparison table
    print_comparison(results)

    # Print cost
    for llm_model in LLM_MODELS:
        print(f"  {llm_model} cost: ${clients[llm_model].total_cost:.4f}")


def print_comparison(results: list):
    """Print side-by-side comparison table."""
    # Group by (source_model, entry_id)
    grouped = {}
    for r in results:
        key = (r["model"], r["entry_id"])
        llm = r.get("llm", "?")
        grouped.setdefault(key, {})[llm] = r

    all_dims = INTRINSIC_DIMS + ALIGNMENT_DIMS

    print("\n" + "=" * 120)
    print("COMPARISON: gpt-5-mini vs gpt-5.2")
    print("=" * 120)

    # Header
    header = f"{'Source':<16}{'Entry':>6}  {'Dimension':<24}"
    for llm in LLM_MODELS:
        short_name = llm.split("/")[1]
        header += f"  {short_name:>10}"
    header += f"  {'Diff':>8}"
    print(header)
    print("-" * 120)

    total_diffs = {dim: [] for dim in all_dims}

    for (source_model, entry_id) in sorted(grouped.keys()):
        llm_results = grouped[(source_model, entry_id)]

        # Print steps/findings count
        for llm_model in LLM_MODELS:
            r = llm_results.get(llm_model, {})
            short = llm_model.split("/")[1]
            steps = r.get("num_steps", "?")
            findings = r.get("num_global_findings", "?")
            print(f"  {source_model:<14}{entry_id:>6}  {short}: {steps} steps, {findings} global findings")

        # Print scores
        for dim in all_dims:
            scores = []
            for llm_model in LLM_MODELS:
                r = llm_results.get(llm_model, {})
                intrinsic = r.get("intrinsic_scores", {})
                alignment = r.get("alignment_scores", {})
                s = intrinsic.get(dim, alignment.get(dim, None))
                scores.append(s)

            row = f"  {source_model:<14}{entry_id:>6}  {dim:<24}"
            for s in scores:
                row += f"  {s:>10}" if s is not None else f"  {'N/A':>10}"

            if all(s is not None for s in scores):
                diff = scores[1] - scores[0]  # 5.2 - mini
                row += f"  {diff:>+8.1f}"
                total_diffs[dim].append(diff)
            else:
                row += f"  {'N/A':>8}"

            print(row)
        print()

    # Summary
    print("=" * 120)
    print("AVERAGE DIFFERENCES (gpt-5.2 - gpt-5-mini)")
    print("-" * 60)
    for dim in all_dims:
        diffs = total_diffs[dim]
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            print(f"  {dim:<28} avg diff: {avg_diff:>+6.2f}  (n={len(diffs)})")
    print("=" * 120)


if __name__ == "__main__":
    main()
