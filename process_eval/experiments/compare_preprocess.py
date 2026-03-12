#!/usr/bin/env python3
"""
Compare 3 preprocessing strategies, all evaluated with gpt-5.2:
  A) Rule-based preprocessing (current AutoDetectPreprocessor)
  B) LLM preprocessing with gpt-5-nano
  C) LLM preprocessing with gpt-5-mini

Tests 5 entries × 3 source models (chatgpt, glm, mirothinker).
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from process_evaluator.preprocessors import get_preprocessor
from process_evaluator.preprocessors.llm_preprocessor import LLMPreprocessor
from process_evaluator.structuring.prompts import STRUCTURING_PROMPT
from process_evaluator.evaluation.prompts import INTRINSIC_EVAL_PROMPT, ALIGNMENT_EVAL_PROMPT
from process_evaluator.utils.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ENTRY_IDS = [1, 3, 5, 7, 10]
SOURCE_MODELS = ["chatgpt", "glm", "mirothinker"]
DATA_DIR = "../data/method_results"
MAX_CHARS_OVERRIDE = {"chatgpt": 50000}
DEFAULT_MAX_CHARS = 30000

EVAL_MODEL = "openai/gpt-5.2"

PREPROCESS_STRATEGIES = [
    {"name": "rule", "type": "rule"},
    {"name": "llm_nano", "type": "llm", "model": "openai/gpt-5-nano"},
    {"name": "llm_mini", "type": "llm", "model": "openai/gpt-5-mini"},
]

INTRINSIC_DIMS = ["search_breadth", "analytical_depth", "progressive_refinement", "critical_thinking", "efficiency"]
ALIGNMENT_DIMS = ["findings_to_report", "report_to_process", "contradiction"]


def load_entries():
    entries = {}
    for model in SOURCE_MODELS:
        fn = os.path.join(DATA_DIR, f"{model}_text_100.json")
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
        for e in data:
            if e.get("id") in ENTRY_IDS:
                entries[(model, e["id"])] = e
    return entries


def preprocess_entry(strategy: dict, entry: dict, model_name: str, llm_clients: dict) -> str:
    """Preprocess a single entry using the given strategy."""
    process_text = entry.get("process", "")
    query = entry.get("query", entry.get("rewritten_query", ""))
    max_chars = MAX_CHARS_OVERRIDE.get(model_name, DEFAULT_MAX_CHARS)

    if strategy["type"] == "rule":
        pp = get_preprocessor(max_chars=max_chars)
        return pp.preprocess(process_text)
    else:
        llm = llm_clients[strategy["model"]]
        pp = LLMPreprocessor(llm_client=llm, max_chars=max_chars, query=query)
        return pp.preprocess(process_text)


def evaluate_entry(eval_llm: LLMClient, preprocessed: str, entry: dict) -> dict:
    """Structure + evaluate a preprocessed entry using the eval model."""
    query = entry.get("query", entry.get("rewritten_query", ""))
    report = entry.get("response", "")
    result = {}

    # Structure
    prompt = STRUCTURING_PROMPT.format(query=query, process_text=preprocessed)
    structured = eval_llm.generate_json([{"role": "user", "content": prompt}])
    if structured is None:
        result["error"] = "structuring failed"
        return result

    result["num_steps"] = len(structured.get("steps", []))
    result["num_global_findings"] = len(structured.get("global_findings", []))

    # Intrinsic eval
    intrinsic_prompt = INTRINSIC_EVAL_PROMPT.format(
        query=query,
        structured_process=json.dumps(structured, ensure_ascii=False, indent=2),
    )
    intrinsic = eval_llm.generate_json([{"role": "user", "content": intrinsic_prompt}])
    if intrinsic:
        result["intrinsic_scores"] = {}
        for dim in INTRINSIC_DIMS:
            if dim in intrinsic and isinstance(intrinsic[dim], dict) and "score" in intrinsic[dim]:
                result["intrinsic_scores"][dim] = intrinsic[dim]["score"]

    # Alignment eval
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
    alignment = eval_llm.generate_json([{"role": "user", "content": alignment_prompt}])
    if alignment:
        result["alignment_scores"] = {}
        for dim in ALIGNMENT_DIMS:
            if dim in alignment and isinstance(alignment[dim], dict) and "score" in alignment[dim]:
                result["alignment_scores"][dim] = alignment[dim]["score"]

    return result


def run_one_task(strategy, model_name, entry_id, entry, llm_clients, eval_llm):
    """Run preprocessing + evaluation for one (strategy, entry) pair."""
    logger.info(f"Starting: {strategy['name']} | {model_name}_{entry_id}")

    # Preprocess
    preprocessed = preprocess_entry(strategy, entry, model_name, llm_clients)
    preprocessed_len = len(preprocessed)

    # Evaluate
    eval_result = evaluate_entry(eval_llm, preprocessed, entry)
    eval_result["strategy"] = strategy["name"]
    eval_result["model"] = model_name
    eval_result["entry_id"] = entry_id
    eval_result["preprocessed_len"] = preprocessed_len
    eval_result["raw_len"] = len(entry.get("process", ""))

    logger.info(f"Done: {strategy['name']} | {model_name}_{entry_id} | "
                f"raw={eval_result['raw_len']:,} -> preprocessed={preprocessed_len:,} | "
                f"steps={eval_result.get('num_steps', '?')}")
    return eval_result


def main():
    load_dotenv()

    entries = load_entries()
    logger.info(f"Loaded {len(entries)} entries")

    # Create LLM clients
    llm_clients = {}
    for s in PREPROCESS_STRATEGIES:
        if s["type"] == "llm" and s["model"] not in llm_clients:
            llm_clients[s["model"]] = LLMClient(
                model=s["model"], api_type="openrouter",
                max_tokens=4096, temperature=0.1, retry_count=3,
            )

    eval_llm = LLMClient(
        model=EVAL_MODEL, api_type="openrouter",
        max_tokens=8192, temperature=0.1, retry_count=3,
    )

    # Build tasks
    tasks = []
    for strategy in PREPROCESS_STRATEGIES:
        for (model_name, entry_id), entry in entries.items():
            tasks.append((strategy, model_name, entry_id, entry))

    logger.info(f"Total tasks: {len(tasks)} "
                f"({len(PREPROCESS_STRATEGIES)} strategies × {len(entries)} entries)")

    # Run tasks — serialize by strategy to avoid rate limit issues
    # (each strategy's preprocessing uses different LLM or no LLM)
    results = []
    for strategy in PREPROCESS_STRATEGIES:
        strategy_tasks = [(s, m, eid, e) for s, m, eid, e in tasks if s["name"] == strategy["name"]]
        logger.info(f"\n=== Running strategy: {strategy['name']} ({len(strategy_tasks)} tasks) ===")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    run_one_task, s, m, eid, e, llm_clients, eval_llm
                ): (s["name"], m, eid)
                for s, m, eid, e in strategy_tasks
            }
            for future in as_completed(futures):
                info = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed: {info} | {e}")

    # Save raw results
    os.makedirs("experiments/outputs", exist_ok=True)
    out_file = f"experiments/outputs/preprocess_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Raw results saved to {out_file}")

    # Print comparison
    print_comparison(results)

    # Print costs
    print("\nCosts:")
    for model, client in llm_clients.items():
        print(f"  {model} (preprocessing): ${client.total_cost:.4f}")
    print(f"  {EVAL_MODEL} (eval): ${eval_llm.total_cost:.4f}")


def print_comparison(results: list):
    """Print comparison across preprocessing strategies."""
    all_dims = INTRINSIC_DIMS + ALIGNMENT_DIMS

    # Group by (model, entry_id, strategy)
    grouped = {}
    for r in results:
        key = (r["model"], r["entry_id"])
        grouped.setdefault(key, {})[r["strategy"]] = r

    strategies = [s["name"] for s in PREPROCESS_STRATEGIES]

    print(f"\n{'=' * 130}")
    print(f"PREPROCESSING COMPARISON (all evaluated with {EVAL_MODEL})")
    print(f"{'=' * 130}")

    # Compression ratio table
    print(f"\n--- Compression Ratios ---")
    print(f"  {'Source':<14}{'Entry':>6}  {'Raw':>10}", end="")
    for s in strategies:
        print(f"  {s:>12}", end="")
    print()
    print(f"  {'-' * 80}")

    for (model, entry_id) in sorted(grouped.keys()):
        strats = grouped[(model, entry_id)]
        raw_len = strats.get(strategies[0], {}).get("raw_len", 0)
        row = f"  {model:<14}{entry_id:>6}  {raw_len:>10,}"
        for s in strategies:
            pp_len = strats.get(s, {}).get("preprocessed_len", 0)
            ratio = pp_len / raw_len if raw_len > 0 else 0
            row += f"  {pp_len:>7,}({ratio:.0%})"
        print(row)

    # Score comparison table
    print(f"\n--- Scores by Strategy ---")
    header = f"  {'Source':<14}{'Entry':>6}  {'Dimension':<24}"
    for s in strategies:
        header += f"  {s:>10}"
    print(header)
    print(f"  {'-' * 100}")

    strategy_dim_scores = {s: {d: [] for d in all_dims} for s in strategies}

    for (model, entry_id) in sorted(grouped.keys()):
        strats = grouped[(model, entry_id)]

        # Steps/findings
        for s in strategies:
            r = strats.get(s, {})
            steps = r.get("num_steps", "?")
            findings = r.get("num_global_findings", "?")
            print(f"  {model:<14}{entry_id:>6}  {s}: {steps} steps, {findings} findings")

        for dim in all_dims:
            row = f"  {model:<14}{entry_id:>6}  {dim:<24}"
            for s in strategies:
                r = strats.get(s, {})
                intrinsic = r.get("intrinsic_scores", {})
                alignment = r.get("alignment_scores", {})
                score = intrinsic.get(dim, alignment.get(dim))
                if score is not None:
                    row += f"  {score:>10}"
                    strategy_dim_scores[s][dim].append(score)
                else:
                    row += f"  {'N/A':>10}"
            print(row)
        print()

    # Summary averages
    print(f"{'=' * 130}")
    print("AVERAGE SCORES BY STRATEGY")
    print(f"{'-' * 80}")
    header = f"  {'Dimension':<28}"
    for s in strategies:
        header += f"  {s:>12}"
    print(header)
    print(f"  {'-' * 70}")

    for dim in all_dims:
        row = f"  {dim:<28}"
        for s in strategies:
            scores = strategy_dim_scores[s][dim]
            if scores:
                avg = sum(scores) / len(scores)
                row += f"  {avg:>10.2f}({len(scores)})"
            else:
                row += f"  {'N/A':>12}"
        print(row)

    # Overall averages
    print(f"  {'-' * 70}")
    row = f"  {'OVERALL':<28}"
    for s in strategies:
        all_scores = []
        for dim in all_dims:
            scores = strategy_dim_scores[s][dim]
            if scores:
                all_scores.append(sum(scores) / len(scores))
        if all_scores:
            overall = sum(all_scores) / len(all_scores)
            row += f"  {overall:>12.2f}"
        else:
            row += f"  {'N/A':>12}"
    print(row)
    print(f"{'=' * 130}")

    # Per source model summary
    print(f"\nPER SOURCE MODEL AVERAGES")
    print(f"{'-' * 80}")
    for src_model in SOURCE_MODELS:
        print(f"\n  {src_model}:")
        for s in strategies:
            i_scores = []
            a_scores = []
            for (model, entry_id) in sorted(grouped.keys()):
                if model != src_model:
                    continue
                r = grouped[(model, entry_id)].get(s, {})
                intrinsic = r.get("intrinsic_scores", {})
                alignment = r.get("alignment_scores", {})
                i_vals = [intrinsic[d] for d in INTRINSIC_DIMS if d in intrinsic]
                a_vals = [alignment[d] for d in ALIGNMENT_DIMS if d in alignment]
                if i_vals:
                    i_scores.append(sum(i_vals) / len(i_vals))
                if a_vals:
                    a_scores.append(sum(a_vals) / len(a_vals))
            i_avg = f"{sum(i_scores)/len(i_scores):.2f}" if i_scores else "N/A"
            a_avg = f"{sum(a_scores)/len(a_scores):.2f}" if a_scores else "N/A"
            print(f"    {s:<14} intrinsic={i_avg:>6}  alignment={a_avg:>6}")


if __name__ == "__main__":
    main()
