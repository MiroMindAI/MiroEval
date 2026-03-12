#!/usr/bin/env python3
"""
LLM-based preprocessing experiment (full-scale):
  - Preprocessing: gpt-5-mini (16K max_tokens)
  - Structuring + Evaluation: gpt-5.2
  - All entries × 11 source models
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

from process_evaluator.preprocessors.llm_preprocessor import LLMPreprocessor
from process_evaluator.structuring.prompts import STRUCTURING_PROMPT
from process_evaluator.evaluation.prompts import INTRINSIC_EVAL_PROMPT, ALIGNMENT_EVAL_PROMPT
from process_evaluator.utils.llm_client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ENTRY_IDS = None  # None = all entries
SOURCE_MODELS = ["chatgpt", "claude", "doubao", "gemini", "glm", "kimi", "minimax", "mirothinker", "mirothinker_base_17", "mirothinker_v17", "qwen"]
DATA_DIR = "../data/method_results"

PREPROCESS_MODEL = "gpt-5-mini"
EVAL_MODEL = "gpt-5.2"
MAX_CHARS = 30000

INTRINSIC_DIMS = ["search_breadth", "analytical_depth", "progressive_refinement", "critical_thinking", "efficiency"]
ALIGNMENT_DIMS = ["findings_to_report", "report_to_process", "contradiction"]


def load_entries():
    entries = {}
    for model in SOURCE_MODELS:
        fn = os.path.join(DATA_DIR, f"{model}_text_100.json")
        if not os.path.exists(fn):
            # try with different suffix
            import glob
            candidates = glob.glob(os.path.join(DATA_DIR, f"{model}_text_*.json"))
            if candidates:
                fn = candidates[0]
            else:
                logger.warning(f"No data file for {model}")
                continue
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
        for e in data:
            if ENTRY_IDS is None or e.get("id") in ENTRY_IDS:
                entries[(model, e["id"])] = e
    return entries


def run_one(preprocess_llm, eval_llm, model_name, entry):
    entry_id = entry.get("id", "?")
    query = entry.get("query", entry.get("rewritten_query", ""))
    process_text = entry.get("process", "")
    report = entry.get("response", "")

    result = {"model": model_name, "entry_id": entry_id}

    if not process_text.strip():
        result["error"] = "empty process"
        return result

    # LLM Preprocess
    pp = LLMPreprocessor(llm_client=preprocess_llm, max_chars=MAX_CHARS, query=query)
    preprocessed = pp.preprocess(process_text)
    result["raw_len"] = len(process_text)
    result["preprocessed_len"] = len(preprocessed)

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
        result["intrinsic_scores"] = {
            dim: intrinsic[dim]["score"]
            for dim in INTRINSIC_DIMS
            if dim in intrinsic and isinstance(intrinsic[dim], dict) and "score" in intrinsic[dim]
        }

    # Alignment eval
    global_findings = structured.get("global_findings", [])
    findings_text = json.dumps(global_findings, ensure_ascii=False, indent=2)
    truncated_report = report[:30000]
    if len(report) > 30000:
        truncated_report += "\n\n[... report truncated ...]"

    alignment_prompt = ALIGNMENT_EVAL_PROMPT.format(
        query=query, global_findings=findings_text, report=truncated_report,
    )
    alignment = eval_llm.generate_json([{"role": "user", "content": alignment_prompt}])
    if alignment:
        result["alignment_scores"] = {
            dim: alignment[dim]["score"]
            for dim in ALIGNMENT_DIMS
            if dim in alignment and isinstance(alignment[dim], dict) and "score" in alignment[dim]
        }

    return result


def main():
    load_dotenv()

    entries = load_entries()
    logger.info(f"Loaded {len(entries)} entries across {len(SOURCE_MODELS)} models")

    preprocess_llm = LLMClient(model=PREPROCESS_MODEL, api_type="openai", max_tokens=16384, temperature=0.1, retry_count=3)
    eval_llm = LLMClient(model=EVAL_MODEL, api_type="openai", max_tokens=16384, temperature=0.1, retry_count=3)

    tasks = list(entries.items())
    logger.info(f"Total tasks: {len(tasks)}")

    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {
            executor.submit(run_one, preprocess_llm, eval_llm, model, entry): (model, entry.get("id"))
            for (model, eid), entry in tasks
        }
        for future in as_completed(futures):
            model, eid = futures[future]
            try:
                r = future.result()
                results.append(r)
                scores = r.get("intrinsic_scores", {})
                s_str = ", ".join(f"{k}={v}" for k, v in scores.items())
                logger.info(f"Done: {model}_{eid} | {s_str}")
            except Exception as e:
                logger.error(f"Failed: {model}_{eid} | {e}")

    # Save
    os.makedirs("experiments/outputs", exist_ok=True)
    out_file = f"experiments/outputs/llm_preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {out_file}")

    # Print results
    print_results(results)
    print(f"\nCosts: preprocess(gpt-5-mini)=${preprocess_llm.total_cost:.4f}, eval(gpt-5.2)=${eval_llm.total_cost:.4f}")


def print_results(results):
    all_dims = INTRINSIC_DIMS + ALIGNMENT_DIMS
    dim_short = {
        'search_breadth': 'breadth', 'analytical_depth': 'depth',
        'progressive_refinement': 'refine', 'critical_thinking': 'critical',
        'efficiency': 'effic', 'findings_to_report': 'f->r',
        'report_to_process': 'r->p', 'contradiction': 'contra',
    }

    from collections import defaultdict
    model_scores = defaultdict(lambda: {d: [] for d in all_dims})

    for r in results:
        model = r["model"]
        for dim in INTRINSIC_DIMS:
            s = r.get("intrinsic_scores", {}).get(dim)
            if s is not None:
                model_scores[model][dim].append(s)
        for dim in ALIGNMENT_DIMS:
            s = r.get("alignment_scores", {}).get(dim)
            if s is not None:
                model_scores[model][dim].append(s)

    header = f"\n{'Model':<22}"
    for d in all_dims:
        header += f" {dim_short[d]:>8}"
    header += f" {'Intr':>6} {'Align':>6} {'Overall':>8} {'n':>4}"
    print(header)
    print("=" * len(header))

    model_avgs = {}
    for model in sorted(model_scores.keys()):
        scores = model_scores[model]
        row = f"{model:<22}"
        i_avgs, a_avgs = [], []
        n = 0
        for d in all_dims:
            vals = scores[d]
            n = max(n, len(vals))
            if vals:
                avg = sum(vals) / len(vals)
                row += f" {avg:>8.2f}"
                (i_avgs if d in INTRINSIC_DIMS else a_avgs).append(avg)
            else:
                row += f" {'N/A':>8}"
        i_avg = sum(i_avgs) / len(i_avgs) if i_avgs else 0
        a_avg = sum(a_avgs) / len(a_avgs) if a_avgs else 0
        overall = (i_avg + a_avg) / 2
        row += f" {i_avg:>6.2f} {a_avg:>6.2f} {overall:>8.2f} {n:>4}"
        model_avgs[model] = overall
        print(row)

    print("=" * len(header))


if __name__ == "__main__":
    main()
