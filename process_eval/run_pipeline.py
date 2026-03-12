#!/usr/bin/env python3
"""
Process Evaluation Pipeline for Deep Research Agents.

Usage:
    python run_pipeline.py                           # Full pipeline
    python run_pipeline.py --phase phase1            # Only structuring
    python run_pipeline.py --phase phase2            # Only evaluation
    python run_pipeline.py --models chatgpt claude   # Specific models
    python run_pipeline.py --max-entries 5           # Pilot run
    python run_pipeline.py --clear-cache
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from process_evaluator.pipeline import ProcessEvalPipeline
from process_evaluator.utils.config import load_config


def setup_logging(config: dict):
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    handlers = []
    if log_cfg.get("console", True):
        handlers.append(logging.StreamHandler(sys.stdout))

    log_file = log_cfg.get("file")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Process Evaluation Pipeline")
    parser.add_argument("--config", default="config/process_eval.yaml", help="Config YAML path")
    parser.add_argument("--phase", choices=["all", "phase1", "phase2"], default=None)
    parser.add_argument("--models", nargs="+", default=None, help="Models to evaluate")
    parser.add_argument("--max-entries", type=int, default=None, help="Max entries per model")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--entry-ids", nargs="+", type=int, default=None, help="Specific entry IDs")
    parser.add_argument("--clear-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load env
    load_dotenv()

    # Load config
    config = load_config(args.config)

    # CLI overrides
    if args.phase:
        config.setdefault("execution", {})["phases"] = args.phase
    if args.models:
        config["target_models"] = args.models
    if args.max_entries is not None:
        config.setdefault("entry_selection", {})["max_entries_per_model"] = args.max_entries
    if args.max_workers is not None:
        config.setdefault("execution", {})["max_workers"] = args.max_workers
    if args.entry_ids is not None:
        config.setdefault("entry_selection", {})["entry_ids"] = args.entry_ids

    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("=== Process Evaluation Pipeline ===")
    logger.info(f"Models: {config.get('target_models', [])}")
    logger.info(f"Phase: {config.get('execution', {}).get('phases', 'all')}")

    pipeline = ProcessEvalPipeline(config)

    if args.clear_cache:
        logger.info("Clearing all caches...")
        pipeline.structuring_cache.clear()
        pipeline.intrinsic_cache.clear()
        pipeline.alignment_cache.clear()

    phase = config.get("execution", {}).get("phases", "all")
    if phase == "phase1":
        pipeline.run_phase1()
    elif phase == "phase2":
        pipeline.run_phase2()
    else:
        pipeline.run_full()


if __name__ == "__main__":
    main()
