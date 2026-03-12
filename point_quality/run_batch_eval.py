#!/usr/bin/env python3
"""
Batch evaluation script for standalone JSON files (e.g., chatgpt_text_100.json).

These files contain both query and response in each entry, so we adapt them
to the PointwiseEvaluator's expected format (self.queries + self.model_results).

Usage:
    # Full evaluation (generate criteria + score)
    python run_batch_eval.py --input ../data/input_queries/text/chatgpt_text_100.json --model_name chatgpt

    # Reuse criteria from a previous run (only re-score)
    python run_batch_eval.py --input ../results/new_model.json --model_name new_model --criteria_file outputs/chatgpt_results.json
"""
import json
import os
import sys
import logging
import argparse
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from deepresearcharena.evaluator.pointwise_evaluator import PointwiseEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("outputs/batch_eval.log")
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Batch pointwise evaluation for standalone JSON files')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file (e.g., chatgpt_text_100.json)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name for the target model (e.g., chatgpt)')
    parser.add_argument('--evaluator_model', type=str, default='openai/gpt-5',
                        help='Evaluator LLM model name')
    parser.add_argument('--api_type', type=str, default='openrouter',
                        help='API type: openai or openrouter')
    parser.add_argument('--max_queries', type=int, default=None,
                        help='Maximum number of queries to evaluate (default: all)')
    parser.add_argument('--max_workers', type=int, default=20,
                        help='Maximum parallel workers for query-level parallelization')
    parser.add_argument('--cache_dir', type=str, default='outputs/cache',
                        help='Cache directory')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Data directory (for resolving attachment paths)')
    parser.add_argument('--criteria_file', type=str, default=None,
                        help='Path to a previous results JSON to reuse criteria (skip Stages 0-3, only run Stage 4 scoring)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: outputs/<model_name>_results.json)')
    args = parser.parse_args()

    # Load input data
    print(f"Loading data from: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries")

    # Create evaluator (data_dir is not used for loading, but required by base class)
    os.makedirs("outputs", exist_ok=True)
    evaluator = PointwiseEvaluator(
        data_dir="../data",
        model_name=args.evaluator_model,
        api_type=args.api_type,
        cache_dir=args.cache_dir
    )

    # Populate evaluator's queries and model_results from the JSON file
    skipped = 0
    for entry in data:
        qid = entry['id']
        task_prompt = entry.get('rewritten_query') or entry.get('query', '')
        report = entry.get('response')

        # Skip entries without response
        if not report:
            skipped += 1
            continue

        # Build query dict with 'prompt' key (expected by pointwise_core.py)
        query_dict = {
            'id': qid,
            'prompt': task_prompt,
        }

        # Handle attachments from 'files' field
        files = entry.get('files', [])
        if files:
            attachment_parts = []
            data_dir = os.path.abspath(args.data_dir)
            for file_info in files:
                # file_info has 'dir' field with relative path like 'multimodal-attachments/101/xxx.jpg'
                file_path = file_info.get('dir', '')
                if file_path:
                    # Resolve relative to data_dir/input_queries/multimodal/
                    full_path = os.path.join(data_dir, "input_queries", "multimodal", file_path)
                    if os.path.isfile(full_path):
                        content = evaluator._read_attachment_file(full_path)
                        attachment_parts.append(content)
                    else:
                        logger.warning(f"Attachment file not found: {full_path}")

            if attachment_parts:
                # Store per-file contents as list (for per-file key facts extraction)
                query_dict['attachment_parts'] = attachment_parts
                # Also store combined text for _has_attachment() compatibility
                query_dict['attachment'] = "\n\n---\n\n".join(attachment_parts)
                print(f"  Query {qid}: resolved {len(attachment_parts)} attachment(s)")

        evaluator.queries[qid] = query_dict

        # Set model result
        if args.model_name not in evaluator.model_results:
            evaluator.model_results[args.model_name] = {}
        evaluator.model_results[args.model_name][qid] = report

    if skipped:
        print(f"Skipped {skipped} entries without response")

    print(f"Prepared {len(evaluator.queries)} queries for model '{args.model_name}'")

    # Load pre-generated criteria if provided
    preloaded_criteria = {}
    if args.criteria_file:
        print(f"Loading pre-generated criteria from: {args.criteria_file}")
        with open(args.criteria_file, 'r', encoding='utf-8') as f:
            prev_results = json.load(f)
        for qid_str, qr in prev_results.get('query_results', {}).items():
            qid = int(qid_str)
            if qid in evaluator.queries:
                preloaded_criteria[qid] = {
                    'query_id': qid,
                    'all_criteria': qr['all_criteria'],
                    'all_dims_with_definition': qr['all_dims_with_definition'],
                    'dimension_weights': qr['dimension_weights'],
                    'additional_dimensions': qr['additional_dimensions'],
                    'key_facts': qr.get('key_facts'),
                    'has_attachment': qr.get('has_attachment', False),
                }
        print(f"  Loaded criteria for {len(preloaded_criteria)} queries (skipping Stages 0-3)")

    # Build query selection config
    query_selection_config = {
        'selection_method': 'first',
    }
    if args.max_queries is not None:
        query_selection_config['max_queries'] = args.max_queries

    # Run evaluation
    print("=" * 80)
    print(f"Starting pointwise evaluation")
    print(f"  Evaluator model: {args.evaluator_model}")
    print(f"  Target model: {args.model_name}")
    print(f"  Queries: {args.max_queries or len(evaluator.queries)}")
    print(f"  Max workers: {args.max_workers}")
    if preloaded_criteria:
        print(f"  Criteria: reusing from {args.criteria_file}")
    print("=" * 80)

    if preloaded_criteria:
        # Skip Stages 0-3, only run Stage 4 (scoring) using preloaded criteria
        results = evaluator.evaluate_all_queries_with_criteria(
            model_names=[args.model_name],
            preloaded_criteria=preloaded_criteria,
            query_selection_config=query_selection_config,
            max_workers=args.max_workers
        )
    else:
        results = evaluator.evaluate_all_queries(
            model_names=[args.model_name],
            query_selection_config=query_selection_config,
            max_workers=args.max_workers
        )

    # Print results
    evaluator.print_results(results)

    # Save results
    output_file = args.output or f"outputs/{args.model_name}_results.json"
    evaluator.save_results(results, output_file)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    summary = results.get('summary', {})
    model_summary = summary.get('models', {}).get(args.model_name, {})
    if model_summary:
        print(f"\n{'=' * 60}")
        print(f"  Model: {args.model_name}")
        print(f"  Average S_quality: {model_summary.get('average_total_score', 0):.3f}")
        print(f"  Queries evaluated: {model_summary.get('total_queries', 0)}")
        dim_avgs = model_summary.get('dimension_averages', {})
        if dim_avgs:
            print(f"  Dimension averages:")
            for dim, score in sorted(dim_avgs.items()):
                print(f"    {dim}: {score:.3f}")
        print(f"{'=' * 60}")

    print(f"\nTotal API cost: ${evaluator.client._total_cost:.4f}")


if __name__ == "__main__":
    main()
