"""
Convert a method_results JSON array file into individual per-item files
for the factual_eval benchmark.

The input format (data/method_results/<model>.json) is a JSON array where
each element has the MiroEval shared schema:
  {id, chat_id, query, rewritten_query, annotation, response, process, files, ...}

The output format (miroflow/data/factual-eval/<model_dir>/<model_name>_<id>.json)
is one JSON file per item, same schema, one object per file.

Usage:
    python utils/convert_to_factual_eval.py \\
        --input ../data/method_results/mirothinker_v17_text_100.json \\
        --output-dir ../../miroflow/data/factual-eval/mirothinker-v17-text-only-50 \\
        [--model-name mirothinker_v17] \\
        [--num-samples 50] \\
        [--category text]

Run from the MiroEval/ root directory.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a method_results JSON array to individual factual-eval files"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source JSON array file (e.g. data/method_results/mirothinker_v17_text_100.json)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write individual JSON files into",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Prefix for output filenames (default: derived from --output-dir name)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of items to convert (default: all)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter by annotation.category (e.g. 'text', 'image', 'doc'). Default: no filter",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output-dir (default: skip existing)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Derive model name from output dir if not specified
    model_name = args.model_name or output_dir.name.replace("-", "_")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {input_path}, got {type(data).__name__}")

    print(f"Loaded {len(data)} items from {input_path}")

    # Optional category filter
    if args.category:
        data = [
            item for item in data
            if item.get("annotation", {}).get("category") == args.category
        ]
        print(f"After filtering category='{args.category}': {len(data)} items")

    # Apply sample limit
    if args.num_samples is not None and len(data) > args.num_samples:
        data = data[: args.num_samples]
        print(f"Limited to {args.num_samples} items")

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for item in data:
        item_id = item.get("id", written + 1)
        output_path = output_dir / f"{model_name}_{item_id}.json"

        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)
        written += 1

    print(f"Done: {written} written, {skipped} skipped → {output_dir}")
    if skipped:
        print("  (use --overwrite to replace existing files)")


if __name__ == "__main__":
    main()