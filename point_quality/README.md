# Point Quality Evaluation

A pointwise evaluation framework for assessing deep research system outputs. It uses an LLM-as-judge approach with a multi-stage pipeline to generate query-specific evaluation criteria and score model responses.

## Pipeline Overview

The evaluation runs in 5 stages:

| Stage | Name | Description |
|-------|------|-------------|
| 0 | Key Facts Extraction | Extract key facts from query attachments (images, PDFs) |
| 1 | Dimension Generation | Generate query-specific evaluation dimensions (up to 3) in addition to 4 fixed dimensions |
| 2 | Weight Assignment | Assign weights to each dimension (fixed dimensions get >= 80%) |
| 3 | Criteria Generation | Generate fine-grained scoring criteria for each dimension |
| 4 | Scoring | Score the model response against all criteria |

**Fixed dimensions**: Coverage, Insight, Instruction Following, Clarity

**Query-specific dimensions**: Dynamically generated based on query content (e.g., attachment-grounded analysis, domain-specific expertise)

## Project Structure

```
point_quality/
├── run_batch_eval.py                    # Main evaluation script
└── deepresearcharena/
    ├── evaluator/
    │   ├── base_evaluator.py            # Base class with LLM client, attachment reading, caching
    │   ├── pointwise_evaluator.py       # Orchestrator: query selection, parallelization, result aggregation
    │   └── pointwise_core.py            # Core logic: dimension/weight/criteria generation and scoring
    ├── prompts/
    │   └── pointwise_prompts.py         # All prompt templates for each pipeline stage
    ├── cache/
    │   ├── cache_manager.py             # Cache orchestration across pipeline stages
    │   └── file_cache.py                # JSON file-based cache implementation
    ├── config/
    │   └── pointwise.yaml               # Default configuration
    └── utils/
        ├── llm_call.py                  # OpenAI-compatible API client with cost tracking
        └── config.py                    # YAML config loader
```

## Installation

```bash
pip install openai python-dotenv pyyaml PyPDF2
```

Create a `.env` file in this directory:

```
OPENROUTER_API_KEY=your_api_key_here
```

## Usage

### Full evaluation (generate criteria + score)

```bash
python run_batch_eval.py \
  --input ../results/model_results.json \
  --model_name model_name \
  --output outputs/model_eval.json \
  --max_workers 20
```

### Reuse criteria from a previous run (only re-score)

This skips Stages 0-3 and only runs Stage 4, ensuring all models are scored against the same criteria:

```bash
python run_batch_eval.py \
  --input ../results/another_model.json \
  --model_name another_model \
  --criteria_file outputs/base_model_eval.json \
  --output outputs/another_model_eval.json \
  --cache_dir outputs/cache_another_model \
  --max_workers 20
```

### Input format

The input JSON should be a list of entries:

```json
[
  {
    "id": 101,
    "rewritten_query": "the query text...",
    "response": "the model's response...",
    "files": [
      {"dir": "multimodal-attachments/101/image.jpg"}
    ]
  }
]
```

- `rewritten_query` or `query`: the task prompt
- `response`: the model's output to evaluate
- `files` (optional): list of attachment file references

### Output format

The output JSON contains per-query results with:

- `all_criteria`: generated evaluation criteria per dimension
- `dimension_weights`: weight assigned to each dimension
- `model_results`: per-model scores including raw criterion-level scores, dimension scores, and total weighted score

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | Path to input JSON file |
| `--model_name` | (required) | Name for the target model |
| `--evaluator_model` | `openai/gpt-5` | LLM model used as judge |
| `--api_type` | `openrouter` | API type: `openai` or `openrouter` |
| `--max_queries` | all | Maximum number of queries to evaluate |
| `--max_workers` | 20 | Parallel workers for query-level evaluation |
| `--cache_dir` | `outputs/cache` | Cache directory (use separate dirs for concurrent runs) |
| `--criteria_file` | None | Previous results JSON to reuse criteria |
| `--output` | `outputs/<model>_results.json` | Output file path |
| `--data_dir` | `../data` | Data directory for resolving attachment paths |
