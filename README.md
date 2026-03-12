# MiroEval

MiroEval is a comprehensive evaluation framework for Deep Research systems, providing automated assessment across three complementary dimensions: **Factual** correctness, **Point**-wise quality, and **Process** quality.

All three evaluation modules share a unified `data/` directory as their input data source. Each sub-project manages its own `.env` file for API keys (see `.env.template` in each sub-project).

## Architecture

```
MiroEval/
├── data/                      # Shared data directory (input queries + model results)
│   ├── input_queries/         # Evaluation query sets
│   ├── method_results/        # Text-only model results (JSON array per model)
│   └── method_multimodal_results/  # Multimodal model results (JSON array per model)
├── factual_eval/              # Factual evaluation (MiroFlow-based fact-checking agent)
├── point_quality/             # Quality evaluation (adaptive point-wise scoring)
├── process_eval/              # Process evaluation (intrinsic process quality + report alignment)
└── scripts/                   # Shared utility scripts (data conversion, etc.)
```

---

## Data Format

### Input Queries (`data/input_queries/`)

| File / Directory | Description | Count |
|------------------|-------------|-------|
| `mirobench_text.json` | Text-only query set | 100 |
| `mirobench_multimodal.json` | Multimodal query set (with image/document attachments) | 47 |
| `multimodal-attachments/` | Attachment files referenced by multimodal queries, organized by query ID (e.g., `102/`, `132/`). Contains images, PDFs, and other documents. | — |

**Query Schema:**

```json
{
  "id": 1,
  "chat_id": "uuid",
  "query": "Original query text",
  "rewritten_query": "Expanded/rewritten query",
  "files": [],
  "annotation": {
    "category": "text | image | doc | multi_doc",
    "language": "zh | en",
    "pattern": "T1-T6",
    "domain": "tech | finance | medical | ..."
  }
}
```

**Pattern Taxonomy (T1-T6):**
- T1: Landscape Survey
- T2: Comparative Evaluation
- T3: Fact Verification
- T4: Problem Diagnosis
- T5: Decision Analysis
- T6: Scheme Design

**Domain Distribution:** tech, finance, medical, engineering, business, humanities, science, lifestyle

### Model Results (`data/method_results/`, `data/method_multimodal_results/`)

One JSON file per model, containing a JSON array of complete query-response pairs.

**Text-only models (14):** chatgpt, claude, deepseek, doubao, gemini, glm, grok, kimi, manus, minimax, mirothinker, mirothinker_base_17, mirothinker_v17, qwen

**Multimodal models (10):** claude, gemini, glm, grok, manus, minimax, mirothinker, mirothinker_v17, openai, qwen

**Result Schema:**

```json
{
  "id": 1,
  "chat_id": "uuid",
  "query": "Query text",
  "rewritten_query": "Rewritten query",
  "files": [],
  "annotation": { "category": "...", "language": "...", "pattern": "...", "domain": "..." },
  "response": "Model-generated research report",
  "process": "String of research process"
}
```

The `response` field contains the model's final report output. The `process` field contains the model's intermediate research process trace (format varies by model).

---

## 1. Factual Eval

Active fact-checking powered by the [MiroFlow](https://github.com/MiroMindAI/MiroFlow) agent framework. Automatically extracts and verifies key factual statements in reports via search engines.

### How It Works

1. **Report Segmentation**: Splits the model-generated report into logical segments
2. **Per-segment Fact-checking**: Deploys an agent for each segment to gather evidence via web search
3. **Verdict**: Labels each factual statement as `Right` (correct) / `Wrong` (incorrect) / `Unknown` (unverifiable) / `Conflict` (contradictory evidence between web and attachment sources)

### Directory Structure

```
factual_eval/
├── config/                    # Hydra configuration files
│   ├── benchmark/             # Base benchmark config (factual-eval.yaml)
│   ├── llm/                   # LLM model configs
│   ├── tool/                  # Tool configs (search, browsing, etc.)
│   ├── prompts/               # Prompt templates
│   ├── benchmark_factual-eval_text.yaml        # Canonical config for text-only models
│   └── benchmark_factual-eval_multimodal.yaml  # Canonical config for multimodal models
├── miroflow/                  # MiroFlow core framework
│   ├── agents/                # Agent implementations (iterative + rollback)
│   ├── benchmark/             # Evaluation runners and verifiers
│   ├── llm/                   # Multi-provider LLM support
│   ├── tool/                  # MCP server tool integration
│   ├── io_processor/          # I/O processors (segmentation, summarization, etc.)
│   └── utils/                 # Utility functions
├── utils/
│   ├── convert_to_factual_eval.py  # Convert method_results JSON array → per-item files
│   └── check_progress_factual_eval.py
├── scripts/
│   └── run_factual_eval.sh    # Main run script
├── .env.template              # Environment variables template
└── pyproject.toml             # Dependencies (Python >= 3.11)
```

### Data Loading

Factual eval reads per-item JSON files from `data/factual-eval/<model-dir>/` inside `factual_eval/`.
The base data directory is configured via the `DATA_DIR` environment variable (default: `./data` relative to `factual_eval/`).

**Step 1 — Convert raw results to per-item files** (one-time, skip if already done):

```bash
cd factual_eval

# Convert a method_results JSON array → individual files in factual_eval/data/factual-eval/
python utils/convert_to_factual_eval.py \
    --input ../data/method_results/mirothinker_v17_text_100.json \
    --output-dir data/factual-eval/mirothinker-v17-text-100
```

The output format is one JSON file per item (same schema as the source), named `<model-name>_<id>.json`.

The loader also supports reading directly from a JSON array file via `--source-file` (see Usage below).

For multimodal queries, attachment files are stored in `data/input_queries/multimodal-attachments/<query_id>/` and referenced via the `files` field.

### Setup

```bash
cd factual_eval

# Install dependencies
uv sync

# Configure API keys (copy template and fill in values)
cp .env.template .env
# Edit .env with your API keys (OPENAI_API_KEY, SERPER_API_KEY, etc.)
# Optionally set DATA_DIR to the absolute path of miroflow/data/
```

### Usage

```bash
cd factual_eval

# Evaluate a specific model (from pre-converted per-item files)
bash scripts/run_factual_eval.sh --model-dir mirothinker-v17-text-100

# Multimodal evaluation
bash scripts/run_factual_eval.sh \
    --config config/benchmark_factual-eval_multimodal.yaml \
    --model-dir mirothinker-v17-multimodal

# Limit number of tasks (for testing)
bash scripts/run_factual_eval.sh --model-dir chatgpt-text-only --max-tasks 5

# Resume a previous run
bash scripts/run_factual_eval.sh --result-dir logs/factual-eval/prev_run
```

### Output Format

Each query produces a JSON result containing a `core_state` list:

```json
{
  "core_state": [
    {
      "statement": "The statement being verified",
      "verification": "Right | Wrong | Unknown | Conflict",
      "evidence": [
        { "source": "Evidence source URL", "excerpt": "Quoted key text from source" }
      ],
      "reasoning": "Explanation of the verification reasoning and process"
    }
  ]
}
```

**Key Metric:** Correct Statement Ratio = Right / (Right + Wrong + Unknown)

---

## 2. Point Quality

Adaptive Point-wise Quality Evaluation system that dynamically generates evaluation dimensions, criteria, and weights for each query task, enabling fine-grained quality assessment.

### How It Works

The evaluation pipeline consists of 5 stages:

1. **Dimension Generation**: LLM generates 1-3 task-specific additional dimensions (supplementing 4 fixed dimensions)
2. **Weight Assignment**: Assigns normalized weights to all dimensions (summing to 1.0)
3. **Criteria Generation**: Generates 1-10 specific evaluation criteria per dimension
4. **Per-criteria Scoring**: Scores the report against each criterion (0-10)
5. **Hierarchical Aggregation**: Criteria scores -> dimension scores -> total weighted score

**4 Fixed Dimensions:**
| Dimension | Description |
|-----------|-------------|
| Coverage | Breadth, depth, and relevance of coverage |
| Insight | Depth, originality, logic, and analytical value |
| Instruction Following | Accuracy in meeting all query requirements |
| Clarity | Readability, fluency, structure, and ease of understanding |

### Directory Structure

```
point_quality/
├── deepresearcharena/         # Core evaluation framework
│   ├── evaluator/             # Evaluator implementations
│   │   ├── base_evaluator.py
│   │   ├── pointwise_core.py
│   │   └── pointwise_evaluator.py
│   ├── prompts/               # Prompt templates
│   ├── cache/                 # Caching system
│   ├── config/                # YAML configuration files
│   └── utils/                 # LLM calls, config loading
├── run_batch_eval.py          # Entry point script
├── .env.template              # Environment variables template
└── requirements.txt
```

### Data Loading

Takes a model result JSON file from the shared `data/` directory as input via the `--input` flag:

```bash
python run_batch_eval.py --input ../data/method_results/mirothinker_v17_text_100.json --model_name mirothinker_v17
```

The input file is a JSON array of entries, each containing `query`, `rewritten_query`, `response`, and optional `files` for attachments. Attachment paths are resolved relative to `{data_dir}/input_queries/multimodal/`.

### Setup

```bash
cd point_quality

pip install -r requirements.txt

# Configure API keys (copy template and fill in values)
cp .env.template .env
# Edit .env with your API keys
```

### Usage

The API type is auto-detected from the model name: models with a provider prefix (e.g., `openai/gpt-5`) use OpenRouter, plain model names (e.g., `gpt-5.2`) use OpenAI directly.

```bash
# Text-only evaluation
python run_batch_eval.py --input ../data/method_results/mirothinker_v17_text_100.json --model_name mirothinker_v17

# Multimodal evaluation (attachments resolved automatically from data/input_queries/multimodal/)
python run_batch_eval.py --input ../data/method_multimodal_results/mirothinker_v17_multimodal.json --model_name mirothinker_v17

# Specify evaluator model and query count
python run_batch_eval.py --input ../data/method_results/claude_text_100.json --model_name claude \
    --evaluator_model gpt-5.2 --max_queries 50

# Reuse criteria from a previous run (only re-score)
python run_batch_eval.py --input ../data/method_results/gemini_text_100.json --model_name gemini \
    --criteria_file outputs/mirothinker_v17_results.json
```

### Configuration

Configuration file located at `deepresearcharena/config/pointwise.yaml`. Key fields:

```yaml
evaluator_model:
  name: "gpt-5.2"             # Judge LLM
  api_type: "auto"             # auto (detect by model name), openai, or openrouter
  temperature: 0.1

evaluation:
  max_workers: 20              # Parallel workers
  scoring:
    score_range: [0, 10]
    decimal_places: 2
```

### Output Format

```json
{
  "summary": {
    "mirothinker": {
      "average_total_score": 8.807,
      "dimension_averages": {
        "coverage_score": 8.5,
        "insight_score": 8.6,
        "instruction_following_score": 9.48,
        "clarity_score": 9.36,
        "total_weighted_score": 8.807
      }
    }
  },
  "query_results": { ... }
}
```

---

## 3. Process Eval

Evaluates the quality of a model's research process (intermediate reasoning, search strategies, etc.) and the alignment between the process and the final report.

### How It Works

The evaluation consists of two phases:

**Phase 1 - Structuring:**
- Auto-detects different models' process trace formats (JSON array, block tags, step tags, plain text, etc.)
- Uses LLM to unify heterogeneous formats into a structured JSON schema (step list + global findings)

**Phase 2 - Evaluation:**
- **Intrinsic Evaluation**: 5 dimensions assessing the research process quality itself
- **Alignment Evaluation**: 3 dimensions assessing consistency between process and report

**8 Evaluation Dimensions:**

| Type | Dimension | Description |
|------|-----------|-------------|
| Intrinsic | search_breadth | Diversity of sources and angles explored |
| Intrinsic | analytical_depth | Depth of analysis and insight |
| Intrinsic | progressive_refinement | Ability to iteratively deepen investigation |
| Intrinsic | critical_thinking | Cross-verification and critical reasoning |
| Intrinsic | efficiency | Conciseness and effectiveness of steps |
| Alignment | findings_to_report | Fraction of process findings covered in the report |
| Alignment | report_to_process | Whether report claims can be traced back to the process |
| Alignment | contradiction | Consistency between process and report (10 = fully consistent) |

### Directory Structure

```
process_eval/
├── run_pipeline.py            # Entry point script
├── config/
│   ├── process_eval.yaml      # Text-only evaluation config
│   └── process_eval_multimodal.yaml  # Multimodal evaluation config
├── process_evaluator/         # Core package
│   ├── pipeline.py            # Pipeline orchestrator
│   ├── data_loader.py         # Data loading
│   ├── preprocessors/         # Multi-format preprocessors (auto-detection)
│   ├── structuring/           # LLM-based structuring
│   ├── evaluation/            # Intrinsic + alignment evaluators
│   ├── cache/                 # Thread-safe JSON caching
│   └── utils/                 # LLM client, config loading
├── .env.template              # Environment variables template
└── requirements.txt
```

### Data Loading

Reads model result files directly from the shared data directory. The data loader auto-discovers files by matching `{model_name}_{data_type}*.json` pattern:

```yaml
# config/process_eval.yaml
data:
  data_dir: "../data/method_results"           # Text-only
# config/process_eval_multimodal.yaml
data:
  data_dir: "../data/method_multimodal_results" # Multimodal
```

The `process` field in each model result file contains the research process trace; the `response` field contains the final report.

### Setup

```bash
cd process_eval

pip install -r requirements.txt   # openai, pyyaml, tqdm, python-dotenv

# Configure API keys (copy template and fill in values)
cp .env.template .env
# Edit .env with your OPENAI_API_KEY (or OPENROUTER_API_KEY)
```

### Usage

Supports both **text-only** and **multimodal** evaluation, selected via config file:
- Text-only (default): `config/process_eval.yaml` — reads from `data/method_results/`
- Multimodal: `config/process_eval_multimodal.yaml` — reads from `data/method_multimodal_results/`

```bash
# Text-only evaluation (default)
python run_pipeline.py

# Multimodal evaluation
python run_pipeline.py --config config/process_eval_multimodal.yaml

# Run structuring phase only
python run_pipeline.py --phase phase1

# Run evaluation phase only (requires phase1 to be completed first)
python run_pipeline.py --phase phase2

# Specify models and entry count
python run_pipeline.py --models claude gemini --max-entries 10

# Clear cache and re-run
python run_pipeline.py --clear-cache
```

### Output Format

```json
{
  "summary": {
    "mirothinker": {
      "search_breadth": { "avg": 8.2, "count": 100 },
      "analytical_depth": { "avg": 7.8, "count": 100 },
      "progressive_refinement": { "avg": 8.1, "count": 100 },
      "critical_thinking": { "avg": 7.5, "count": 100 },
      "efficiency": { "avg": 7.9, "count": 100 },
      "findings_to_report": { "avg": 8.3, "count": 100 },
      "report_to_process": { "avg": 7.6, "count": 100 },
      "contradiction": { "avg": 8.8, "count": 100 },
      "overall_avg": 8.03
    }
  },
  "entry_results": { ... }
}
```

---

## Module Comparison

| Aspect | Factual Eval | Point Quality | Process Eval |
|--------|-------------|---------------|--------------|
| **Goal** | Report factual correctness | Report content quality | Research process quality |
| **Method** | Agent + web search verification | LLM multi-dimension scoring | LLM structuring + scoring |
| **Data Input** | response (report text) | response (report text) | process + response |
| **Scoring Scale** | Right / Wrong / Unknown | 0-10 continuous | 1-10 integer |
| **Judge LLM** | GPT-5-mini (default) | GPT-5.2 (default) | GPT-5.2 (default) |
| **Parallelism** | Async + semaphore | ThreadPoolExecutor | ThreadPoolExecutor |
| **Caching** | None (agent state) | Multi-level JSON cache | Three-level JSON cache |
| **Python** | >= 3.11 (uv) | >= 3.10 (pip) | >= 3.10 (pip) |

## License

Apache-2.0
