# MiroEval

MiroEval is a comprehensive evaluation framework for Deep Research systems, providing automated assessment across three complementary dimensions: **Factual** correctness, **Point**-wise quality, and **Process** quality.

All three evaluation modules share a unified `data/` directory as their input data source.

## Architecture

```
MiroEval/
├── data/                      # Shared data directory (input queries + model results)
│   ├── input_queries/         # Evaluation query sets
│   ├── method_results/        # Text-only model results (14 models × 100 queries)
│   └── method_multimodal_results/  # Multimodal model results (10 models × 47 queries)
├── factual-eval/              # Factual evaluation (MiroFlow-based fact-checking agent)
├── point_quality/             # Quality evaluation (adaptive point-wise scoring)
└── process_eval/              # Process evaluation (intrinsic process quality + report alignment)
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

One JSON file per model, containing complete query-response pairs.

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
3. **Verdict**: Labels each factual statement as `Right` (correct) / `Wrong` (incorrect) / `Unknown` (unverifiable)

### Directory Structure

```
factual-eval/
├── config/                    # Hydra configuration files
│   ├── benchmark/             # Benchmark configs (factual-eval.yaml, etc.)
│   ├── llm/                   # LLM model configs
│   ├── tool/                  # Tool configs (search, browsing, etc.)
│   ├── prompts/               # Prompt templates
│   └── benchmark_factual-eval_*.yaml  # Per-model evaluation configs
├── miroflow/                  # MiroFlow core framework
│   ├── agents/                # Agent implementations (iterative + rollback)
│   ├── benchmark/             # Evaluation runners and verifiers
│   ├── llm/                   # Multi-provider LLM support
│   ├── tool/                  # MCP server tool integration
│   ├── io_processor/          # I/O processors (segmentation, summarization, etc.)
│   └── utils/                 # Utility functions
├── scripts/
│   ├── run_factual_eval.sh    # Main run script
│   └── benchmark/             # Per-model benchmark scripts
└── pyproject.toml             # Dependencies (Python >= 3.11)
```

### Data Loading

- Data path is specified via Hydra config: `data_dir: "${data_dir}/factual-eval/mirothinker-text-only-gen"`
- Data must be preprocessed into `standardized_data.jsonl` format in the target directory
- Each entry contains `task_id`, `task_question` (query + report segment), `ground_truth`, etc.
- For multimodal queries, associated attachment files (images, PDFs, etc.) are stored in `data/input_queries/multimodal-attachments/<query_id>/` and referenced via the `files` field in the query

### Setup

```bash
cd factual-eval

# Install dependencies
uv sync

# Configure API keys
export OPENAI_API_KEY="..."       # GPT models
export SERPER_API_KEY="..."       # Google search
# Optional: ANTHROPIC_API_KEY, OPENROUTER_API_KEY, JINA_API_KEY, etc.
```

### Usage

```bash
# Run factual evaluation with default config
bash scripts/run_factual_eval.sh

# Specify model config
bash scripts/run_factual_eval.sh --config config/benchmark_factual-eval_mirothinker-text-only.yaml

# Limit number of tasks (for testing)
bash scripts/run_factual_eval.sh --max-tasks 5

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
      "verification": "Right | Wrong | Unknown",
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
├── example_pointwise_usage.py # Usage example
└── requirements.txt
```

### Data Loading

Reads from the `data/` directory (path specified in YAML config):

```
<data_dir>/
├── input_queries/
│   └── query.jsonl            # One query per line: {id, topic, language, prompt}
└── method_results/
    └── <model_name>/          # One directory per model
        └── <query_id>.json    # Contains entries[].response
```

### Setup

```bash
cd point_quality

pip install -r requirements.txt

# Configure API key (uses OpenRouter to call the judge LLM)
export OPENROUTER_API_KEY="..."
```

### Usage

```bash
# Run with default config
python example_pointwise_usage.py

# Specify config file
python example_pointwise_usage.py --config deepresearcharena/config/pointwise.yaml

# Specify models and query count
python example_pointwise_usage.py --models mirothinker gemini --max_queries 50

# Dry-run validation
python example_pointwise_usage.py --dry_run
```

### Configuration

Configuration file located at `deepresearcharena/config/pointwise.yaml`. Key fields:

```yaml
evaluator_model:
  name: "google/gemini-2.5-pro"   # Judge LLM
  api_type: "openrouter"
  temperature: 0.1

target_models:                     # Models to evaluate
  - "mirothinker"

evaluation:
  max_workers: 20                  # Parallel workers
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
├── experiments/               # Experimental comparison scripts
└── requirements.txt
```

### Data Loading

Reads model result files directly from the shared data directory:

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

export OPENAI_API_KEY="..."        # Or OPENROUTER_API_KEY
```

### Usage

```bash
# Full pipeline
python run_pipeline.py

# Run structuring phase only
python run_pipeline.py --phase phase1

# Run evaluation phase only (requires phase1 to be completed first)
python run_pipeline.py --phase phase2

# Specify models and entry count
python run_pipeline.py --models claude gemini --max-entries 10

# Multimodal evaluation
python run_pipeline.py --config config/process_eval_multimodal.yaml

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
| **Judge LLM** | GPT-5-mini (default) | Gemini 2.5 Pro | GPT-5.2 (default) |
| **Parallelism** | Async + semaphore | ThreadPoolExecutor | ThreadPoolExecutor |
| **Caching** | None (agent state) | Multi-level JSON cache | Three-level JSON cache |
| **Python** | >= 3.11 (uv) | >= 3.10 (pip) | >= 3.10 (pip) |

## License

Apache-2.0

