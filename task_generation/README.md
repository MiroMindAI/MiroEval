# Task Generation

Generate high-quality deep-research evaluation queries through a 6-step pipeline combining real user query patterns, real-time web trends, LLM generation, and multi-stage filtering.

## Pipeline Overview

| Step | What it does | How |
|------|-------------|-----|
| **0. Load seeds** | Read anonymized query examples by topic | `input/seed_patterns.json` |
| **1. Fetch trends** | Real-time Google search per subtopic | Serper API, parallel |
| **2. Generate** | LLM creates queries with trend + seed context | GPT-4o, 6 per topic x 12 topics |
| **3. Search validate** | Verify each query has real search results | Serper: >=3 results, >=2 unique domains |
| **4. DR filter** | LLM judges if deep research is truly needed | Confidence >= 0.7 |
| **5. Quality filter** | Generate baseline answer, keep only hard ones | quality in {low, medium} AND requires_search AND score <= 0.75 |
| **6. Export** | Normalize domains, format output | 11 canonical domain labels |

Each step caches its output as `intermediate_N_*.json` -- rerunning skips completed steps. Use `--clean` to reset.

## Quick Start

```bash
pip install openai requests python-dotenv

# Copy and fill in API keys
cp .env.example .env

# Run the pipeline (seed_patterns.json must exist)
python pipeline.py

# Force clean rerun
python pipeline.py --clean
```

## Output Format

```json
{
  "id": 1,
  "chat_id": "uuid",
  "query": "Full research query text...",
  "files": [],
  "annotation": {
    "category": "text-auto",
    "language": "en",
    "domain": "finance",
    "topic": "Finance & Macro",
    "persona": "hedge fund PM",
    "anchored_event": "structural trend grounding this query",
    "time_sensitive": false,
    "dr_confidence": 0.92,
    "quality_score": 0.35,
    "search_complexity": "High",
    "search_validation": {
      "result_count": 8,
      "source_diversity": 5,
      "has_recent_results": true
    }
  }
}
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `openai/gpt-4o` | LLM model |
| `--num-topics` | `12` | Topics from the pool |
| `--num-per-topic` | `6` | Queries generated per topic |
| `--max-workers` | `10` | Thread pool concurrency |
| `--dr-threshold` | `0.7` | Min DR confidence to pass |
| `--quality-threshold` | `0.75` | Max baseline quality score (lower = harder) |
| `--clean` | `false` | Clear caches and rerun |

## Design Decisions

- **Search grounding**: Trend injection (step 1) + search validation (step 3) ensure queries are answerable with real web content, not LLM hallucinations.
- **Dual filtering**: DR filter removes trivially answerable queries; quality filter removes queries the LLM can answer well without search. Together they select genuinely challenging evaluation items.
- **11 canonical domains**: `finance, policy, tech, cybersecurity, health, science, education, legal, energy, trade, crypto` -- consistent labels for downstream analysis.
- **Time sensitivity**: Queries target structural trends (3-6 month relevance), not fleeting news.
