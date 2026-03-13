#!/usr/bin/env python3
"""
Task Generation — Main Pipeline

Generates deep-research evaluation queries in 6 steps:

  Step 1  Fetch real-time trends (Serper)
  Step 2  LLM generates queries (seeds + trends → prompt)
  Step 3  Search validation (verify searchability)
  Step 4  Deep Research filter (LLM judges necessity)
  Step 5  Baseline + quality filter (keep only hard queries)
  Step 6  Export final dataset

Usage:
    python pipeline.py                         # default run
    python pipeline.py --clean                 # clear cache & rerun
    python pipeline.py --num-per-topic 8 --max-workers 12
"""

import argparse
import json
import logging
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from openai import OpenAI

from config import (
    DEFAULT_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OUTPUT_DIR,
    SEED_PATTERNS_FILE,
    SERPER_API_KEY,
    TOPIC_POOL,
    normalize_domain,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")


# ═══════════════════════════════════════════════════════════════════════════
# LLM & Search clients
# ═══════════════════════════════════════════════════════════════════════════

class LLMClient:
    """Synchronous OpenAI-compatible client with retry (for ThreadPoolExecutor)."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, timeout=120,
        )
        self.model = model

    def call(self, messages: list, max_tokens: int = 4096,
             temperature: float = 0.7) -> Optional[str]:
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                )
                return r.choices[0].message.content or ""
            except Exception as e:
                log.warning(f"LLM error (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        return None

    def call_json(self, messages: list, **kw) -> Optional[dict]:
        raw = self.call(messages, **kw)
        return _parse_json(raw) if raw else None


def _parse_json(text: str):
    """Best-effort JSON extraction from LLM output (handles markdown fences)."""
    if not text:
        return None
    # Try fenced block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw object
    for open_ch, close_ch in [("{", "}"), ("[", "]")]:
        start = text.find(open_ch)
        if start >= 0:
            end = text.rfind(close_ch) + 1
            if end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
    return None


def search_serper(query: str, num_results: int = 8) -> list[dict]:
    """Google search via Serper API."""
    if not SERPER_API_KEY:
        return []
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=15,
        )
        if r.status_code == 200:
            return [
                {"title": i.get("title", ""), "snippet": i.get("snippet", ""),
                 "link": i.get("link", ""), "date": i.get("date", "")}
                for i in r.json().get("organic", [])[:num_results]
            ]
    except Exception as e:
        log.warning(f"Serper error: {e}")
    return []


# ═══════════════════════════════════════════════════════════════════════════
# Step 0 — Load seed queries from seed_patterns.json
# ═══════════════════════════════════════════════════════════════════════════

def load_seeds() -> Dict[str, List[str]]:
    """Load anonymized seed queries grouped by topic."""
    seeds: Dict[str, List[str]] = defaultdict(list)

    if not SEED_PATTERNS_FILE.exists():
        log.warning(f"{SEED_PATTERNS_FILE} not found — run prepare_seeds.py first.")
        return dict(seeds)

    with open(SEED_PATTERNS_FILE) as f:
        data = json.load(f)

    for sq in data.get("seed_queries", []):
        q = sq.get("query", "")
        if q and len(q) > 20:
            seeds[sq.get("topic", "Other")].append(q)

    total = sum(len(v) for v in seeds.values())
    log.info(f"Loaded {total} seeds across {len(seeds)} topics")
    return dict(seeds)


def match_seeds(topic_name: str, seeds: Dict[str, List[str]], n: int = 5) -> List[str]:
    """Pick the best seed queries for a topic (direct match → keyword → fallback)."""
    matched = list(seeds.get(topic_name, []))

    if len(matched) < n:
        topic_words = {w.lower() for w in topic_name.split() if len(w) > 3}
        for key, vals in seeds.items():
            if key != topic_name and topic_words & {w.lower() for w in key.split() if len(w) > 3}:
                matched.extend(vals)

    if len(matched) < 2:
        for v in seeds.values():
            matched.extend(v)

    seen: set[str] = set()
    unique = []
    for q in matched:
        k = q[:100]
        if k not in seen:
            seen.add(k)
            unique.append(q)
    return unique[:n]


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Fetch real-time trends via Serper
# ═══════════════════════════════════════════════════════════════════════════

def fetch_trends(topics: list[dict], max_workers: int = 5) -> Dict[str, list]:
    """Search each subtopic for recent news and return grouped by topic."""
    log.info(f"Fetching trends for {len(topics)} topics...")
    trends: Dict[str, list] = {}

    def _fetch_one(topic: dict):
        results = []
        for sub in topic.get("subtopics", []):
            month_label = datetime.now().strftime("%B %Y")
            for r in search_serper(f"{sub} news developments {month_label}", 5):
                r["subtopic"] = sub
                results.append(r)
        return topic["topic"], results

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(_fetch_one, t) for t in topics]):
            try:
                name, results = fut.result()
                trends[name] = results
                log.info(f"  {name}: {len(results)} items")
            except Exception as e:
                log.error(f"  Trend fetch error: {e}")

    return trends


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — LLM generates queries
# ═══════════════════════════════════════════════════════════════════════════

_GEN_PROMPT = """You are a deep research query designer. Generate {n} realistic, expert-level queries that REQUIRE deep web search and multi-source synthesis to answer well.

## Topic Area
{topic} — subtopics: {subtopics}

## Recent Real-World Context (use as background — DO NOT invent events)
{trends}

## Example Queries from Real Users (match this specificity and natural language)
{seeds}

## Requirements
1. Frame queries around structural trends, ongoing debates, or comparative analyses that remain meaningful over 3-6 months. Do NOT pin to a single fleeting news event.
2. Each query must need 2-3 rounds of search from different angles.
3. Queries must require synthesis across multiple credible sources (reports, papers, news, official documents).
4. Mix personas: expert professionals (analyst, engineer, physician) and informed non-experts (grad student, startup founder, journalist, investor).
5. About 40% may include broad time context (e.g., "in 2025-2026") but avoid exact dates/weeks.
6. Length: 40-120 words. Specific but not verbose.
7. Include cross-country comparisons, conflicting expert views, forward-looking predictions, or quantitative analysis.
8. NO generic or textbook questions — each must need CURRENT information.
9. AVOID politically sensitive topics: specific leaders, partisan conflicts, military operations, territorial disputes, regime criticism. Focus on policy mechanisms, industry, technology, economics.
10. Domain label must be one of: finance, policy, tech, cybersecurity, health, science, education, legal, energy, trade, crypto.

Output ONLY a JSON array:
[
  {{
    "query": "...",
    "domain": "...",
    "persona": "...",
    "anchored_event": "...",
    "time_sensitive": true/false,
    "expected_difficulty": "high/very_high"
  }}
]"""


def generate_queries(
    llm: LLMClient, topics: list[dict], trends: Dict[str, list],
    seeds: Dict[str, List[str]], num_per_topic: int, max_workers: int,
) -> list[dict]:
    """Generate queries for all topics in parallel."""
    log.info(f"Generating {num_per_topic} queries × {len(topics)} topics...")

    def _gen_one(topic: dict):
        name = topic["topic"]
        subtopics = ", ".join(topic.get("subtopics", []))
        t_list = trends.get(name, [])
        s_list = match_seeds(name, seeds)

        trends_txt = ("\n".join(
            f"- [{t.get('date', 'recent')}] {t['title']}: {t['snippet'][:150]}"
            for t in t_list[:15]
        ) if t_list else "(No trends found — use known recent developments)")

        seeds_txt = ("\n".join(f"- {s[:300]}" for s in s_list[:5])
                     if s_list else "(No seeds available)")

        prompt = _GEN_PROMPT.format(
            n=num_per_topic, topic=name, subtopics=subtopics,
            trends=trends_txt, seeds=seeds_txt,
        )
        raw = llm.call([{"role": "user", "content": prompt}],
                        max_tokens=8192, temperature=0.8)
        parsed = _parse_json(raw) if raw else None
        if parsed is None:
            log.error(f"  [{name}] Parse failed")
            return []
        qs = parsed if isinstance(parsed, list) else parsed.get("queries", [parsed])
        for q in qs:
            q["topic"] = name
        log.info(f"  [{name}] → {len(qs)} queries")
        return qs

    all_queries: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed([ex.submit(_gen_one, t) for t in topics]):
            try:
                all_queries.extend(fut.result())
            except Exception as e:
                log.error(f"  Generation error: {e}")

    log.info(f"Total generated: {len(all_queries)}")
    return all_queries


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Search validation
# ═══════════════════════════════════════════════════════════════════════════

def validate_searchability(queries: list[dict], max_workers: int) -> list[dict]:
    """Keep only queries that produce real search results (≥3 results, ≥2 sources)."""
    log.info(f"Search-validating {len(queries)} queries...")

    def _val(q: dict, idx: int):
        results = search_serper(q["query"][:200], 10)
        domains = set()
        for r in results:
            try:
                domains.add(urlparse(r.get("link", "")).netloc)
            except Exception:
                pass
        rc, sd = len(results), len(domains)
        recent = any("2026" in (r.get("date", "") + r.get("snippet", "")) for r in results)
        ok = rc >= 3 and sd >= 2
        return idx, {
            "searchable": ok, "result_count": rc, "source_diversity": sd,
            "has_recent_results": recent, "top_sources": list(domains)[:5],
        }

    passed: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_val, q, i): i for i, q in enumerate(queries)}
        for f in as_completed(futs):
            idx = futs[f]
            try:
                _, info = f.result()
                queries[idx]["search_validation"] = info
                if info["searchable"]:
                    passed.append(queries[idx])
                else:
                    log.info(f"  [{idx}] FAIL: results={info['result_count']} sources={info['source_diversity']}")
            except Exception as e:
                log.error(f"  [{idx}] Validation error: {e}")

    log.info(f"Search validation: {len(passed)}/{len(queries)} pass")
    return passed


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Deep Research filter  (LLM judges if deep research is truly needed)
# ═══════════════════════════════════════════════════════════════════════════

_DR_PROMPT = """Evaluate whether this query truly requires deep web search and multi-source synthesis.

Query: {query}

Criteria:
1. Needs up-to-date information (not answerable from pre-2023 knowledge alone)
2. Needs cross-verification from multiple credible sources
3. Needs multi-angle, multi-layered investigation
4. Is NOT a textbook question LLMs can already answer well

Output JSON only:
{{
  "needs_deep_research": true/false,
  "confidence_score": 0.0-1.0,
  "reasoning": "Brief rationale (50 words)",
  "search_complexity": "High"/"Medium"/"Low"
}}"""


def filter_deep_research(
    llm: LLMClient, queries: list[dict], threshold: float, max_workers: int,
) -> list[dict]:
    """Keep queries where LLM says deep research is needed with high confidence."""
    log.info(f"DR filtering {len(queries)} queries (threshold={threshold})...")

    def _filt(q: dict, idx: int):
        r = llm.call_json([{"role": "user", "content": _DR_PROMPT.format(query=q["query"])}],
                          temperature=0.3)
        return idx, r

    passed: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_filt, q, i): i for i, q in enumerate(queries)}
        for f in as_completed(futs):
            idx = futs[f]
            try:
                _, r = f.result()
                if r:
                    queries[idx]["dr_filter"] = r
                    if r.get("needs_deep_research") and r.get("confidence_score", 0) >= threshold:
                        passed.append(queries[idx])
                    else:
                        log.info(f"  [{idx}] DR-drop conf={r.get('confidence_score', 0):.2f}")
            except Exception as e:
                log.error(f"  [{idx}] DR error: {e}")

    log.info(f"DR filter: {len(passed)}/{len(queries)} pass")
    return passed


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Baseline answer + quality filter  (keep only hard queries)
# ═══════════════════════════════════════════════════════════════════════════

_BASELINE_PROMPT = """Based solely on your existing knowledge, without any search tools, answer this query concisely but thoroughly.

Query: {query}

If information is uncertain or possibly outdated, say so. Do not fabricate facts."""

_QUALITY_PROMPT = """Assess this no-search baseline answer for the given query.

Query: {query}
No-search answer: {baseline}

Output JSON only:
{{
  "overall_quality": "low"/"medium"/"high",
  "quality_score": 0.0-1.0,
  "timeliness_score": 0.0-1.0,
  "requires_search": true/false,
  "missing_aspects": "key missing info (brief)"
}}"""


def filter_quality(
    llm: LLMClient, queries: list[dict], threshold: float, max_workers: int,
) -> list[dict]:
    """Generate baseline answers and keep only queries the LLM answers poorly."""
    log.info(f"Quality filtering {len(queries)} queries (threshold={threshold})...")

    def _proc(q: dict, idx: int):
        baseline = llm.call(
            [{"role": "user", "content": _BASELINE_PROMPT.format(query=q["query"])}],
            max_tokens=3000, temperature=0.3,
        )
        if not baseline:
            return idx, None
        qa = llm.call_json(
            [{"role": "user", "content": _QUALITY_PROMPT.format(
                query=q["query"], baseline=baseline[:5000])}],
            temperature=0.3,
        )
        return idx, qa

    passed: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_proc, q, i): i for i, q in enumerate(queries)}
        for f in as_completed(futs):
            idx = futs[f]
            try:
                _, qa = f.result()
                if not qa:
                    continue
                queries[idx]["quality_assessment"] = qa
                quality = qa.get("overall_quality", "high")
                score = qa.get("quality_score", 1.0)
                needs = qa.get("requires_search", False)
                if quality in ("low", "medium") and needs and score <= threshold:
                    passed.append(queries[idx])
                else:
                    log.info(f"  [{idx}] Quality-drop q={quality} s={score:.2f} search={needs}")
            except Exception as e:
                log.error(f"  [{idx}] Quality error: {e}")

    log.info(f"Quality filter: {len(passed)}/{len(queries)} pass")
    return passed


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Export
# ═══════════════════════════════════════════════════════════════════════════

def export(queries: list[dict], output_file: str) -> list[dict]:
    """Export final dataset in standard annotation format."""
    output = []
    for i, q in enumerate(queries, 1):
        output.append({
            "id": i,
            "chat_id": str(uuid.uuid4()),
            "query": q["query"],
            "files": [],
            "annotation": {
                "category": "text-auto",
                "language": "en",
                "domain": normalize_domain(q.get("domain", "other")),
                "topic": q.get("topic", ""),
                "persona": q.get("persona", ""),
                "anchored_event": q.get("anchored_event", ""),
                "time_sensitive": q.get("time_sensitive", False),
                "dr_confidence": q.get("dr_filter", {}).get("confidence_score", 0),
                "quality_score": q.get("quality_assessment", {}).get("quality_score", 0),
                "search_complexity": q.get("dr_filter", {}).get("search_complexity", ""),
                "search_validation": {
                    "result_count": q.get("search_validation", {}).get("result_count", 0),
                    "source_diversity": q.get("search_validation", {}).get("source_diversity", 0),
                    "has_recent_results": q.get("search_validation", {}).get("has_recent_results", False),
                },
            },
        })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info(f"Exported {len(output)} queries → {output_file}")
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Caching helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save(data, name: str):
    p = OUTPUT_DIR / f"intermediate_{name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Cached: {p}")


def _load(name: str):
    p = OUTPUT_DIR / f"intermediate_{name}.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        log.info(f"Loaded cache: {p} ({len(d) if isinstance(d, (list, dict)) else '?'} items)")
        return d
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def run(
    model: str = DEFAULT_MODEL,
    num_topics: int = 12,
    num_per_topic: int = 6,
    max_workers: int = 10,
    dr_threshold: float = 0.7,
    quality_threshold: float = 0.75,
    output_file: str = str(OUTPUT_DIR / "generated.json"),
):
    log.info("=" * 60)
    log.info("Task Generation V2 Pipeline")
    log.info(f"Model={model}  Topics={num_topics}  PerTopic={num_per_topic}  Workers={max_workers}")
    log.info("=" * 60)

    llm = LLMClient(model)
    topics = TOPIC_POOL[:num_topics]

    # Step 0: seeds
    seeds = load_seeds()

    # Step 1: trends
    log.info("\n--- Step 1: Fetch trends ---")
    trends = _load("1_trends")
    if trends is None:
        trends = fetch_trends(topics, min(max_workers, 5))
        _save(trends, "1_trends")

    # Step 2: generate
    log.info("\n--- Step 2: Generate queries ---")
    generated = _load("2_generated")
    if generated is None:
        generated = generate_queries(llm, topics, trends, seeds, num_per_topic, max_workers)
        _save(generated, "2_generated")
    if not generated:
        return log.error("No queries generated.") or []

    # Step 3: search validate
    log.info("\n--- Step 3: Search validation ---")
    validated = _load("3_validated")
    if validated is None:
        validated = validate_searchability(generated, max_workers)
        _save(validated, "3_validated")
    if not validated:
        return log.error("No queries passed validation.") or []

    # Step 4: DR filter
    log.info("\n--- Step 4: Deep Research filter ---")
    dr_filtered = _load("4_dr_filtered")
    if dr_filtered is None:
        dr_filtered = filter_deep_research(llm, validated, dr_threshold, max_workers)
        _save(dr_filtered, "4_dr_filtered")
    if not dr_filtered:
        return log.error("No queries passed DR filter.") or []

    # Step 5: quality filter
    log.info("\n--- Step 5: Baseline + quality filter ---")
    final = _load("5_quality_filtered")
    if final is None:
        final = filter_quality(llm, dr_filtered, quality_threshold, max_workers)
        _save(final, "5_quality_filtered")

    # Step 6: export
    log.info("\n--- Step 6: Export ---")
    output = export(final, output_file)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info(f"  Generated:       {len(generated)}")
    log.info(f"  Search-valid:    {len(validated)}")
    log.info(f"  DR-filtered:     {len(dr_filtered)}")
    log.info(f"  Quality-filtered:{len(final)}")
    log.info(f"  Final output:    {len(output)}")
    if output:
        dist = Counter(e["annotation"]["domain"] for e in output)
        log.info("  Domain distribution:")
        for d, c in dist.most_common():
            log.info(f"    {d}: {c}")

    return output


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Task Generation V2 Pipeline")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--num-topics", type=int, default=12)
    ap.add_argument("--num-per-topic", type=int, default=6)
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--dr-threshold", type=float, default=0.7)
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--output", default=str(OUTPUT_DIR / "generated.json"))
    ap.add_argument("--clean", action="store_true", help="Clear cache and rerun all steps")
    args = ap.parse_args()

    if args.clean:
        for f in OUTPUT_DIR.glob("intermediate_*.json"):
            f.unlink()
            log.info(f"Removed {f}")

    run(
        model=args.model,
        num_topics=args.num_topics,
        num_per_topic=args.num_per_topic,
        max_workers=args.max_workers,
        dr_threshold=args.dr_threshold,
        quality_threshold=args.quality_threshold,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
