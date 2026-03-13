"""Shared configuration for the task generation pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SEED_PATTERNS_FILE = INPUT_DIR / "seed_patterns.json"

# ---------------------------------------------------------------------------
# API keys (loaded from .env)
# ---------------------------------------------------------------------------
load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")

DEFAULT_MODEL = "openai/gpt-5.2"

# ---------------------------------------------------------------------------
# Topic Pool — 12 topics with subtopics for trend searching
# ---------------------------------------------------------------------------
TOPIC_POOL = [
    {
        "topic": "AI Policy & Regulation",
        "subtopics": ["EU AI Act implementation", "US state AI laws", "AI safety frameworks"],
    },
    {
        "topic": "Cybersecurity",
        "subtopics": ["zero-day exploits", "agentic SOC", "AI-powered social engineering"],
    },
    {
        "topic": "Finance & Macro",
        "subtopics": ["central bank policy", "sovereign debt", "infrastructure investment"],
    },
    {
        "topic": "Crypto & Digital Assets",
        "subtopics": ["stablecoin regulation", "DeFi compliance", "CBDC adoption"],
    },
    {
        "topic": "Healthcare & Pharma",
        "subtopics": ["gene therapy trials", "GLP-1 market dynamics", "FDA regulatory shifts"],
    },
    {
        "topic": "International Trade",
        "subtopics": ["global supply chain restructuring", "free trade agreements impact",
                       "cross-border regulatory harmonization"],
    },
    {
        "topic": "AI Engineering",
        "subtopics": ["LLM benchmarking", "agentic coding tools", "model deployment architecture"],
    },
    {
        "topic": "Climate & Energy",
        "subtopics": ["data center sustainability", "carbon pricing", "grid constraints"],
    },
    {
        "topic": "Education & Workforce",
        "subtopics": ["AI in K-12 policy", "workforce reskilling", "immigration & talent"],
    },
    {
        "topic": "Legal & Compliance",
        "subtopics": ["AI privilege doctrine", "GDPR enforcement", "algorithmic discrimination"],
    },
    {
        "topic": "Biotech & Science",
        "subtopics": ["computational biology", "quantum computing", "open access publishing"],
    },
    {
        "topic": "Supply Chain & Industrial",
        "subtopics": ["nearshoring trends", "autonomous logistics", "semiconductor supply"],
    },
]

TOPIC_NAMES = [t["topic"] for t in TOPIC_POOL]

# ---------------------------------------------------------------------------
# Canonical domain labels (11 fixed labels for consistent downstream use)
# ---------------------------------------------------------------------------
CANONICAL_DOMAINS = {
    "finance", "policy", "tech", "cybersecurity", "health",
    "science", "education", "legal", "energy", "trade", "crypto",
}


def normalize_domain(raw: str) -> str:
    """Map a free-form domain string to one of the canonical labels."""
    r = raw.lower().strip()

    # Direct substring match
    for canon in CANONICAL_DOMAINS:
        if canon in r:
            return canon

    # Keyword fallbacks
    _FALLBACK = [
        (["macro", "invest", "banking", "econom"], "finance"),
        (["regulat", "compliance", "legislation", "governance"], "policy"),
        (["software", "engineer", "ai ", "ml ", "model", "llm", "machine learn"], "tech"),
        (["medical", "pharma", "drug", "clinical", "biotech"], "health"),
        (["supply chain", "logistic", "industrial", "manufactur"], "trade"),
        (["workforce", "hiring", "talent", "hr", "human resource"], "education"),
        (["climate", "carbon", "grid", "renewable"], "energy"),
        (["quantum", "biology", "research", "publish"], "science"),
        (["geopolit", "international"], "trade"),
    ]
    for keywords, domain in _FALLBACK:
        if any(w in r for w in keywords):
            return domain

    return "tech"  # safe fallback
