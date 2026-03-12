# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal QA MCP server.

Provides a query_attachment tool that converts file attachments to text,
chunks them, builds an embedding index, and performs retrieval-augmented QA.
Results are cached in data/cache to avoid recomputation.
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import dotenv
import numpy as np
import tiktoken
from fastmcp import FastMCP
from filelock import FileLock
from openai import OpenAI

from miroflow.utils.file_content_utils import process_file_content

# Load .env so this subprocess picks up API keys / base URLs
dotenv.load_dotenv()

# Environment variables (set via tool config YAML)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CACHE_DIR = os.environ.get("CACHE_DIR", "data/cache")
CHUNK_TOKEN_SIZE = int(os.environ.get("CHUNK_TOKEN_SIZE", "2000"))
QA_MODEL = os.environ.get("QA_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

# Initialize FastMCP server
mcp = FastMCP("multimodal-qa-mcp-server")

# Lazy-initialized OpenAI client
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _get_cache_dir(file_path: str) -> Path:
    """Compute cache directory for a given file path using SHA256 hash."""
    abs_path = os.path.abspath(file_path)
    hash_hex = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()
    return Path(CACHE_DIR) / hash_hex


def _is_cache_valid(cache_dir: Path, file_path: str) -> bool:
    """Check if cached data exists and is still valid (file not modified)."""
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        return False

    required_files = ["text_content.txt", "chunks.json", "embeddings.npy"]
    for fname in required_files:
        if not (cache_dir / fname).exists():
            return False

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        # Check file modification time
        current_mtime = os.path.getmtime(file_path)
        if metadata.get("file_mtime") != current_mtime:
            return False
        # Check chunk token size consistency
        if metadata.get("chunk_token_size") != CHUNK_TOKEN_SIZE:
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Text conversion and chunking
# ---------------------------------------------------------------------------

def _convert_file_to_text(file_path: str) -> str:
    """Convert any supported file to text using existing file_content_utils."""
    result = process_file_content(
        task_description="",
        task_file_name=file_path,
        openai_api_key=OPENAI_API_KEY,
        openai_base_url=OPENAI_BASE_URL,
    )
    return result.strip()


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of approximately chunk_size tokens using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(enc.decode(chunk_tokens))
    return chunks


# ---------------------------------------------------------------------------
# LLM helpers (summarization and QA)
# ---------------------------------------------------------------------------

def _summarize_chunk(chunk_text: str, client: OpenAI) -> str:
    """Generate a brief summary of a single chunk."""
    try:
        response = client.chat.completions.create(
            model=QA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise summarization assistant. Summarize the given text in 2-3 sentences, capturing the key information.",
                },
                {"role": "user", "content": chunk_text[:8000]},  # safety truncation
            ],
            max_tokens=8192,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return (content or "").strip() or "[Empty response from model]"
    except Exception as e:
        return f"[Summary unavailable: {e}]"


def _summarize_chunks(chunks: list[str], client: OpenAI) -> list[str]:
    """Summarize all chunks sequentially (called during cache build)."""
    summaries = []
    for chunk in chunks:
        summaries.append(_summarize_chunk(chunk, client))
    return summaries


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str], client: OpenAI) -> np.ndarray:
    """Compute embeddings for a list of texts using OpenAI API."""
    # OpenAI embedding API supports batching up to ~2048 inputs
    # Truncate each text to avoid token limit (8191 for text-embedding-3-small)
    truncated = [t[:24000] for t in texts]  # ~6000 tokens safety margin

    all_embeddings = []
    batch_size = 64
    for i in range(0, len(truncated), batch_size):
        batch = truncated[i : i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        for item in response.data:
            all_embeddings.append(item.embedding)

    return np.array(all_embeddings, dtype=np.float32)


def _cosine_similarity(query_emb: np.ndarray, chunk_embs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query embedding and chunk embeddings."""
    # Normalize
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    chunk_norms = chunk_embs / (
        np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9
    )
    return chunk_norms @ query_norm


# ---------------------------------------------------------------------------
# Cache build and load
# ---------------------------------------------------------------------------

def _build_cache(file_path: str) -> dict:
    """
    Build the full cache for a file: convert, chunk, summarize, embed.
    Returns the cache dict with keys: chunks, embeddings, text.
    """
    client = _get_client()
    abs_path = os.path.abspath(file_path)
    cache_dir = _get_cache_dir(file_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Convert file to text
    text = _convert_file_to_text(file_path)
    if not text:
        raise ValueError(f"File produced no extractable text: {file_path}")

    # Chunk
    chunks = _chunk_text(text, CHUNK_TOKEN_SIZE)

    # Summarize each chunk
    summaries = _summarize_chunks(chunks, client)

    # Embed (use summaries + chunk text for better retrieval)
    embed_inputs = [
        f"Summary: {s}\n\nContent: {c[:4000]}" for s, c in zip(summaries, chunks)
    ]
    embeddings = _embed_texts(embed_inputs, client)

    # Build chunks data
    chunks_data = [
        {"index": i, "text": c, "summary": s}
        for i, (c, s) in enumerate(zip(chunks, summaries))
    ]

    # Write cache files
    with open(cache_dir / "text_content.txt", "w", encoding="utf-8") as f:
        f.write(text)

    with open(cache_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    np.save(cache_dir / "embeddings.npy", embeddings)

    metadata = {
        "file_path": abs_path,
        "file_mtime": os.path.getmtime(file_path),
        "chunk_token_size": CHUNK_TOKEN_SIZE,
        "num_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "qa_model": QA_MODEL,
        "created_at": time.time(),
    }
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "chunks": chunks_data,
        "embeddings": embeddings,
        "text": text,
    }


def _load_cache(cache_dir: Path) -> dict:
    """Load cached chunks and embeddings from disk."""
    with open(cache_dir / "chunks.json", "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    embeddings = np.load(cache_dir / "embeddings.npy")

    with open(cache_dir / "text_content.txt", "r", encoding="utf-8") as f:
        text = f.read()

    return {
        "chunks": chunks_data,
        "embeddings": embeddings,
        "text": text,
    }


def _get_or_build_cache(file_path: str) -> dict:
    """Get cache if valid, otherwise build it. Uses file lock for concurrency safety."""
    cache_dir = _get_cache_dir(file_path)
    lock_path = cache_dir.parent / f"{cache_dir.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path), timeout=600):
        if _is_cache_valid(cache_dir, file_path):
            return _load_cache(cache_dir)
        return _build_cache(file_path)


# ---------------------------------------------------------------------------
# Retrieval and QA
# ---------------------------------------------------------------------------

def _retrieve_chunks(
    query: str, cache: dict, client: OpenAI, top_k: int = 5
) -> list[dict]:
    """Retrieve the most relevant chunks for a query using embedding similarity."""
    chunks_data = cache["chunks"]
    embeddings = cache["embeddings"]

    if len(chunks_data) <= top_k:
        return chunks_data

    # Embed query
    query_emb = _embed_texts([query], client)[0]

    # Compute similarities
    similarities = _cosine_similarity(query_emb, embeddings)

    # Get top_k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [chunks_data[i] for i in top_indices]


def _qa_chunk(question: str, chunk_text: str, client: OpenAI) -> str:
    """Answer a question based on a single chunk of text."""
    try:
        response = client.chat.completions.create(
            model=QA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise QA assistant. Answer the question based solely on the provided text. "
                        "If the text does not contain relevant information to answer the question, "
                        "respond with 'No relevant information found in this section.'"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Text:\n{chunk_text}\n\nQuestion: {question}",
                },
            ],
            max_tokens=8192,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return (content or "").strip() or "[Empty response from model]"
    except Exception as e:
        return f"[QA error: {e}]"


def _aggregate_answers(results: list[dict]) -> str:
    """Format chunk QA results into a structured observation for the agent."""
    parts = []
    parts.append("## Attachment Query Results\n")

    has_relevant = False
    for r in results:
        idx = r["index"]
        summary = r["summary"]
        answer = r["answer"]

        if "No relevant information found" not in answer:
            has_relevant = True

        parts.append(f"### Section {idx + 1}")
        parts.append(f"**Summary:** {summary}")
        parts.append(f"**Answer:** {answer}\n")

    if not has_relevant:
        parts.append(
            "\n**Note:** None of the retrieved sections contained directly relevant information for this question. "
            "Consider rephrasing the question or checking other sources."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MCP Tool
# ---------------------------------------------------------------------------

@mcp.tool()
async def query_attachment(
    file_path: str, question: str, top_k: int = 5
) -> str:
    """Query the content of an attached file to answer a specific question.

    This tool converts the file to text (supporting PDF, DOCX, images, audio,
    video, and many other formats), chunks it, and uses retrieval-augmented QA
    to find and answer based on the most relevant sections. Results are cached
    for efficiency on repeated queries to the same file.

    Args:
        file_path: Absolute path to the attachment file.
        question: The specific question to answer based on the file content.
        top_k: Number of most relevant chunks to examine (default: 5).

    Returns:
        Aggregated answer based on the most relevant sections of the file.
    """
    if not OPENAI_API_KEY:
        return "[ERROR]: OPENAI_API_KEY is not set, query_attachment tool is not available."

    # Validate file exists
    if not os.path.isfile(file_path):
        return f"[ERROR]: File not found: {file_path}"

    try:
        # Get or build cache (handles concurrency via file lock)
        cache = await asyncio.to_thread(_get_or_build_cache, file_path)
    except ValueError as e:
        return f"[ERROR]: {str(e)}"
    except Exception as e:
        return f"[ERROR]: Failed to process file '{file_path}': {str(e)}"

    client = _get_client()
    chunks_data = cache["chunks"]

    # Short text optimization: single chunk → direct QA, no retrieval needed
    if len(chunks_data) == 1:
        answer = await asyncio.to_thread(
            _qa_chunk, question, chunks_data[0]["text"], client
        )
        return (
            f"## Attachment Query Results\n\n"
            f"**File:** {file_path}\n"
            f"**Summary:** {chunks_data[0]['summary']}\n"
            f"**Answer:** {answer}"
        )

    # Retrieve most relevant chunks
    try:
        relevant_chunks = await asyncio.to_thread(
            _retrieve_chunks, question, cache, client, top_k
        )
    except Exception as e:
        return f"[ERROR]: Retrieval failed: {str(e)}"

    # QA per chunk (sequentially to avoid rate limits)
    results = []
    for chunk in relevant_chunks:
        answer = await asyncio.to_thread(
            _qa_chunk, question, chunk["text"], client
        )
        results.append({
            "index": chunk["index"],
            "summary": chunk["summary"],
            "answer": answer,
        })

    return _aggregate_answers(results)


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
