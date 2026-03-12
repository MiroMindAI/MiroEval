# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
Hybrid Multimodal QA MCP server.

Combines native file input (for images, PDFs, and other natively supported
formats) with RAG-based fallback (for DOCX, DOC, and other formats that
require text extraction). This gives the best of both worlds:
- Native: direct model understanding of images and PDFs (no information loss)
- RAG fallback: text extraction + chunking + embedding retrieval for formats
  that the native API cannot handle
"""

import asyncio
import base64
import hashlib
import json
import mimetypes
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
CHUNK_TOKEN_SIZE = int(os.environ.get("CHUNK_TOKEN_SIZE", "4096"))
QA_MODEL = os.environ.get("QA_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")

# File size limit (50 MB per file, OpenAI constraint for native input)
MAX_FILE_SIZE = 50 * 1024 * 1024

# ---------------------------------------------------------------------------
# MIME type mapping
# ---------------------------------------------------------------------------
_MIME_OVERRIDES: dict[str, str] = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pdf": "application/pdf",
    ".csv": "text/csv",
    ".tsv": "text/tsv",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".json": "application/json",
    ".xml": "application/xml",
    ".rtf": "application/rtf",
    ".odt": "application/vnd.oasis.opendocument.text",
    # Images
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".heif": "image/heif",
    ".heic": "image/heic",
}

# Formats that the OpenAI native file input API can handle directly
_NATIVE_EXTENSIONS: set[str] = {
    # Images (via image_url)
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".heif", ".heic",
    # Documents (via file input) - PDF and plain text
    ".pdf", ".txt", ".md", ".csv", ".tsv", ".html", ".htm",
    ".json", ".xml",
}

# Initialize FastMCP server
mcp = FastMCP("multimodal-qa-hybrid-mcp-server")

# Lazy-initialized OpenAI client
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


# ---------------------------------------------------------------------------
# Routing: decide native vs RAG
# ---------------------------------------------------------------------------

def _should_use_native(file_path: str) -> bool:
    """Determine whether to use native file input or RAG fallback."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in _NATIVE_EXTENSIONS


# ===================================================================
# NATIVE PATH — direct file input to model
# ===================================================================

def _get_mime_type(file_path: str) -> str:
    """Determine MIME type for a file, preferring our explicit mapping."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[ext]
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "application/octet-stream"


def _is_image(file_path: str) -> bool:
    """Check if the file is an image based on MIME type."""
    return _get_mime_type(file_path).startswith("image/")


def _encode_file(file_path: str) -> tuple[str, str]:
    """Read and base64-encode a file. Returns (base64_data_uri, filename)."""
    mime_type = _get_mime_type(file_path)
    with open(file_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64}"
    return data_uri, os.path.basename(file_path)


def _query_file_native(file_path: str, question: str) -> str:
    """Send a file + question to the model using native file input."""
    client = _get_client()
    content_parts: list[dict] = []

    if _is_image(file_path):
        data_uri, _ = _encode_file(file_path)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": data_uri},
        })
    else:
        data_uri, filename = _encode_file(file_path)
        content_parts.append({
            "type": "file",
            "file": {
                "filename": filename,
                "file_data": data_uri,
            },
        })

    content_parts.append({
        "type": "text",
        "text": question,
    })

    response = client.chat.completions.create(
        model=QA_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise QA assistant. Answer the question based solely "
                    "on the provided file content. Be thorough and specific. "
                    "If the file does not contain relevant information to answer "
                    "the question, respond with "
                    "'No relevant information found in this file.'"
                ),
            },
            {
                "role": "user",
                "content": content_parts,
            },
        ],
        max_tokens=8192,
        temperature=0.0,
    )

    result = response.choices[0].message.content
    return (result or "").strip() or "[Empty response from model]"


# ===================================================================
# RAG FALLBACK PATH — text extraction + chunking + embedding retrieval
# ===================================================================

# --- Cache helpers ---

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
        current_mtime = os.path.getmtime(file_path)
        if metadata.get("file_mtime") != current_mtime:
            return False
        if metadata.get("chunk_token_size") != CHUNK_TOKEN_SIZE:
            return False
        if metadata.get("embedding_model") != EMBEDDING_MODEL:
            return False
        return True
    except Exception:
        return False


# --- Text conversion and chunking ---

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


# --- LLM helpers ---

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
                {"role": "user", "content": chunk_text[:8000]},
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


# --- Embedding helpers ---

def _embed_texts(texts: list[str], client: OpenAI) -> np.ndarray:
    """Compute embeddings for a list of texts using OpenAI API."""
    truncated = [t[:24000] for t in texts]

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
    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    chunk_norms = chunk_embs / (
        np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9
    )
    return chunk_norms @ query_norm


# --- Cache build and load ---

def _build_cache(file_path: str) -> dict:
    """Build the full cache for a file: convert, chunk, summarize, embed."""
    client = _get_client()
    abs_path = os.path.abspath(file_path)
    cache_dir = _get_cache_dir(file_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    text = _convert_file_to_text(file_path)
    if not text:
        raise ValueError(f"File produced no extractable text: {file_path}")

    chunks = _chunk_text(text, CHUNK_TOKEN_SIZE)
    summaries = _summarize_chunks(chunks, client)

    embed_inputs = [
        f"Summary: {s}\n\nContent: {c[:4000]}" for s, c in zip(summaries, chunks)
    ]
    embeddings = _embed_texts(embed_inputs, client)

    chunks_data = [
        {"index": i, "text": c, "summary": s}
        for i, (c, s) in enumerate(zip(chunks, summaries))
    ]

    with open(cache_dir / "text_content.txt", "w", encoding="utf-8") as f:
        f.write(text)

    with open(cache_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    np.save(cache_dir / "embeddings.npy", embeddings)

    metadata = {
        "file_path": abs_path,
        "file_mtime": os.path.getmtime(file_path),
        "chunk_token_size": CHUNK_TOKEN_SIZE,
        "embedding_model": EMBEDDING_MODEL,
        "num_chunks": len(chunks),
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
    """Get cache if valid, otherwise build it. Uses file lock for concurrency."""
    cache_dir = _get_cache_dir(file_path)
    lock_path = cache_dir.parent / f"{cache_dir.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path), timeout=600):
        if _is_cache_valid(cache_dir, file_path):
            return _load_cache(cache_dir)
        return _build_cache(file_path)


# --- Retrieval and QA ---

def _retrieve_chunks(
    query: str, cache: dict, client: OpenAI, top_k: int = 5
) -> list[dict]:
    """Retrieve the most relevant chunks for a query using embedding similarity."""
    chunks_data = cache["chunks"]
    embeddings = cache["embeddings"]

    if len(chunks_data) <= top_k:
        return chunks_data

    query_emb = _embed_texts([query], client)[0]
    similarities = _cosine_similarity(query_emb, embeddings)
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


def _aggregate_answers(results: list[dict], file_path: str) -> str:
    """Format chunk QA results into a structured observation for the agent."""
    parts = []
    parts.append("## Attachment Query Results\n")
    parts.append(f"**File:** {os.path.basename(file_path)}")
    parts.append(f"**Mode:** RAG fallback (text extraction + retrieval)\n")

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


# ===================================================================
# MCP Tool
# ===================================================================

@mcp.tool()
async def query_attachment(
    file_path: str, question: str, top_k: int = 5
) -> str:
    """Query the content of an attached file to answer a specific question.

    This tool uses native file input for images, PDFs, and plain text formats
    (direct model understanding, no information loss). For other formats like
    DOCX, DOC, XLSX, PPTX, etc., it falls back to RAG-based text extraction
    with chunking and embedding retrieval. Results from the RAG path are cached
    for efficiency on repeated queries.

    Args:
        file_path: Absolute path to the attachment file.
        question: The specific question to answer based on the file content.
        top_k: Number of most relevant chunks to examine for RAG fallback (default: 5).

    Returns:
        Answer based on the file content.
    """
    if not OPENAI_API_KEY:
        return "[ERROR]: OPENAI_API_KEY is not set, query_attachment tool is not available."

    if not os.path.isfile(file_path):
        return f"[ERROR]: File not found: {file_path}"

    file_size = os.path.getsize(file_path)

    # --- Native path ---
    if _should_use_native(file_path) and file_size <= MAX_FILE_SIZE:
        try:
            answer = await asyncio.to_thread(_query_file_native, file_path, question)
            filename = os.path.basename(file_path)
            return (
                f"## Attachment Query Results\n\n"
                f"**File:** {filename}\n"
                f"**Mode:** Native file input\n"
                f"**Answer:** {answer}"
            )
        except Exception as e:
            # If native fails, fall through to RAG
            pass

    # --- RAG fallback path ---
    try:
        cache = await asyncio.to_thread(_get_or_build_cache, file_path)
    except ValueError as e:
        return f"[ERROR]: {str(e)}"
    except Exception as e:
        return f"[ERROR]: Failed to process file '{file_path}': {str(e)}"

    client = _get_client()
    chunks_data = cache["chunks"]

    # Short text optimization: single chunk → direct QA
    if len(chunks_data) == 1:
        answer = await asyncio.to_thread(
            _qa_chunk, question, chunks_data[0]["text"], client
        )
        return (
            f"## Attachment Query Results\n\n"
            f"**File:** {os.path.basename(file_path)}\n"
            f"**Mode:** RAG fallback (text extraction + retrieval)\n"
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

    # QA per chunk
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

    return _aggregate_answers(results, file_path)


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
