# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
Native Multimodal QA MCP server.

Uses OpenAI's native file input support to directly send files (PDF, DOCX,
XLSX, PPTX, images, etc.) to the model without chunking or embedding.
This is a simpler alternative to the RAG-based multimodal_qa_mcp_server.
"""

import asyncio
import base64
import mimetypes
import os
from typing import Optional

import dotenv
from fastmcp import FastMCP
from openai import OpenAI

# Load .env so this subprocess picks up API keys / base URLs
dotenv.load_dotenv()

# Environment variables (set via tool config YAML)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
QA_MODEL = os.environ.get("QA_MODEL", "gpt-5-mini")

# File size limit (50 MB per file, OpenAI constraint)
MAX_FILE_SIZE = 50 * 1024 * 1024

# MIME type mapping for common attachment formats
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

# Initialize FastMCP server
mcp = FastMCP("multimodal-qa-native-mcp-server")

# Lazy-initialized OpenAI client
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


def _get_mime_type(file_path: str) -> str:
    """Determine MIME type for a file, preferring our explicit mapping."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[ext]
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "application/octet-stream"


def _encode_file(file_path: str) -> tuple[str, str]:
    """Read and base64-encode a file. Returns (base64_data_uri, filename)."""
    mime_type = _get_mime_type(file_path)
    with open(file_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64}"
    return data_uri, os.path.basename(file_path)


def _is_image(file_path: str) -> bool:
    """Check if the file is an image based on MIME type."""
    return _get_mime_type(file_path).startswith("image/")


def _query_file_sync(file_path: str, question: str) -> str:
    """Send a file + question to the model using native file input."""
    client = _get_client()

    # Build the user message content parts
    content_parts: list[dict] = []

    if _is_image(file_path):
        # Use image_url content type for images
        data_uri, _ = _encode_file(file_path)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": data_uri},
        })
    else:
        # Use file content type for documents
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


@mcp.tool()
async def query_attachment(
    file_path: str, question: str
) -> str:
    """Query the content of an attached file to answer a specific question.

    This tool sends the file directly to the model using OpenAI's native file
    input support. It handles PDF, DOCX, XLSX, PPTX, CSV, images, and many
    other formats without needing to convert them to text first.

    Args:
        file_path: Absolute path to the attachment file.
        question: The specific question to answer based on the file content.

    Returns:
        Answer based on the file content.
    """
    if not OPENAI_API_KEY:
        return "[ERROR]: OPENAI_API_KEY is not set, query_attachment tool is not available."

    if not os.path.isfile(file_path):
        return f"[ERROR]: File not found: {file_path}"

    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        return (
            f"[ERROR]: File too large ({file_size / 1024 / 1024:.1f} MB). "
            f"Maximum supported size is {MAX_FILE_SIZE / 1024 / 1024:.0f} MB."
        )

    try:
        answer = await asyncio.to_thread(_query_file_sync, file_path, question)
    except Exception as e:
        return f"[ERROR]: Failed to query file '{file_path}': {e}"

    filename = os.path.basename(file_path)
    return (
        f"## Attachment Query Results\n\n"
        f"**File:** {filename}\n"
        f"**Answer:** {answer}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
