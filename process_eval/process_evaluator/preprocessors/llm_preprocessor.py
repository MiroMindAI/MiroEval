"""LLM-based preprocessor: uses a small/cheap LLM to compress process traces.

Format-agnostic — works with any process string regardless of structure.
Handles long processes by chunking, compressing each chunk independently,
then concatenating the results.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BasePreprocessor

logger = logging.getLogger(__name__)

# Each chunk sent to LLM should be under this many chars
# (conservative to fit in context with prompt overhead)
CHUNK_INPUT_CHARS = 40000

COMPRESS_PROMPT = """\
You are a research process trace compressor. Your job is to compress a raw process trace while preserving all information needed to evaluate the research quality.

**User Query (for context):**
{query}

**This is chunk {chunk_idx} of {total_chunks} from the full process trace.**

**Raw Process Trace Chunk:**
{chunk_text}

## What to PRESERVE — keep these faithfully, as close to the original as possible:
1. **Every search/retrieval query** the agent issued (preserve the exact keywords or questions)
2. **All reasoning, analysis, and interpretation** by the agent — this is the most important content. Keep the agent's own thinking, arguments, and conclusions in detail.
3. **Strategy decisions** — why the agent chose a particular search direction, why it changed course, how it prioritized
4. **Key factual findings and data points** the agent discovered and used
5. **Failed attempts, dead ends, errors** — searches that returned nothing useful, pages that failed to load, approaches that were abandoned and why
6. **Cross-checking and verification** — any time the agent compared sources, noted contradictions, or validated claims
7. **The chronological order** of all actions

## What to REMOVE — these add bulk but not evaluative value:
1. **Full text of retrieved web pages/documents** — only keep what the agent actually extracted or referenced from them
2. **Duplicate information** — if the same fact appears in multiple search results, keep it once
3. **Raw HTML, JSON metadata, URL lists** — unless the agent specifically discussed a URL
4. **Formatting markers, XML/JSON tags, structural metadata** (e.g., `<think>`, `[web_search]`, step type fields)
5. **Verbose tool call parameters and raw API responses**
6. **Boilerplate text** from search result snippets that the agent did not engage with

## Output:
Write the compressed trace as concise prose, preserving the chronological flow.
Do NOT add any analysis, commentary, or evaluation — just compress faithfully.
Do NOT impose a target length — let the actual information content determine the output length. A chunk with dense reasoning should produce longer output than a chunk that is mostly search result boilerplate.
"""


class LLMPreprocessor(BasePreprocessor):
    """Preprocessor that uses an LLM to compress process traces.

    Format-agnostic: handles any process string by chunking and compressing.
    """

    def __init__(self, llm_client, max_chars: int = 30000, query: str = "",
                 chunk_workers: int = 5):
        super().__init__(max_chars=max_chars)
        self.llm = llm_client
        self.query = query
        self.chunk_workers = chunk_workers

    def preprocess(self, process_text: str) -> str:
        if not process_text or not process_text.strip():
            return ""

        # If short enough, no need to chunk — just compress once
        if len(process_text) <= CHUNK_INPUT_CHARS:
            return self._compress_chunk(process_text, 1, 1)

        # Split into chunks
        chunks = self._split_into_chunks(process_text)
        logger.info(f"LLM preprocessor: split into {len(chunks)} chunks "
                    f"(total {len(process_text):,} chars)")

        # Compress each chunk in parallel
        if len(chunks) == 1:
            return self._compress_chunk(chunks[0], 1, 1)

        compressed_chunks = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=self.chunk_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._compress_chunk, chunk, idx + 1, len(chunks)
                ): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    compressed_chunks[idx] = future.result()
                except Exception as e:
                    logger.error(f"Chunk {idx + 1} compression failed: {e}")
                    # Fallback: hard truncate this chunk
                    compressed_chunks[idx] = chunks[idx][:CHUNK_INPUT_CHARS // 5]

        result = "\n\n---\n\n".join(c for c in compressed_chunks if c)

        # Final safety truncation (shouldn't normally trigger)
        if len(result) > self.max_chars:
            result = result[:self.max_chars] + "\n\n[... truncated ...]"

        return result

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split process text into chunks at natural boundaries."""
        # Try to split at structural boundaries first
        chunks = []

        # For JSON arrays, split by step objects
        stripped = text.strip()
        if stripped.startswith("["):
            try:
                steps = json.loads(stripped)
                if isinstance(steps, list) and steps:
                    return self._chunk_json_steps(steps)
            except (json.JSONDecodeError, TypeError):
                pass

        # For block-tagged text, split at block boundaries
        block_pattern = r"\[(reasoning|web_search|scrape|run_python_code)\]"
        block_matches = list(re.finditer(block_pattern, text))
        if len(block_matches) >= 2:
            return self._chunk_at_markers(text, block_matches)

        # For step-tagged text, split at step boundaries
        step_matches = list(re.finditer(r"\[Step \d+\]", text))
        if len(step_matches) >= 2:
            return self._chunk_at_markers(text, step_matches)

        # Plain text: split at paragraph boundaries
        return self._chunk_plain_text(text)

    def _chunk_json_steps(self, steps: list) -> list[str]:
        """Group JSON step objects into chunks under CHUNK_INPUT_CHARS."""
        chunks = []
        current_steps = []
        current_len = 0

        for step in steps:
            step_str = json.dumps(step, ensure_ascii=False)
            if current_len + len(step_str) > CHUNK_INPUT_CHARS and current_steps:
                chunks.append(json.dumps(current_steps, ensure_ascii=False, indent=1))
                current_steps = []
                current_len = 0
            current_steps.append(step)
            current_len += len(step_str)

        if current_steps:
            chunks.append(json.dumps(current_steps, ensure_ascii=False, indent=1))

        return chunks

    def _chunk_at_markers(self, text: str, matches: list) -> list[str]:
        """Split text into chunks at marker boundaries."""
        # Get all segment boundaries
        boundaries = [m.start() for m in matches] + [len(text)]

        chunks = []
        chunk_start = 0
        current_len = 0

        for i in range(len(boundaries) - 1):
            seg_len = boundaries[i + 1] - boundaries[i]
            if current_len + seg_len > CHUNK_INPUT_CHARS and current_len > 0:
                chunks.append(text[chunk_start:boundaries[i]])
                chunk_start = boundaries[i]
                current_len = 0
            current_len += seg_len

        if chunk_start < len(text):
            chunks.append(text[chunk_start:])

        return chunks if chunks else [text]

    def _chunk_plain_text(self, text: str) -> list[str]:
        """Split plain text into chunks at paragraph boundaries."""
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_parts = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > CHUNK_INPUT_CHARS and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            current_parts.append(para)
            current_len += len(para)

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return chunks if chunks else [text]

    def _compress_chunk(self, chunk_text: str, chunk_idx: int, total_chunks: int) -> str:
        """Compress a single chunk using the LLM."""
        prompt = COMPRESS_PROMPT.format(
            query=self.query,
            chunk_idx=chunk_idx,
            total_chunks=total_chunks,
            chunk_text=chunk_text,
        )

        messages = [{"role": "user", "content": prompt}]
        result = self.llm.generate(messages)

        if result == "$ERROR$" or not result or not result.strip():
            logger.error(f"LLM compression failed for chunk {chunk_idx}/{total_chunks} "
                         f"(empty or error response)")
            return chunk_text[:8000]

        return result
