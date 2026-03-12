import json
import logging
import re
from .base import BasePreprocessor

logger = logging.getLogger(__name__)

MAX_SEARCH_RESULTS = 3
MAX_SNIPPET_CHARS = 200
MAX_SCRAPE_CHARS = 500


class BlockTextPreprocessor(BasePreprocessor):
    """Preprocessor for MiroThinker's [reasoning]/[web_search]/[scrape] block format."""

    def preprocess(self, process_text: str) -> str:
        if not process_text or not process_text.strip():
            return ""

        blocks = self._split_blocks(process_text)
        lines = []

        for block_type, block_content in blocks:
            formatted = self._format_block(block_type, block_content)
            if formatted:
                lines.append(formatted)

        result = "\n\n".join(lines)

        # If still over budget, adaptively compress each block
        if len(result) > self.max_chars:
            result = self._adaptive_compress(lines)

        return result

    def _adaptive_compress(self, lines: list[str]) -> str:
        """Compress blocks proportionally so ALL blocks fit within max_chars."""
        total_len = sum(len(l) for l in lines)
        if total_len == 0:
            return ""

        header_budget = len(lines) * 40
        content_budget = max(self.max_chars - header_budget, self.max_chars // 2)
        ratio = content_budget / total_len

        compressed = []
        for line in lines:
            if ratio >= 1.0:
                compressed.append(line)
                continue
            parts = line.split("\n", 1)
            header = parts[0]
            body = parts[1] if len(parts) > 1 else ""
            allowed = max(int(len(body) * ratio), 80)
            if len(body) > allowed:
                body = body[:allowed] + "..."
            compressed.append(f"{header}\n{body}" if body else header)

        return "\n\n".join(compressed)

    def _split_blocks(self, text: str) -> list[tuple[str, str]]:
        """Split text into (block_type, content) pairs.

        Recognizes both bracket tags ([reasoning], [web_search], etc.)
        and <think>...</think> blocks (used by mirothinker_base_17).
        """
        # Match both [tag] markers and <think>...</think> blocks
        pattern = r"\[(reasoning|web_search|scrape|run_python_code)\]|<think>(.*?)</think>"
        matches = list(re.finditer(pattern, text, re.DOTALL))

        if not matches:
            return []

        blocks = []
        for idx, m in enumerate(matches):
            if m.group(1) is not None:
                # Bracket tag: [web_search], [scrape], etc.
                block_type = m.group(1)
                start = m.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                content = text[start:end].strip()
                # Strip any embedded <think>...</think> from the content
                # (they'll be captured as separate blocks)
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
                if content:
                    blocks.append((block_type, content))
            else:
                # <think>...</think> block → treat as reasoning
                think_content = m.group(2).strip()
                if think_content:
                    blocks.append(("reasoning", think_content))

        return blocks

    def _format_block(self, block_type: str, content: str) -> str:
        if block_type == "reasoning":
            # Strip <think> tags
            cleaned = re.sub(r"</?think>", "", content).strip()
            if not cleaned:
                return ""
            return f"[Reasoning]\n{cleaned}"

        if block_type == "web_search":
            return self._format_search(content)

        if block_type == "scrape":
            return self._format_scrape(content)

        if block_type == "run_python_code":
            return f"[Code Execution]\n{content[:500]}"

        return f"[{block_type}]\n{content[:500]}"

    def _format_search(self, content: str) -> str:
        """Format web search block: keep query + top-N results."""
        # First line is usually the query, rest is JSON
        lines = content.split("\n", 1)
        query = lines[0].strip()

        results = []
        if len(lines) > 1:
            try:
                data = json.loads(lines[1])
                organic = data.get("organic", []) if isinstance(data, dict) else []
                for item in organic[:MAX_SEARCH_RESULTS]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")[:MAX_SNIPPET_CHARS]
                    results.append(f"  - {title}: {snippet}")
            except (json.JSONDecodeError, TypeError):
                pass

        parts = [f"[Search] {query}"]
        parts.extend(results)
        return "\n".join(parts)

    def _format_scrape(self, content: str) -> str:
        """Format scrape block: keep URL + truncated content."""
        lines = content.split("\n", 1)
        url = lines[0].strip()
        body = lines[1][:MAX_SCRAPE_CHARS] if len(lines) > 1 else ""
        return f"[Scrape] {url}\n{body}"
