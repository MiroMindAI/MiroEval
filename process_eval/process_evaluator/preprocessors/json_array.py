import json
import logging
from .base import BasePreprocessor

logger = logging.getLogger(__name__)

# Step types whose content should be kept in full
KEEP_FULL_TYPES = {
    "think", "thinking", "content", "message", "agent_think",
    "synthesis", "research_plan", "response", "stage",
    "launch_research", "complete_task", "agent_start",
    "tldr", "final_report", "response_summary",
}

# Step types containing search/tool results to truncate
TRUNCATE_TYPES = {
    "search", "terminal", "tool", "web_search", "web_fetch",
    "url_extraction", "skill", "read_file", "data_files",
}

MAX_SEARCH_RESULTS = 3
MAX_SNIPPET_CHARS = 200


class JsonArrayPreprocessor(BasePreprocessor):
    """Preprocessor for JSON-array process fields (GLM, Kimi, Minimax, Claude-JSON)."""

    def __init__(self, max_chars: int = 30000, model_name: str = ""):
        super().__init__(max_chars=max_chars)
        self.model_name = model_name

    def preprocess(self, process_text: str) -> str:
        if not process_text or not process_text.strip():
            return ""

        try:
            steps = json.loads(process_text)
        except (json.JSONDecodeError, TypeError):
            logger.warning("JSON parse failed, falling back to plain text truncation")
            return self._truncate(process_text)

        if not isinstance(steps, list):
            return self._truncate(process_text)

        lines = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            step_id = step.get("step", "?")
            step_type = step.get("type", "unknown")
            line = self._format_step(step_id, step_type, step)
            if line:
                lines.append(line)

        result = "\n\n".join(lines)

        # If still over budget, adaptively compress each step
        if len(result) > self.max_chars:
            result = self._adaptive_compress(lines)

        return result

    def _adaptive_compress(self, lines: list[str]) -> str:
        """Compress steps proportionally so ALL steps fit within max_chars."""
        total_len = sum(len(l) for l in lines)
        if total_len == 0:
            return ""

        # Reserve chars for headers (~50 per step) and separators
        header_budget = len(lines) * 60
        content_budget = max(self.max_chars - header_budget, self.max_chars // 2)
        ratio = content_budget / total_len

        compressed = []
        for line in lines:
            if ratio >= 1.0:
                compressed.append(line)
                continue
            # Split into header (first line) and body
            parts = line.split("\n", 1)
            header = parts[0]
            body = parts[1] if len(parts) > 1 else ""
            # Allocate chars to body proportionally, with a minimum
            allowed = max(int(len(body) * ratio), 100)
            if len(body) > allowed:
                body = body[:allowed] + "..."
            compressed.append(f"{header}\n{body}" if body else header)

        return "\n\n".join(compressed)

    def _format_step(self, step_id, step_type: str, step: dict) -> str:
        header = f"[Step {step_id}] [{step_type}]"

        if step_type in KEEP_FULL_TYPES:
            content = self._get_content(step)
            if not content:
                # For stage type, show metadata
                if step_type == "stage":
                    name = step.get("name", "")
                    status = step.get("status", "")
                    duration = step.get("duration", "")
                    return f"{header} {name} ({status}, {duration}ms)"
                return header
            # For thinking type (minimax), prefer thinking_text
            if step_type == "thinking":
                thinking_text = step.get("thinking_text", "")
                summary = step.get("thinking_summary", "")
                plan = step.get("plan_items", [])
                parts = [header]
                if thinking_text:
                    parts.append(thinking_text)
                if summary:
                    parts.append(f"Summary: {summary}")
                if plan:
                    parts.append("Plan: " + "; ".join(plan) if isinstance(plan, list) else str(plan))
                return "\n".join(parts)
            return f"{header}\n{content}"

        if step_type in TRUNCATE_TYPES:
            return f"{header}\n{self._truncate_search_step(step)}"

        # Unknown type: keep content but truncate
        content = self._get_content(step)
        if content:
            return f"{header}\n{content[:500]}"
        return header

    def _get_content(self, step: dict) -> str:
        """Extract the main text content from a step."""
        for key in ("content", "thinking_text", "text"):
            val = step.get(key)
            if val and isinstance(val, str):
                return val
        return ""

    def _truncate_search_step(self, step: dict) -> str:
        """Extract and truncate search/tool results."""
        parts = []

        # Search queries/keywords
        for key in ("keywords", "queries", "query", "content"):
            val = step.get(key)
            if val:
                if isinstance(val, list):
                    parts.append(f"Queries: {', '.join(str(v) for v in val[:5])}")
                elif isinstance(val, str) and len(val) < 500:
                    parts.append(f"Query: {val}")
                break

        # Search results / sources
        sources = self._extract_sources(step)
        for i, src in enumerate(sources[:MAX_SEARCH_RESULTS]):
            title = src.get("title", "")
            snippet = src.get("snippet", "")[:MAX_SNIPPET_CHARS]
            parts.append(f"  [{i+1}] {title}: {snippet}")

        # Tool name (Kimi)
        if step.get("name"):
            parts.insert(0, f"Tool: {step['name']}")

        # Terminal output (GLM) - truncate heavily
        output = step.get("output", "")
        if output and isinstance(output, str) and not sources:
            parts.append(f"Output: {output[:300]}")

        return "\n".join(parts) if parts else "(search/tool step, details omitted)"

    def _extract_sources(self, step: dict) -> list[dict]:
        """Extract source objects from various field names."""
        for key in ("sources", "search_results", "organic"):
            val = step.get(key)
            if isinstance(val, list) and val:
                return val
            if isinstance(val, str):
                # Try to parse embedded JSON
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass
        return []
