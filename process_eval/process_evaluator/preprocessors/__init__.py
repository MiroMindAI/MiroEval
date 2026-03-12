import json
import re

from .base import BasePreprocessor
from .json_array import JsonArrayPreprocessor
from .block_text import BlockTextPreprocessor
from .step_text import StepTextPreprocessor
from .plain_text import PlainTextPreprocessor


class AutoDetectPreprocessor(BasePreprocessor):
    """Auto-detect process format and delegate to the appropriate preprocessor.

    Supports any process string — no model name needed.
    Detection priority:
      1. JSON array of step objects (e.g. GLM, Kimi, Minimax, some Claude)
      2. Block-tagged text with [reasoning]/[web_search]/[scrape] (e.g. MiroThinker)
      3. Step-tagged text with [Step N] [Tag] (e.g. Gemini)
      4. Plain text fallback (ChatGPT, Qwen, Grok, etc.)
    """

    def __init__(self, max_chars: int = 30000, **kwargs):
        super().__init__(max_chars=max_chars)
        self._json_pp = JsonArrayPreprocessor(max_chars=max_chars)
        self._block_pp = BlockTextPreprocessor(max_chars=max_chars)
        self._step_pp = StepTextPreprocessor(max_chars=max_chars)
        self._plain_pp = PlainTextPreprocessor(max_chars=max_chars)

    def preprocess(self, process_text: str) -> str:
        if not process_text or not process_text.strip():
            return ""

        fmt = self.detect_format(process_text)

        if fmt == "json_array":
            return self._json_pp.preprocess(process_text)
        elif fmt == "block_text":
            return self._block_pp.preprocess(process_text)
        elif fmt == "step_text":
            return self._step_pp.preprocess(process_text)
        else:
            return self._plain_pp.preprocess(process_text)

    @staticmethod
    def detect_format(process_text: str) -> str:
        """Detect the format of a process string.

        Returns one of: 'json_array', 'block_text', 'step_text', 'plain_text'
        """
        stripped = process_text.strip()

        # 1. Try JSON array: starts with [ and parses as a list of dicts with "step"/"type" keys
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if (
                    isinstance(parsed, list)
                    and len(parsed) > 0
                    and isinstance(parsed[0], dict)
                    and ("step" in parsed[0] or "type" in parsed[0])
                ):
                    return "json_array"
            except (json.JSONDecodeError, TypeError):
                pass

        # 2. Block-tagged: has [reasoning] or [web_search] markers
        block_tags = re.findall(r"\[(reasoning|web_search|scrape|run_python_code)\]", stripped)
        if len(block_tags) >= 2:
            return "block_text"

        # 3. Step-tagged: has [Step N] markers
        step_tags = re.findall(r"\[Step \d+\]", stripped)
        if len(step_tags) >= 2:
            return "step_text"

        # 4. Fallback: plain text
        return "plain_text"


def get_preprocessor(model_name: str | None = None, max_chars: int = 30000, **config) -> BasePreprocessor:
    """Get a preprocessor. Always returns AutoDetectPreprocessor (model_name is ignored)."""
    return AutoDetectPreprocessor(max_chars=max_chars, **config)


__all__ = [
    "BasePreprocessor", "get_preprocessor",
    "AutoDetectPreprocessor",
    "JsonArrayPreprocessor", "BlockTextPreprocessor",
    "StepTextPreprocessor", "PlainTextPreprocessor",
]
