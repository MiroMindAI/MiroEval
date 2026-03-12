import re
from .base import BasePreprocessor


class PlainTextPreprocessor(BasePreprocessor):
    """Preprocessor for unstructured text processes (ChatGPT, Qwen, Grok, Claude-text)."""

    def preprocess(self, process_text: str) -> str:
        if not process_text or not process_text.strip():
            return ""

        # Normalize whitespace: collapse 3+ newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", process_text)
        # Strip trailing whitespace per line
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        return self._truncate(text)
