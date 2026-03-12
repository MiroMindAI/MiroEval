from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """Base class for process text preprocessors."""

    def __init__(self, max_chars: int = 30000):
        self.max_chars = max_chars

    @abstractmethod
    def preprocess(self, process_text: str) -> str:
        """Clean and truncate process text for LLM consumption."""
        ...

    def _truncate(self, text: str) -> str:
        """Truncate text to max_chars at a paragraph boundary."""
        if len(text) <= self.max_chars:
            return text

        # Find the last paragraph break before max_chars
        cut = text[:self.max_chars]
        last_break = cut.rfind("\n\n")
        if last_break > self.max_chars * 0.7:
            cut = cut[:last_break]

        return cut + "\n\n[... truncated ...]"
