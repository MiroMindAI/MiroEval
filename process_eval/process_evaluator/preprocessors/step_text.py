import re
from .base import BasePreprocessor


class StepTextPreprocessor(BasePreprocessor):
    """Preprocessor for Gemini's [Step N] [Tag] text format."""

    def preprocess(self, process_text: str) -> str:
        if not process_text or not process_text.strip():
            return ""

        # Split on [Step N] boundaries
        parts = re.split(r"(\[Step \d+\])", process_text)

        steps = []
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if re.match(r"\[Step \d+\]", part):
                content = parts[i + 1] if i + 1 < len(parts) else ""
                steps.append(f"{part} {content.strip()}")
                i += 2
            else:
                if part:
                    steps.append(part)
                i += 1

        result = "\n\n".join(steps)
        return self._truncate(result)
