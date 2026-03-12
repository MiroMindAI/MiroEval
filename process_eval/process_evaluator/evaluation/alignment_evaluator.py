import json
import logging
from ..cache.file_cache import FileCache
from ..utils.llm_client import LLMClient
from .prompts import ALIGNMENT_EVAL_PROMPT

logger = logging.getLogger(__name__)

DIMENSIONS = ["findings_to_report", "report_to_process", "contradiction"]
REPORT_MAX_CHARS = 30000


class AlignmentEvaluator:
    """Evaluate alignment between process findings and final report."""

    def __init__(self, llm_client: LLMClient, cache: FileCache, report_max_chars: int = REPORT_MAX_CHARS):
        self.llm = llm_client
        self.cache = cache
        self.report_max_chars = report_max_chars

    def evaluate(
        self, entry_id, model_name: str, structured_process: dict, report: str, query: str
    ) -> dict | None:
        cache_key = f"{model_name}_{entry_id}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Extract global findings for the prompt
        global_findings = structured_process.get("global_findings", [])
        findings_text = json.dumps(global_findings, ensure_ascii=False, indent=2)

        # Truncate report
        truncated_report = report[:self.report_max_chars]
        if len(report) > self.report_max_chars:
            truncated_report += "\n\n[... report truncated ...]"

        prompt = ALIGNMENT_EVAL_PROMPT.format(
            query=query,
            global_findings=findings_text,
            report=truncated_report,
        )
        messages = [{"role": "user", "content": prompt}]

        result = self.llm.generate_json(messages)
        if result is None:
            logger.error(f"Alignment evaluation failed for {cache_key}")
            return None

        for dim in DIMENSIONS:
            if dim not in result:
                logger.error(f"Missing dimension '{dim}' for {cache_key}")
                return None
            if not isinstance(result[dim], dict) or "score" not in result[dim]:
                logger.error(f"Invalid format for dimension '{dim}' in {cache_key}")
                return None

        self.cache.set(cache_key, result)
        scores_str = ", ".join(f"{d}={result[d]['score']}" for d in DIMENSIONS)
        logger.info(f"Alignment eval {cache_key}: {scores_str}")
        return result
