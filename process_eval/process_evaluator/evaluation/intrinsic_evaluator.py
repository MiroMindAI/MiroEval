import json
import logging
from ..cache.file_cache import FileCache
from ..utils.llm_client import LLMClient
from .prompts import INTRINSIC_EVAL_PROMPT

logger = logging.getLogger(__name__)

DIMENSIONS = ["search_breadth", "analytical_depth", "progressive_refinement", "critical_thinking", "efficiency"]


class IntrinsicEvaluator:
    """Evaluate intrinsic quality of structured process."""

    def __init__(self, llm_client: LLMClient, cache: FileCache):
        self.llm = llm_client
        self.cache = cache

    def evaluate(self, entry_id, model_name: str, structured_process: dict, query: str) -> dict | None:
        cache_key = f"{model_name}_{entry_id}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = INTRINSIC_EVAL_PROMPT.format(
            query=query,
            structured_process=json.dumps(structured_process, ensure_ascii=False, indent=2),
        )
        messages = [{"role": "user", "content": prompt}]

        result = self.llm.generate_json(messages)
        if result is None:
            logger.error(f"Intrinsic evaluation failed for {cache_key}")
            return None

        # Validate all dimensions present
        for dim in DIMENSIONS:
            if dim not in result:
                logger.error(f"Missing dimension '{dim}' for {cache_key}")
                return None
            if not isinstance(result[dim], dict) or "score" not in result[dim]:
                logger.error(f"Invalid format for dimension '{dim}' in {cache_key}")
                return None

        self.cache.set(cache_key, result)
        scores_str = ", ".join(f"{d}={result[d]['score']}" for d in DIMENSIONS)
        logger.info(f"Intrinsic eval {cache_key}: {scores_str}")
        return result
