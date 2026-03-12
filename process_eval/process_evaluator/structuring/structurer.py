import logging
from ..cache.file_cache import FileCache
from ..utils.llm_client import LLMClient
from .prompts import STRUCTURING_PROMPT

logger = logging.getLogger(__name__)


class ProcessStructurer:
    """LLM-based process structuring: convert preprocessed text to unified schema."""

    def __init__(self, llm_client: LLMClient, cache: FileCache):
        self.llm = llm_client
        self.cache = cache

    def structure(self, entry_id, model_name: str, preprocessed_text: str, query: str) -> dict | None:
        cache_key = f"{model_name}_{entry_id}"

        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        if not preprocessed_text.strip():
            logger.warning(f"Empty process for {cache_key}, skipping")
            return None

        prompt = STRUCTURING_PROMPT.format(
            query=query,
            process_text=preprocessed_text,
        )
        messages = [{"role": "user", "content": prompt}]

        result = self.llm.generate_json(messages)
        if result is None:
            logger.error(f"Structuring failed for {cache_key}")
            return None

        # Validate basic structure
        if "steps" not in result or "global_findings" not in result:
            logger.error(f"Invalid structure for {cache_key}: missing required keys")
            return None

        self.cache.set(cache_key, result)
        logger.info(f"Structured {cache_key}: {len(result['steps'])} steps, {len(result['global_findings'])} findings")
        return result
