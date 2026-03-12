import json
import logging
import os
import re
import time

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClient:
    """OpenRouter / OpenAI compatible LLM client."""

    def __init__(
        self,
        model: str = "openai/gpt-5-mini",
        api_type: str = "openrouter",
        max_tokens: int = 8192,
        temperature: float = 0.1,
        retry_count: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.model = model
        self.api_type = api_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry_count = retry_count
        self.retry_backoff = retry_backoff
        self.total_cost = 0.0

        if api_type == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL)
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = os.environ.get("OPENAI_BASE_URL", "")
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)

    def generate(self, messages: list[dict], max_tokens: int | None = None, temperature: float | None = None) -> str:
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        for attempt in range(self.retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content or ""
                if response.usage:
                    # Rough cost tracking for openrouter
                    self.total_cost += (response.usage.prompt_tokens * 1.25 + response.usage.completion_tokens * 10) / 1_000_000
                    # Warn if reasoning consumed all tokens
                    if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                        reasoning = getattr(response.usage.completion_tokens_details, 'reasoning_tokens', 0) or 0
                        if reasoning > 0 and not content.strip():
                            logger.warning(f"Reasoning model used {reasoning} tokens for thinking but produced empty content. "
                                           f"Consider increasing max_tokens (currently {max_tokens}).")
                return content

            except Exception as e:
                wait = self.retry_backoff ** attempt
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.retry_count}): {e}. Retrying in {wait}s")
                time.sleep(wait)

        logger.error(f"LLM call failed after {self.retry_count} attempts")
        return "$ERROR$"

    def generate_json(self, messages: list[dict], **kwargs) -> dict | None:
        raw = self.generate(messages, **kwargs)
        if raw == "$ERROR$":
            return None
        return extract_json(raw)


def extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response, handling code fences and tags."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try code fence
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try <json_output> tags
    match = re.search(r"<json_output>\s*(.*?)\s*</json_output>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } or [ ... ] block
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    logger.warning(f"Failed to extract JSON from response (len={len(text)})")
    return None
