# SPDX-FileCopyrightText: 2025 MiromindAI
#
# SPDX-License-Identifier: Apache-2.0

"""
Report segment processor - segments reports into logical parts using LLM
"""

import asyncio
import json

from miroflow.io_processor.base import BaseIOProcessor
from miroflow.agents.context import AgentContext
from miroflow.registry import register, ComponentType
from miroflow.logging.task_tracer import get_tracer

logger = get_tracer()


@register(ComponentType.IO_PROCESSOR, "ReportSegmentProcessor")
class ReportSegmentProcessor(BaseIOProcessor):
    """Segments a report into logical parts using LLM-based semantic segmentation."""

    USE_PROPAGATE_MODULE_CONFIGS = ("llm", "prompt")

    async def run_internal(self, ctx: AgentContext) -> AgentContext:
        report_text = ctx.get("task_description", "")

        system_prompt = self.prompt_manager.render_prompt(
            "segment_system_prompt", context={}
        )
        user_prompt = self.prompt_manager.render_prompt(
            "segment_user_prompt", context={"report_text": report_text}
        )

        max_retries = 5
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.llm_client.create_message(
                    system_prompt=system_prompt,
                    message_history=[
                        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
                    ],
                    tool_definitions=[],
                )

                segments = json.loads(response.response_text.strip())
                if isinstance(segments, list) and all(
                    isinstance(s, str) for s in segments
                ):
                    logger.info(
                        f"Successfully segmented report into {len(segments)} parts"
                    )
                    return AgentContext(segments=segments)
                else:
                    raise ValueError("LLM did not return a valid list of strings")
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Segment report attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)

        raise RuntimeError(
            f"Failed to segment report after {max_retries} attempts: {last_error}"
        )
