STRUCTURING_PROMPT = """\
You are an expert analyst tasked with converting a deep research agent's raw process trace into a structured representation.

## Input

**User Query:**
{query}

**Raw Process Trace:**
{process_text}

## Task

Analyze this process trace and extract a structured representation. The trace may come from different AI research agents and can be in various formats (step-by-step logs, thinking traces, search results, narrative descriptions, etc.).

For each identifiable step in the process, determine:
1. **action_type**: one of: `plan`, `search`, `read`, `analyze`, `synthesize`, `verify`
   - `plan`: Setting research strategy, decomposing the task, identifying what to investigate
   - `search`: Issuing search queries, browsing for information sources
   - `read`: Reading/scraping specific documents, extracting information from sources
   - `analyze`: Deep analysis of gathered information, reasoning about findings
   - `synthesize`: Combining information from multiple sources, drafting conclusions
   - `verify`: Cross-checking facts, validating claims, identifying contradictions
2. **summary**: A concise 1-2 sentence description of what this step accomplished
3. **key_findings**: Specific factual claims or insights that were **newly discovered in this step**. Do NOT repeat findings already listed in a previous step. If a step merely confirms or reuses earlier information, leave key_findings empty or only note genuinely new information. Failed attempts, dead ends, and resource constraints encountered in this step should also be recorded here (e.g., "S&P page failed to load", "abandoned T-Mobile spread search due to search budget limit", "used index OAS as approximation because direct data was paywalled").

After analyzing all steps, identify the **global_findings**: the most important conclusions that emerge from synthesizing information **across multiple steps**. Global findings should NOT duplicate individual step-level key_findings — they should represent higher-level insights, cross-step conclusions, or cumulative understanding that no single step produced alone.

## Output Format

Return a JSON object with exactly this structure:
```json
{{
  "steps": [
    {{
      "step_id": 1,
      "action_type": "plan",
      "summary": "...",
      "key_findings": ["...", "..."]
    }}
  ],
  "global_findings": [
    {{
      "finding": "A cross-step conclusion or synthesized insight",
      "first_found_at_step": 3,
      "related_steps": [1, 5, 8],
      "evidence_strength": "strong"
    }}
  ]
}}
```

**step_id**: Use sequential numbering starting from 1. Do NOT copy step IDs from the original trace.

**related_steps**: All steps that contributed to this finding — including earlier steps it depends on and later steps where it was refined or confirmed. This is NOT directional; it simply lists all relevant steps.

For `evidence_strength`, use:
- `strong`: Confirmed by multiple sources or authoritative references
- `moderate`: Supported by at least one credible source
- `weak`: Mentioned but not well-supported, or inferred

## Guidelines
- Merge very small consecutive steps of the same type into one step
- Do NOT invent findings not present in the trace
- For search steps, the findings should reflect what was actually found, not just what was searched for
- Keep global_findings focused on the most substantive discoveries (aim for 5-15 findings)
- If the process trace is very short or uninformative, still extract what you can but have fewer steps/findings
- Pay special attention to: failed searches, abandoned research paths, forced approximations due to resource limits (e.g., search budget, paywalls), and strategy changes caused by constraints. These are valuable findings.
"""
