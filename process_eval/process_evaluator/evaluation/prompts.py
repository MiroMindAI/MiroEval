INTRINSIC_EVAL_PROMPT = """\
You are an expert evaluator assessing the quality of a deep research agent's investigation process.

## Input

**User Query:**
{query}

**Structured Process:**
{structured_process}

## Task

Evaluate the research process on the following 5 dimensions. Score each from 1-10 with a brief justification.

### Dimensions

1. **search_breadth** (1-10): Did the agent explore diverse information sources and angles? Did it search for multiple aspects of the query rather than just one narrow perspective?
   - 1-3: Minimal searching, only 1-2 queries on the same narrow topic
   - 4-6: Some diversity but missed important angles
   - 7-10: Comprehensive search covering multiple relevant dimensions

2. **analytical_depth** (1-10): Did the agent analyze information deeply rather than just collecting surface-level facts? Did it interpret, compare, and draw insights from the gathered data?
   - 1-3: Merely listed facts without analysis
   - 4-6: Some analysis but mostly surface-level
   - 7-10: Deep analysis with meaningful insights and interpretations

3. **progressive_refinement** (1-10): Did the agent iteratively deepen its understanding? Did later steps build on earlier findings? Did the research trajectory show increasing sophistication?
   - 1-3: Flat trajectory, no visible learning across steps
   - 4-6: Some progression but largely parallel/independent steps
   - 7-10: Clear iterative deepening with each stage building on previous discoveries

4. **critical_thinking** (1-10): Did the agent question assumptions, cross-check information, identify contradictions, or acknowledge limitations?
   - 1-3: No evidence of critical evaluation
   - 4-6: Occasional questioning but mostly accepted information at face value
   - 7-10: Actively verified claims, noted discrepancies, and evaluated source reliability

5. **efficiency** (1-10): Was the process efficient? Did each step contribute meaningfully, or were there redundant searches, circular reasoning, or wasted effort?
   - 1-3: Highly redundant, many wasted steps
   - 4-6: Some redundancy but generally productive
   - 7-10: Lean process where almost every step contributed value

## Output Format

Return a JSON object:
```json
{{
  "search_breadth": {{"score": 8, "justification": "..."}},
  "analytical_depth": {{"score": 7, "justification": "..."}},
  "progressive_refinement": {{"score": 6, "justification": "..."}},
  "critical_thinking": {{"score": 5, "justification": "..."}},
  "efficiency": {{"score": 7, "justification": "..."}}
}}
```
"""


ALIGNMENT_EVAL_PROMPT = """\
You are an expert evaluator assessing the alignment between a research agent's investigation process and its final report.

## Input

**User Query:**
{query}

**Key Findings from Process:**
{global_findings}

**Final Report (may be truncated):**
{report}

## Task

Evaluate the alignment between the process findings and the final report on 3 dimensions. Score each from 1-10 with a brief justification.

### Dimensions

1. **findings_to_report** (1-10): What fraction of the key findings discovered during the process actually appear in the final report?
   - 1-3: Most process findings are absent from the report
   - 4-6: Some findings appear but many important ones are missing
   - 7-10: Nearly all process findings are incorporated into the report

2. **report_to_process** (1-10): Can the major claims and conclusions in the report be traced back to findings in the process? Or does the report contain substantial content that has no basis in the documented process?
   - 1-3: Report contains many claims with no process basis (possible hallucination)
   - 4-6: Most report claims have some process basis but some appear unsupported
   - 7-10: Nearly all report content is traceable to process findings

3. **contradiction** (1-10): Are the process findings and report conclusions consistent with each other? (10 = perfectly consistent, no contradictions)
   - 1-3: Major contradictions between process and report
   - 4-6: Minor inconsistencies or selective presentation
   - 7-10: Process and report are fully consistent

## Output Format

Return a JSON object:
```json
{{
  "findings_to_report": {{"score": 8, "justification": "..."}},
  "report_to_process": {{"score": 7, "justification": "..."}},
  "contradiction": {{"score": 9, "justification": "..."}}
}}
```
"""
