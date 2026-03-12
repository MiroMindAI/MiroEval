import json
import logging
from typing import Dict, List, Any, Optional
from statistics import mean
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class PointwiseEvaluatorCore:
    """Core methods for pointwise evaluation"""

    def extract_key_facts(self, query_id: int) -> List[Dict[str, str]]:
        """
        Extract key facts from query attachments using LLM.

        If the query has multiple attachment files (stored in 'attachment_parts'),
        each file is processed independently and the results are merged.
        This ensures focused extraction per file rather than a single pass
        over concatenated content.
        """
        cache_key = f"key_facts_{query_id}"
        cached_result = self.cache_manager.get("key_facts", cache_key)

        if cached_result is not None:
            logger.info(f"Using cached key facts for query {query_id}")
            return cached_result

        query_data = self.queries[query_id]
        task_prompt = query_data['prompt']

        # Check for per-file attachment parts (preferred) or single combined attachment
        attachment_parts = query_data.get('attachment_parts', [])
        if not attachment_parts:
            # Fallback: use combined attachment as a single part
            attachment_content = query_data.get('attachment', '')
            if attachment_content and attachment_content.strip():
                attachment_parts = [attachment_content]

        if not attachment_parts:
            return []

        # Extract key facts from each file independently
        all_key_facts = []
        for i, part_content in enumerate(attachment_parts):
            if not part_content or not part_content.strip():
                continue

            logger.info(f"Extracting key facts from attachment {i+1}/{len(attachment_parts)} for query {query_id}")

            formatted_prompt = self.key_facts_extraction_prompt.format(
                task_prompt=task_prompt,
                attachment_content=part_content
            )

            messages = [{"role": "user", "content": formatted_prompt}]
            response = self.generate_llm_response(messages, max_tokens=8192, temperature=0.1)

            try:
                facts_json = self.extract_json_from_response(response)
                if facts_json:
                    facts = json.loads(facts_json)
                    all_key_facts.extend(facts)
                    logger.info(f"Extracted {len(facts)} key facts from attachment {i+1} for query {query_id}")
                else:
                    logger.warning(f"Could not extract key facts JSON from attachment {i+1} for query {query_id}")
            except Exception as e:
                logger.error(f"Failed to parse key facts from attachment {i+1} for query {query_id}: {e}")

        logger.info(f"Total key facts for query {query_id}: {len(all_key_facts)} (from {len(attachment_parts)} file(s))")

        self.cache_manager.set("key_facts", cache_key, all_key_facts)
        return all_key_facts

    def generate_query_dimensions(self, query_id: int, key_facts: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Generate query-specific evaluation dimensions"""
        cache_key = f"dimensions_{query_id}"
        cached_result = self.cache_manager.get("dimensions", cache_key)
        
        if cached_result is not None:
            logger.info(f"Using cached dimensions for query {query_id}")
            return cached_result
        
        query_data = self.queries[query_id]
        task_prompt = query_data['prompt']

        logger.info(f"Generating query-specific dimensions for query {query_id}")

        # Use attachment-aware prompt if key facts are available
        if key_facts:
            key_facts_json = json.dumps(key_facts, ensure_ascii=False, indent=2)
            formatted_prompt = self.dimension_generation_with_attachment_prompt.format(
                task_prompt=task_prompt,
                key_facts_json=key_facts_json
            )
        else:
            formatted_prompt = self.dimension_generation_prompt.format(
                task_prompt=task_prompt
            )
        
        messages = [{"role": "user", "content": formatted_prompt}]
        response = self.generate_llm_response(messages, max_tokens=8192, temperature=0.1)

        try:
            dimensions_json = self.extract_json_from_response(response)
            if dimensions_json:
                dimensions = json.loads(dimensions_json)
                logger.info(f"Generated {len(dimensions)} dimensions for query {query_id}")
            else:
                logger.warning(f"Could not extract JSON from response for query {query_id}")
                dimensions = []
        except Exception as e:
            logger.error(f"Failed to parse dimensions for query {query_id}: {e}")
            dimensions = []
        
        # Cache and return results
        self.cache_manager.set("dimensions", cache_key, dimensions)
        return dimensions
    
    def generate_hierarchical_weights(self, query_id: int, additional_dimensions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Generate hierarchical weights for all dimensions"""
        cache_key = f"weights_{query_id}_{len(additional_dimensions)}"
        cached_result = self.cache_manager.get("weights", cache_key)
        
        if cached_result is not None:
            logger.info(f"Using cached weights for query {query_id}")
            return cached_result
        
        query_data = self.queries[query_id]
        task_prompt = query_data['prompt']
        
        logger.info(f"Generating hierarchical weights for query {query_id}")
        
        additional_dimensions_json = json.dumps(additional_dimensions, ensure_ascii=False, indent=2)
        
        formatted_prompt = self.weight_generation_prompt.format(
            task_prompt=task_prompt,
            additional_dimensions_json=additional_dimensions_json
        )
        
        messages = [{"role": "user", "content": formatted_prompt}]
        response = self.generate_llm_response(messages, max_tokens=8192, temperature=0.1)
        # Extract weights from response
        try:
            weights_json = self.extract_json_from_analysis_output(response)
            if weights_json:
                weights = json.loads(weights_json)
                # Normalize weights to sum to 1.0
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v/total_weight for k, v in weights.items()}
                
                # Convert dimension names to lowercase with underscores
                new_weights = {}
                for dim in weights:
                    dim_name = dim.lower().replace(' ', '_').replace('-', '_')
                    new_weights[dim_name] = weights[dim]
                weights = new_weights
                
                logger.info(f"Generated weights for query {query_id}: {weights}")
            else:
                logger.warning(f"Could not extract JSON from weights response for query {query_id}")
                weights = self._get_default_weights(additional_dimensions)
        except Exception as e:
            logger.error(f"Failed to parse weights for query {query_id}: {e}")
            weights = self._get_default_weights(additional_dimensions)

        # Cache and return results
        self.cache_manager.set("weights", cache_key, weights)
        return new_weights
    
    def _get_default_weights(self, additional_dimensions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get default equal weights if generation failed"""
        num_dims = 4 + len(additional_dimensions)
        equal_weight = 1.0 / num_dims
        
        default_weights = {
            "coverage": equal_weight,
            "insight": equal_weight,
            "instruction_following": equal_weight,
            "clarity": equal_weight
        }
        
        for dim in additional_dimensions:
            dim_name = dim.get('meta_dimension_name', '').lower().replace(' ', '_').replace('-', '_')
            default_weights[dim_name] = equal_weight
            
        return default_weights

    def generate_dimension_criteria(self, query_id: int, dimension_name: str,
                                  all_dims_with_definition: Dict[str, str],
                                  key_facts: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Generate specific criteria for a dimension. Uses key facts for Grounding-driven criteria."""
        cache_key = f"criteria_{query_id}_{dimension_name}"
        cached_result = self.cache_manager.get("criteria", cache_key)

        if cached_result is not None:
            return cached_result

        query_data = self.queries[query_id]
        task_prompt = query_data['prompt']

        logger.info(f"Generating criteria for dimension '{dimension_name}' in query {query_id}")

        # Format meta dimensions string
        meta_dims_str = "\n".join([
            f"- **{dim}**: {all_dims_with_definition[dim]}"
            for dim in all_dims_with_definition
        ])

        # Use key-facts-driven prompt for dynamic dimensions with attachments
        if key_facts:
            key_facts_json = json.dumps(key_facts, ensure_ascii=False, indent=2)
            formatted_prompt = self.criteria_generation_with_key_facts_prompt.format(
                task_prompt=task_prompt,
                num_dimensions=len(all_dims_with_definition),
                meta_dimensions=meta_dims_str,
                dimension_name=dimension_name,
                key_facts_json=key_facts_json
            )
        else:
            formatted_prompt = self.criteria_generation_prompt.format(
                task_prompt=task_prompt,
                num_dimensions=len(all_dims_with_definition),
                meta_dimensions=meta_dims_str,
                dimension_name=dimension_name
            )
        
        dim_definition = all_dims_with_definition.get(dimension_name, "")

        messages = [{"role": "user", "content": formatted_prompt}]

        # Try up to 2 attempts for criteria generation
        criteria = None
        for attempt in range(2):
            response = self.generate_llm_response(messages, max_tokens=8192, temperature=0.1)
            try:
                criteria_json = self.extract_json_from_analysis_output(response)
                if criteria_json:
                    parsed = json.loads(criteria_json)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        # Normalize criterion weights to sum to 1.0
                        total_weight = sum(item.get('weight', 0) for item in parsed)
                        if total_weight > 0:
                            for item in parsed:
                                item['weight'] = item.get('weight', 0) / total_weight
                        criteria = parsed
                        logger.info(f"Generated {len(criteria)} criteria for dimension '{dimension_name}'")
                        break
                    else:
                        logger.warning(f"Invalid criteria format for dimension '{dimension_name}' (attempt {attempt+1})")
                else:
                    logger.warning(f"Could not extract JSON from criteria response for dimension '{dimension_name}' (attempt {attempt+1})")
            except Exception as e:
                logger.error(f"Failed to parse criteria for dimension '{dimension_name}' (attempt {attempt+1}): {e}")

        if criteria is None:
            logger.warning(f"All attempts failed for criteria generation of '{dimension_name}', using fallback")
            criteria = self._get_default_criteria(dimension_name, dim_definition)
        
        # Cache and return results
        self.cache_manager.set("criteria", cache_key, criteria)
        return criteria
    
    def _get_default_criteria(self, dimension_name: str, definition: str = "") -> List[Dict[str, Any]]:
        """Get default criteria if generation failed. Generates 3 meaningful criteria from the dimension definition."""
        if not definition:
            definition = f"Quality of {dimension_name}"

        return [
            {
                "criterion": f"Core quality of {dimension_name}",
                "explanation": f"How well the report addresses the primary aspects of: {definition}",
                "weight": 0.5
            },
            {
                "criterion": f"Depth and specificity of {dimension_name}",
                "explanation": f"Whether the report provides detailed, specific analysis rather than superficial coverage for: {definition}",
                "weight": 0.3
            },
            {
                "criterion": f"Relevance and task-alignment of {dimension_name}",
                "explanation": f"Whether the report's treatment of this dimension is well-aligned with the task requirements: {definition}",
                "weight": 0.2
            }
        ]

    def _score_single_dimension(self, query_id: int, task_prompt: str, report: str,
                               dim_name: str, criteria_list: List[Dict[str, Any]],
                               key_facts: List[Dict[str, str]] = None) -> tuple[str, List[Dict[str, Any]]]:
        """Score a single dimension of a report with retry mechanism"""
        logger.info(f"Scoring dimension: {dim_name}")

        # Format criteria for this single dimension
        criteria_for_dimension = [
            {
                "criterion": item["criterion"],
                "explanation": item["explanation"]
            } for item in criteria_list
        ]

        criteria_json = json.dumps(criteria_for_dimension, ensure_ascii=False, indent=2)

        # Use key-facts-aware scoring prompt when key facts are available
        if key_facts:
            key_facts_json = json.dumps(key_facts, ensure_ascii=False, indent=2)
            formatted_prompt = self.scoring_with_key_facts_prompt.format(
                task_prompt=task_prompt,
                report=report,
                criteria_of_one_dimension_json=criteria_json,
                key_facts_json=key_facts_json
            )
        else:
            formatted_prompt = self.scoring_prompt.format(
                task_prompt=task_prompt,
                report=report,
                criteria_of_one_dimension_json=criteria_json
            )
        
        messages = [{"role": "user", "content": formatted_prompt}]
        
        # Retry mechanism: try up to 3 times
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.generate_llm_response(messages, max_tokens=8192, temperature=0.1)

                # Parse the response which should be in format: {"criterion_1": {"analysis": "...", "report_score_0_to_10": x.xx}, ...}
                dimension_response = json.loads(self.extract_json_from_analysis_output(response))

                # Convert to the desired format with criterion text included
                dimension_scores = []
                resp_map = {item["criterion"]: item for item in dimension_response}
                for criterion_item in criteria_list:
                    name = criterion_item["criterion"]
                    item = resp_map[name]
                    dimension_scores.append({
                        "criterion": name,
                        "analysis": item["analysis"],
                        "report_score_0_to_10": float(item["report_score_0_to_10"])
                    })
                
                logger.info(f"Successfully scored dimension '{dim_name}' with {len(dimension_scores)} criteria")
                return dim_name, dimension_scores
                
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for dimension '{dim_name}': {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying dimension '{dim_name}'...")
                continue
        
        # If all retries failed, log error and raise exception to prevent caching
        logger.error(f"All {max_retries} attempts failed for dimension '{dim_name}'. Last error: {last_exception}")
        raise Exception(f"Dimension '{dim_name}' scoring failed after {max_retries} attempts: {last_exception}")

    def score_report_pointwise(self, query_id: int, report: str, all_criteria: Dict[str, List[Dict[str, Any]]],
                              max_workers: int = None,
                              key_facts: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Score a single report using pointwise evaluation - processes each dimension in parallel"""
        cache_key = f"scores_{query_id}_{hash(report)}"
        cached_result = self.cache_manager.get("scores", cache_key)

        if cached_result is not None:
            return cached_result

        query_data = self.queries[query_id]
        task_prompt = query_data['prompt']

        logger.info(f"Scoring report for query {query_id} - processing {len(all_criteria)} dimensions in parallel")

        # Initialize final scores structure
        final_scores = {}
        has_scoring_errors = False

        # Use a reasonable default for max_workers if not specified
        if max_workers is None:
            max_workers = min(4, len(all_criteria))

        # Process dimensions in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all dimension scoring tasks (pass key_facts for context-aware scoring)
            future_to_dim = {
                executor.submit(
                    self._score_single_dimension,
                    query_id, task_prompt, report, dim_name, criteria_list,
                    key_facts=key_facts
                ): dim_name
                for dim_name, criteria_list in all_criteria.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_dim):
                dim_name = future_to_dim[future]
                try:
                    result_dim_name, dimension_scores = future.result()
                    final_scores[result_dim_name] = dimension_scores
                except Exception as exc:
                    logger.error(f"Dimension {dim_name} scoring failed: {exc}")
                    # Set empty result for failed dimension to avoid KeyError later
                    final_scores[dim_name] = []
                    has_scoring_errors = True
        
        logger.info(f"Completed scoring for query {query_id} across all dimensions")
        
        # Only cache results if all dimensions scored successfully
        if not has_scoring_errors:
            self.cache_manager.set("scores", cache_key, final_scores)
            logger.info(f"Cached successful scores for query {query_id}")
        else:
            logger.info(f"Skipping cache save for query {query_id} due to dimension scoring errors - can retry later")
        
        return final_scores

    def calculate_hierarchical_scores(self, scores: Dict[str, Any],
                                    all_criteria: Dict[str, List[Dict[str, Any]]],
                                    dimension_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate final hierarchical weighted scores.

        If a dimension's scoring failed (empty scores), it is excluded from
        the final aggregation and its weight is redistributed proportionally
        among the successfully scored dimensions.
        """
        final_scores = {}
        dim_score_map = {}  # dim_name -> score (only for successful dimensions)
        failed_dims = []

        for dim_name, criteria_list in all_criteria.items():
            if dim_name not in scores:
                failed_dims.append(dim_name)
                continue

            # Calculate weighted average for this dimension
            dim_scores = scores[dim_name]
            if not isinstance(dim_scores, list) or len(dim_scores) == 0:
                failed_dims.append(dim_name)
                final_scores[f"{dim_name}_score"] = None  # Mark as failed
                continue

            weighted_dim_score = 0.0
            total_criterion_weight = 0.0

            for i, criterion_data in enumerate(criteria_list):
                if i < len(dim_scores):
                    score_item = dim_scores[i]
                    if (isinstance(score_item, dict) and
                        criterion_data['criterion'] == score_item['criterion'] and
                        'report_score_0_to_10' in score_item):

                        score_value = score_item['report_score_0_to_10']
                        criterion_weight = criterion_data['weight']

                        weighted_dim_score += float(score_value) * float(criterion_weight)
                        total_criterion_weight += float(criterion_weight)

            if float(total_criterion_weight) > 0:
                final_dim_score = float(weighted_dim_score) / float(total_criterion_weight)
            else:
                failed_dims.append(dim_name)
                final_scores[f"{dim_name}_score"] = None
                continue

            final_scores[f"{dim_name}_score"] = final_dim_score
            dim_score_map[dim_name] = final_dim_score

        # Redistribute weights: exclude failed dimensions, normalize remaining weights
        if failed_dims:
            logger.warning(
                f"Dimensions with failed scoring (excluded from aggregation): {failed_dims}"
            )

        successful_weight_sum = sum(
            dimension_weights.get(d, 0) for d in dim_score_map
        )

        total_weighted_score = 0.0
        if successful_weight_sum > 0:
            for dim_name, dim_score in dim_score_map.items():
                # Proportionally rescale weight so successful dims sum to 1.0
                rescaled_weight = dimension_weights.get(dim_name, 0) / successful_weight_sum
                total_weighted_score += dim_score * rescaled_weight

        final_scores['total_weighted_score'] = float(total_weighted_score)
        return final_scores
