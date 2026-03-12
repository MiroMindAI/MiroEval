import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm

from .cache.file_cache import FileCache
from .data_loader import DataLoader
from .evaluation.alignment_evaluator import AlignmentEvaluator
from .evaluation.intrinsic_evaluator import IntrinsicEvaluator
from .preprocessors import get_preprocessor
from .structuring.structurer import ProcessStructurer
from .utils.config import get_nested
from .utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ProcessEvalPipeline:
    """Orchestrator for the full process evaluation pipeline."""

    def __init__(self, config: dict):
        self.config = config

        # Data
        data_dir = get_nested(config, "data", "data_dir", default="../data/method_results/mirobench-text-refined-v2/mirobench-text-refined")
        data_type = get_nested(config, "data", "data_type", default="text")
        self.models = config.get("target_models", [])
        self.loader = DataLoader(data_dir, self.models, data_type=data_type)

        # LLM
        llm_cfg = config.get("llm", {})
        self.llm = LLMClient(
            model=llm_cfg.get("model", "google/gemini-2.5-pro"),
            api_type=llm_cfg.get("api_type", "openrouter"),
            max_tokens=llm_cfg.get("max_tokens", 8192),
            temperature=llm_cfg.get("temperature", 0.1),
            retry_count=llm_cfg.get("retry_count", 3),
            retry_backoff=llm_cfg.get("retry_backoff", 2.0),
        )

        # Cache
        cache_dir = get_nested(config, "cache", "cache_dir", default="outputs/cache")
        self.cache_enabled = get_nested(config, "cache", "enabled", default=True)
        self.structuring_cache = FileCache(cache_dir, "structuring")
        self.intrinsic_cache = FileCache(cache_dir, "intrinsic_scores")
        self.alignment_cache = FileCache(cache_dir, "alignment_scores")

        # Components
        self.structurer = ProcessStructurer(self.llm, self.structuring_cache)
        self.intrinsic_eval = IntrinsicEvaluator(self.llm, self.intrinsic_cache)
        report_max = get_nested(config, "preprocessing", "report_max_chars", default=30000)
        self.alignment_eval = AlignmentEvaluator(self.llm, self.alignment_cache, report_max_chars=report_max)

        # Execution settings
        self.max_workers = get_nested(config, "execution", "max_workers", default=10)
        self.continue_on_error = get_nested(config, "execution", "continue_on_error", default=True)

        # Entry selection
        self.max_entries = get_nested(config, "entry_selection", "max_entries_per_model", default=None)
        self.entry_ids = get_nested(config, "entry_selection", "entry_ids", default=None)

        # Preprocessing config
        self.preprocess_config = config.get("preprocessing", {})

    def run_full(self) -> dict:
        """Run the complete pipeline: Phase 1 + Phase 2."""
        all_data = self.loader.load_all()
        if not all_data:
            logger.error("No data loaded")
            return {}

        self._print_data_summary(all_data)

        results = {}
        all_tasks = self._build_task_list(all_data)

        logger.info(f"Processing {len(all_tasks)} entries with {self.max_workers} workers")

        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {
                executor.submit(self._process_entry, model, entry): (model, entry.get("id", idx))
                for model, entry, idx in all_tasks
            }

            with tqdm(total=len(future_to_key), desc="Processing") as pbar:
                for future in as_completed(future_to_key):
                    model, entry_id = future_to_key[future]
                    key = f"{model}_{entry_id}"
                    try:
                        result = future.result()
                        if result:
                            results[key] = result
                            completed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error(f"Error processing {key}: {e}")
                        failed += 1
                        if not self.continue_on_error:
                            raise
                    pbar.update(1)

        logger.info(f"Completed: {completed}, Failed: {failed}")

        # Aggregate and save
        final = self._aggregate_results(results)
        self._save_results(final)
        self._print_summary(final)

        logger.info(f"Total LLM cost: ${self.llm.total_cost:.2f}")
        return final

    def run_phase1(self) -> dict:
        """Run only Phase 1: preprocess + structure."""
        all_data = self.loader.load_all()
        if not all_data:
            return {}

        self._print_data_summary(all_data)
        all_tasks = self._build_task_list(all_data)
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {
                executor.submit(self._process_entry_phase1, model, entry): (model, entry.get("id", idx))
                for model, entry, idx in all_tasks
            }

            with tqdm(total=len(future_to_key), desc="Phase 1: Structuring") as pbar:
                for future in as_completed(future_to_key):
                    model, entry_id = future_to_key[future]
                    key = f"{model}_{entry_id}"
                    try:
                        result = future.result()
                        if result:
                            results[key] = result
                    except Exception as e:
                        logger.error(f"Phase 1 error for {key}: {e}")
                        if not self.continue_on_error:
                            raise
                    pbar.update(1)

        # Save structured processes
        output_cfg = self.config.get("output", {})
        sp_file = output_cfg.get("structured_processes_file", "outputs/results/structured_processes.json")
        os.makedirs(os.path.dirname(sp_file), exist_ok=True)
        with open(sp_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Phase 1 results saved to {sp_file}")

        logger.info(f"Total LLM cost: ${self.llm.total_cost:.2f}")
        return results

    def run_phase2(self, structured_data: dict | None = None) -> dict:
        """Run only Phase 2: evaluate (requires Phase 1 outputs)."""
        all_data = self.loader.load_all()
        if not all_data:
            return {}

        # Load structured processes from cache or file
        if structured_data is None:
            output_cfg = self.config.get("output", {})
            sp_file = output_cfg.get("structured_processes_file", "outputs/results/structured_processes.json")
            if os.path.exists(sp_file):
                with open(sp_file, "r", encoding="utf-8") as f:
                    structured_data = json.load(f)
                logger.info(f"Loaded structured processes from {sp_file}")
            else:
                logger.error(f"No structured processes found at {sp_file}. Run Phase 1 first.")
                return {}

        all_tasks = self._build_task_list(all_data)
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {}
            for model, entry, idx in all_tasks:
                entry_id = entry.get("id", idx)
                key = f"{model}_{entry_id}"
                structured = structured_data.get(key, {}).get("structured_process")
                if structured is None:
                    structured = self.structuring_cache.get(key)
                if structured is None:
                    logger.warning(f"No structured process for {key}, skipping Phase 2")
                    continue
                future = executor.submit(
                    self._process_entry_phase2, model, entry, structured
                )
                future_to_key[future] = (model, entry_id)

            with tqdm(total=len(future_to_key), desc="Phase 2: Evaluation") as pbar:
                for future in as_completed(future_to_key):
                    model, entry_id = future_to_key[future]
                    key = f"{model}_{entry_id}"
                    try:
                        result = future.result()
                        if result:
                            results[key] = result
                    except Exception as e:
                        logger.error(f"Phase 2 error for {key}: {e}")
                        if not self.continue_on_error:
                            raise
                    pbar.update(1)

        final = self._aggregate_results(results)
        self._save_results(final)
        self._print_summary(final)

        logger.info(f"Total LLM cost: ${self.llm.total_cost:.2f}")
        return final

    def _build_task_list(self, all_data: dict) -> list[tuple]:
        """Build list of (model, entry, idx) tuples respecting entry selection."""
        tasks = []
        for model, entries in all_data.items():
            selected = entries
            if self.entry_ids:
                selected = [e for e in entries if e.get("id") in self.entry_ids]
            if self.max_entries:
                selected = selected[:self.max_entries]
            for idx, entry in enumerate(selected):
                tasks.append((model, entry, idx))
        return tasks

    def _get_preprocessor(self, model_name: str):
        overrides = get_nested(self.preprocess_config, "model_overrides", model_name, default={})
        max_chars = overrides.get("max_chars", self.preprocess_config.get("max_chars", 30000))
        return get_preprocessor(max_chars=max_chars)

    def _process_entry(self, model: str, entry: dict) -> dict | None:
        """Full pipeline for a single entry: preprocess -> structure -> evaluate."""
        entry_id = entry.get("id", "?")
        query = entry.get("query", entry.get("rewritten_query", ""))
        process_text = entry.get("process", "")
        report = entry.get("response", "")

        if not process_text.strip():
            logger.warning(f"Empty process for {model}_{entry_id}")
            return None

        # Phase 1: Preprocess + Structure
        pp = self._get_preprocessor(model)
        preprocessed = pp.preprocess(process_text)

        structured = self.structurer.structure(entry_id, model, preprocessed, query)
        if structured is None:
            return None

        # Phase 2: Evaluate
        intrinsic = self.intrinsic_eval.evaluate(entry_id, model, structured, query)
        alignment = self.alignment_eval.evaluate(entry_id, model, structured, report, query)

        return {
            "model": model,
            "entry_id": entry_id,
            "query": query,
            "structured_process": structured,
            "intrinsic_scores": intrinsic,
            "alignment_scores": alignment,
        }

    def _process_entry_phase1(self, model: str, entry: dict) -> dict | None:
        entry_id = entry.get("id", "?")
        query = entry.get("query", entry.get("rewritten_query", ""))
        process_text = entry.get("process", "")

        if not process_text.strip():
            return None

        pp = self._get_preprocessor(model)
        preprocessed = pp.preprocess(process_text)
        structured = self.structurer.structure(entry_id, model, preprocessed, query)
        if structured is None:
            return None

        return {
            "model": model,
            "entry_id": entry_id,
            "query": query,
            "structured_process": structured,
        }

    def _process_entry_phase2(self, model: str, entry: dict, structured: dict) -> dict | None:
        entry_id = entry.get("id", "?")
        query = entry.get("query", entry.get("rewritten_query", ""))
        report = entry.get("response", "")

        intrinsic = self.intrinsic_eval.evaluate(entry_id, model, structured, query)
        alignment = self.alignment_eval.evaluate(entry_id, model, structured, report, query)

        return {
            "model": model,
            "entry_id": entry_id,
            "intrinsic_scores": intrinsic,
            "alignment_scores": alignment,
        }

    def _aggregate_results(self, results: dict) -> dict:
        """Aggregate per-entry results into per-model summaries."""
        from .evaluation.intrinsic_evaluator import DIMENSIONS as INTRINSIC_DIMS
        from .evaluation.alignment_evaluator import DIMENSIONS as ALIGNMENT_DIMS

        model_scores = {}

        for key, entry_result in results.items():
            model = entry_result.get("model", key.rsplit("_", 1)[0])
            if model not in model_scores:
                model_scores[model] = {d: [] for d in INTRINSIC_DIMS + ALIGNMENT_DIMS}

            intrinsic = entry_result.get("intrinsic_scores") or {}
            for dim in INTRINSIC_DIMS:
                if dim in intrinsic and isinstance(intrinsic[dim], dict):
                    score = intrinsic[dim].get("score")
                    if score is not None:
                        model_scores[model][dim].append(score)

            alignment = entry_result.get("alignment_scores") or {}
            for dim in ALIGNMENT_DIMS:
                if dim in alignment and isinstance(alignment[dim], dict):
                    score = alignment[dim].get("score")
                    if score is not None:
                        model_scores[model][dim].append(score)

        # Calculate averages
        summary = {}
        for model, dims in model_scores.items():
            summary[model] = {}
            all_scores = []
            for dim, scores in dims.items():
                if scores:
                    avg = sum(scores) / len(scores)
                    summary[model][dim] = {"avg": round(avg, 2), "count": len(scores)}
                    all_scores.append(avg)
                else:
                    summary[model][dim] = {"avg": None, "count": 0}
            if all_scores:
                summary[model]["overall_avg"] = round(sum(all_scores) / len(all_scores), 2)
            else:
                summary[model]["overall_avg"] = None

        return {
            "timestamp": datetime.now().isoformat(),
            "models": list(model_scores.keys()),
            "summary": summary,
            "entry_results": results,
        }

    def _save_results(self, final: dict):
        output_cfg = self.config.get("output", {})
        results_file = output_cfg.get("results_file", "outputs/results/process_eval_results.json")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {results_file}")

    def _print_data_summary(self, all_data: dict):
        logger.info("=== Data Summary ===")
        for model, entries in all_data.items():
            logger.info(f"  {model}: {len(entries)} entries")

    def _print_summary(self, final: dict):
        from .evaluation.intrinsic_evaluator import DIMENSIONS as INTRINSIC_DIMS
        from .evaluation.alignment_evaluator import DIMENSIONS as ALIGNMENT_DIMS

        summary = final.get("summary", {})
        if not summary:
            return

        all_dims = INTRINSIC_DIMS + ALIGNMENT_DIMS

        # Header
        dim_headers = [d[:8] for d in all_dims]
        header = f"{'Model':<14}" + "".join(f"{h:>10}" for h in dim_headers) + f"{'Overall':>10}"
        print("\n" + "=" * len(header))
        print("Process Evaluation Results")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        # Sort by overall_avg
        sorted_models = sorted(
            summary.items(),
            key=lambda x: x[1].get("overall_avg") or 0,
            reverse=True,
        )

        for rank, (model, scores) in enumerate(sorted_models, 1):
            row = f"{model:<14}"
            for dim in all_dims:
                avg = scores.get(dim, {}).get("avg")
                row += f"{avg:>10.2f}" if avg is not None else f"{'N/A':>10}"
            overall = scores.get("overall_avg")
            row += f"{overall:>10.2f}" if overall is not None else f"{'N/A':>10}"
            print(row)

        print("=" * len(header) + "\n")
