import glob
import json
import logging
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """Load model result JSON files."""

    def __init__(self, data_dir: str, models: list[str], data_type: str = "text"):
        self.data_dir = data_dir
        self.models = models
        self.data_type = data_type

    def load_all(self) -> dict[str, list[dict]]:
        result = {}
        for model in self.models:
            entries = self.load_model(model)
            if entries:
                result[model] = entries
        return result

    def load_model(self, model_name: str) -> list[dict]:
        pattern = os.path.join(self.data_dir, f"{model_name}_{self.data_type}*.json")
        files = glob.glob(pattern)
        if not files:
            # Fallback: try wildcard on the whole suffix (handles "multimodal" vs "multimodel" etc.)
            fallback = os.path.join(self.data_dir, f"{model_name}_*.json")
            files = glob.glob(fallback)
        if not files:
            logger.warning(f"No data file found for model '{model_name}' at {pattern}")
            return []
        if len(files) > 1:
            logger.warning(f"Multiple files found for '{model_name}', using first: {files[0]}")
        filepath = files[0]

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} entries for '{model_name}' from {os.path.basename(filepath)}")
        return data
