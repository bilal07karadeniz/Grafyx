"""Main entry point for the sample project."""

from utils import helper_function, format_output
from models import User, Product


def process_data(data: list) -> dict:
    """Process raw data into structured format."""
    validated = validate(data)
    result = transform(validated)
    return format_output(result)


def validate(data: list) -> list:
    """Validate input data by removing None values."""
    if not data:
        raise ValueError("Empty data")
    return [item for item in data if item is not None]


def transform(data: list) -> dict:
    """Transform validated data into a dictionary."""
    return {"items": data, "count": len(data)}


def batch_process(datasets: list[list]) -> list[dict]:
    """Process multiple datasets."""
    results = []
    for dataset in datasets:
        try:
            result = process_data(dataset)
            results.append(result)
        except ValueError as e:
            results.append({"error": str(e)})
    return results


class DataProcessor:
    """Processes data in batch mode."""

    def __init__(self, config: dict):
        self.config = config
        self._cache: dict = {}

    def run(self, data: list) -> dict:
        """Run the data processor."""
        return process_data(data)

    def run_cached(self, key: str, data: list) -> dict:
        """Run with caching."""
        if key not in self._cache:
            self._cache[key] = self.run(data)
        return self._cache[key]

    def clear_cache(self) -> None:
        """Clear the processing cache."""
        self._cache.clear()
