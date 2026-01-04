"""Task implementations for Information Synthesis & Presentation category."""

from .level1_tasks import BasicWebExtractionGenerator
from .level2_tasks import MultiPointSummaryGenerator
from .level3_tasks import FileDownloadIntegrationGenerator

# Task registry for this category
TASK_GENERATORS = {}


def register_generator(task_type: str, generator_class):
    """Register a task generator."""
    TASK_GENERATORS[task_type] = generator_class


def get_task_generator(task_type: str):
    """Get a task generator instance."""
    _ensure_generators_loaded()
    if task_type in TASK_GENERATORS:
        return TASK_GENERATORS[task_type]()
    return None


def get_all_generators():
    """Get all registered generators."""
    _ensure_generators_loaded()
    return {task_type: gen_class() for task_type, gen_class in TASK_GENERATORS.items()}


def _ensure_generators_loaded():
    """Ensure all generators are loaded and registered."""
    if not TASK_GENERATORS:
        # Level 1 tasks
        register_generator("basic_web_extraction", BasicWebExtractionGenerator)

        # Level 2 tasks
        register_generator("multi_point_summary", MultiPointSummaryGenerator)

        # Level 3 tasks
        register_generator("file_download_integration", FileDownloadIntegrationGenerator)
