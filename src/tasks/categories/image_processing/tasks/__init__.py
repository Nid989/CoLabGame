"""
Task implementations for visual content integration category.
Register all task generators here.
"""

from .level1_tasks import BasicImageInsertionGenerator
from .level2_tasks import ImageResizeInsertionGenerator
from .level3_tasks import ImageModifyCaptionGenerator

# Task registry for visual content integration
TASK_GENERATORS = {}


def register_generator(task_type: str, generator_class):
    """Register a task generator class."""
    TASK_GENERATORS[task_type] = generator_class


def get_task_generator(task_type: str):
    """Get a task generator instance for the given task type."""
    if task_type in TASK_GENERATORS:
        return TASK_GENERATORS[task_type]()
    return None


def get_all_generators():
    """Get all registered task generators."""
    return {task_type: generator_class() for task_type, generator_class in TASK_GENERATORS.items()}


# Register all task generators
register_generator("basic_image_insertion", BasicImageInsertionGenerator)
register_generator("image_resize_insertion", ImageResizeInsertionGenerator)
register_generator("image_modify_caption", ImageModifyCaptionGenerator)
