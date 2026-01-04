"""
Task generators for the Integrated Content Workflow category.
"""

# Task generator registry
TASK_GENERATORS = {}


def register_generator(task_type: str, generator_class):
    """Register a task generator class."""
    TASK_GENERATORS[task_type] = generator_class


def get_task_generator(task_type: str):
    """Get a task generator instance by type."""
    if task_type not in TASK_GENERATORS:
        _ensure_generators_loaded()

    if task_type not in TASK_GENERATORS:
        raise ValueError(f"Unknown task type: {task_type}")

    generator_class = TASK_GENERATORS[task_type]
    return generator_class()


def get_all_generators():
    """Get all available task generators."""
    _ensure_generators_loaded()
    return {task_type: gen_class() for task_type, gen_class in TASK_GENERATORS.items()}


def _ensure_generators_loaded():
    """Ensure all task generators are loaded and registered."""
    if not TASK_GENERATORS:
        # Import task modules to trigger registration
        from . import level1_tasks

        # Register Level 1 tasks
        register_generator("basic_info_gathering", level1_tasks.BasicInfoGatheringTaskGenerator)


__all__ = ["register_generator", "get_task_generator", "get_all_generators"]
