"""
Dynamic task implementations for code maintenance category.
"""

# Task registry
TASK_GENERATORS = {}


def register_generator(task_type: str, generator_class):
    """Register a task implementation."""
    TASK_GENERATORS[task_type] = generator_class


def get_task_generator(task_type: str):
    """Get a task implementation by type."""
    if task_type not in TASK_GENERATORS:
        # Try to import and register task implementations if not already done
        _ensure_generators_loaded()
        if task_type not in TASK_GENERATORS:
            raise ValueError(f"Unknown task type: {task_type}")
    return TASK_GENERATORS[task_type]()


def get_all_generators():
    """Get all registered task implementations."""
    _ensure_generators_loaded()
    return {task_type: gen_class() for task_type, gen_class in TASK_GENERATORS.items()}


def _ensure_generators_loaded():
    """Ensure all task implementations are loaded and registered."""
    if not TASK_GENERATORS:
        # Import and register all task implementations
        from .level1_tasks import (
            BasicPythonSyntaxFixGenerator,
        )
        from .level2_tasks import (
            SimpleLogicCompletionGenerator,
        )
        from .level3_tasks import (
            MultiFileConfigUpdateGenerator,
        )

        # Register Level 1 tasks
        register_generator("basic_python_syntax_fix", BasicPythonSyntaxFixGenerator)

        # Register Level 2 task
        register_generator("simple_logic_completion", SimpleLogicCompletionGenerator)

        # Register Level 3 task
        register_generator("multi_file_config_update", MultiFileConfigUpdateGenerator)


__all__ = ["get_task_generator", "get_all_generators", "register_generator"]
