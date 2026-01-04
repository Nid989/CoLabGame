"""
Task registry for data analysis category.
Manages registration and retrieval of all task generators.
"""

# Global registry for task generators
TASK_GENERATORS = {}


def register_generator(task_type: str, generator_class):
    """Register a task generator class."""
    TASK_GENERATORS[task_type] = generator_class


def get_task_generator(task_type: str):
    """Get a task generator instance by task type."""
    if task_type not in TASK_GENERATORS:
        raise ValueError(f"Unknown task type: {task_type}")

    generator_class = TASK_GENERATORS[task_type]
    return generator_class()


def get_all_generators():
    """Get all registered task generators as instances."""
    return {task_type: gen_class() for task_type, gen_class in TASK_GENERATORS.items()}


def _ensure_generators_loaded():
    """Ensure all generators are loaded and registered."""
    if not TASK_GENERATORS:
        # Import and register Level 1 tasks
        from .level1_tasks import SimpleDataTransferGenerator

        register_generator("simple_data_transfer", SimpleDataTransferGenerator)

        # Import and register Level 2 tasks
        from .level2_tasks import BasicDataAggregationGenerator

        register_generator("basic_data_aggregation", BasicDataAggregationGenerator)

        # Import and register Level 3 tasks
        from .level3_tasks import SimpleCalculationOutputGenerator

        register_generator("simple_calculation_output", SimpleCalculationOutputGenerator)


# Ensure generators are loaded when module is imported
_ensure_generators_loaded()
