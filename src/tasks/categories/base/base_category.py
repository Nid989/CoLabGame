from typing import Dict, List, Any, Optional
from .category_interface import BaseCategoryInterface


class BaseCategory(BaseCategoryInterface):
    """Base category class providing common functionality for all categories."""

    def __init__(self, category_name: str):
        self.category_name = category_name
        self._task_registry = {}
        self._evaluators = None

    def get_category_name(self) -> str:
        """Return the name/identifier of this category."""
        return self.category_name

    def register_task(self, task_type: str, task_class: Any, level: int):
        """Register a task type with its class and level."""
        if level not in self._task_registry:
            self._task_registry[level] = {}
        self._task_registry[level][task_type] = task_class

    def get_supported_levels(self) -> List[int]:
        """Return list of supported levels for this category."""
        return sorted(self._task_registry.keys())

    def get_task_types(self, level: Optional[int] = None) -> Dict[str, Any]:
        """
        Return available task types for the category.
        If level is specified, return only tasks for that level.
        """
        if level is not None:
            return self._task_registry.get(level, {})

        # Return all task types across all levels
        all_tasks = {}
        for level_tasks in self._task_registry.values():
            all_tasks.update(level_tasks)
        return all_tasks

    def get_evaluators(self) -> Any:
        """Return the evaluators object for this category."""
        return self._evaluators

    def set_evaluators(self, evaluators: Any):
        """Set the evaluators object for this category."""
        self._evaluators = evaluators

    def validate_task_config(self, task_type: str, config: Dict[str, Any]) -> bool:
        """Validate if the given config is valid for the specified task type."""
        # Default implementation - can be overridden by specific categories
        task_types = self.get_task_types()
        return task_type in task_types

    def get_default_config(self, task_type: str) -> Dict[str, Any]:
        """Return default configuration for the specified task type."""
        # Default implementation - should be overridden by specific categories
        return {}
