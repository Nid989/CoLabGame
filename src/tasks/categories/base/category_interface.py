from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# Import provider interfaces
from .provider_interfaces import (
    FileProviderInterface,
    ConfigProviderInterface,
    EvaluationProviderInterface,
)


class BaseCategoryInterface(ABC):
    """Abstract interface that all categories must implement for standardization."""

    @abstractmethod
    def get_category_name(self) -> str:
        """Return the name/identifier of this category."""
        pass

    @abstractmethod
    def get_supported_levels(self) -> List[int]:
        """Return list of supported levels for this category (e.g., [1, 2, 3])."""
        pass

    @abstractmethod
    def get_task_types(self, level: Optional[int] = None) -> Dict[str, Any]:
        """
        Return available task types for the category.
        If level is specified, return only tasks for that level.
        Returns dict mapping task_type -> task_class.
        """
        pass

    @abstractmethod
    def get_evaluators(self) -> Any:
        """Return the evaluators object for this category."""
        pass

    @abstractmethod
    def validate_task_config(self, task_type: str, config: Dict[str, Any]) -> bool:
        """Validate if the given config is valid for the specified task type."""
        pass

    @abstractmethod
    def get_default_config(self, task_type: str) -> Dict[str, Any]:
        """Return default configuration for the specified task type."""
        pass

    # New provider methods for the enhanced architecture
    @abstractmethod
    def get_file_provider(self) -> FileProviderInterface:
        """Return the file provider for this category."""
        pass

    @abstractmethod
    def get_config_provider(self) -> ConfigProviderInterface:
        """Return the config provider for this category."""
        pass

    @abstractmethod
    def get_evaluation_provider(self) -> EvaluationProviderInterface:
        """Return the evaluation provider for this category."""
        pass
