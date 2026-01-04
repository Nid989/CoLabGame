from abc import ABC, abstractmethod
from typing import Dict, Any, List
import random


class BaseTask(ABC):
    """Base task class providing common functionality for all task implementations."""

    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    @abstractmethod
    def generate(self, **kwargs) -> Dict[str, Any]:
        """
        Generate a task instance.
        Must return a dictionary with task data.
        """
        pass

    @abstractmethod
    def get_task_type(self) -> str:
        """Return the task type identifier."""
        pass

    @abstractmethod
    def get_level(self) -> int:
        """Return the difficulty level of this task."""
        pass

    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration for this task type."""
        return {}

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate if the given config is valid for this task."""
        return True

    def get_required_config_keys(self) -> List[str]:
        """Return list of required configuration keys."""
        return []

    def get_optional_config_keys(self) -> List[str]:
        """Return list of optional configuration keys."""
        return []
