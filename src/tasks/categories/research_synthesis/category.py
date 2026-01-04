"""
Main category class for Information Synthesis & Presentation.
"""

from typing import Dict, List, Any
from ..base import (
    BaseCategory,
    FileProviderInterface,
    ConfigProviderInterface,
    EvaluationProviderInterface,
)
from .evaluators import ResearchSynthesisEvaluators
from .setup_config import ResearchSynthesisSetupConfig
from .tasks import get_task_generator, get_all_generators
from .providers import (
    ResearchSynthesisFileProvider,
    ResearchSynthesisConfigProvider,
    ResearchSynthesisEvaluationProvider,
)


class ResearchSynthesisCategory(BaseCategory):
    """Research Synthesis category implementation."""

    def __init__(self):
        super().__init__("research_synthesis")
        self.set_evaluators(ResearchSynthesisEvaluators())
        self.setup_config = ResearchSynthesisSetupConfig()

        # Initialize providers
        self._file_provider = ResearchSynthesisFileProvider()
        self._config_provider = ResearchSynthesisConfigProvider()
        self._evaluation_provider = ResearchSynthesisEvaluationProvider()

        self._load_tasks()

    def _load_tasks(self):
        """Load and register all tasks for this category."""
        # Get all task implementations and register them with their levels
        all_tasks = get_all_generators()

        # Level mapping for research synthesis tasks
        level_mapping = {
            # Level 1 tasks
            "basic_web_extraction": 1,
            # Level 2 tasks
            "multi_point_summary": 2,
            # Level 3 tasks
            "file_download_integration": 3,
        }

        # Register tasks with their appropriate levels
        for task_type, task_impl in all_tasks.items():
            if task_type in level_mapping:
                level = level_mapping[task_type]
                self.register_task(task_type, type(task_impl), level)

    def get_supported_levels(self) -> List[int]:
        """Return list of supported levels for this category."""
        return [1, 2, 3]

    def get_task_implementation(self, task_type: str):
        """Get a task implementation instance for the given task type."""
        return get_task_generator(task_type)

    def validate_task_config(self, task_type: str, config: Dict[str, Any]) -> bool:
        """Validate if the given config is valid for the specified task type."""
        # Check if task type exists
        task_types = self.get_task_types()
        if task_type not in task_types:
            return False

        # Basic config validation
        required_fields = ["task_type", "level"]
        return all(field in config for field in required_fields)

    def get_default_config(self, task_type: str) -> Dict[str, Any]:
        """Return default configuration for the specified task type."""
        task_types = self.get_task_types()
        if task_type not in task_types:
            return {}

        # Get the level for this task type
        level_mapping = {
            "basic_web_extraction": 1,
            "multi_point_summary": 2,
            "file_download_integration": 3,
        }

        return {
            "task_type": task_type,
            "level": level_mapping.get(task_type, 1),
            "category": self.get_category_name(),
        }

    def get_setup_config(self):
        """Get setup config for this category."""
        return self.setup_config

    # Enhanced category interface methods for provider pattern
    def get_file_provider(self) -> FileProviderInterface:
        """Return the file provider for this category."""
        return self._file_provider

    def get_config_provider(self) -> ConfigProviderInterface:
        """Return the config provider for this category."""
        return self._config_provider

    def get_evaluation_provider(self) -> EvaluationProviderInterface:
        """Return the evaluation provider for this category."""
        return self._evaluation_provider
