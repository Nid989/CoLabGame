from typing import Dict, List, Any
from ..base import (
    BaseCategory,
    FileProviderInterface,
    ConfigProviderInterface,
    EvaluationProviderInterface,
)
from .evaluators import DebuggingAndRefactoringEvaluators
from .setup_config import DebuggingAndRefactoringSetupConfig
from .tasks import get_task_generator, get_all_generators
from .providers import (
    DebuggingAndRefactoringFileProvider,
    DebuggingAndRefactoringConfigProvider,
    DebuggingAndRefactoringEvaluationProvider,
)


class DebuggingAndRefactoringCategory(BaseCategory):
    """Debugging and Refactoring category implementation."""

    def __init__(self):
        super().__init__("debugging_and_refactoring")
        self.set_evaluators(DebuggingAndRefactoringEvaluators())
        self.setup_config = DebuggingAndRefactoringSetupConfig()
        self._file_provider = DebuggingAndRefactoringFileProvider()
        self._config_provider = DebuggingAndRefactoringConfigProvider()
        self._evaluation_provider = DebuggingAndRefactoringEvaluationProvider()
        self._load_tasks()

    def _load_tasks(self):
        """Load and register all tasks for this category."""
        all_tasks = get_all_generators()
        level_mapping = {
            "basic_python_syntax_fix": 1,
            "simple_logic_completion": 2,
            "multi_file_config_update": 3,
        }
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
        task_types = self.get_task_types()
        if task_type not in task_types:
            return False
        required_fields = ["task_type", "level"]
        return all(field in config for field in required_fields)

    def get_default_config(self, task_type: str) -> Dict[str, Any]:
        """Return default configuration for the specified task type."""
        task_types = self.get_task_types()
        if task_type not in task_types:
            return {}
        level_mapping = {
            "fix_missing_quote": 1,
            "change_number": 1,
            "uncomment_line": 1,
            "fix_capitalization": 1,
            "fix_multiple_errors": 2,
            "complete_simple_function": 2,
            "update_config_file": 2,
            "add_output_statement": 2,
            "write_simple_function": 3,
            "fix_logic_error": 3,
            "fix_variable_scope": 3,
            "multi_file_config": 3,
        }
        return {
            "task_type": task_type,
            "level": level_mapping.get(task_type, 1),
            "category": self.get_category_name(),
        }

    def get_setup_config(self):
        """Get setup config for this category."""
        return self.setup_config

    def get_file_provider(self) -> FileProviderInterface:
        """Return the file provider for this category."""
        return self._file_provider

    def get_config_provider(self) -> ConfigProviderInterface:
        """Return the config provider for this category."""
        return self._config_provider

    def get_evaluation_provider(self) -> EvaluationProviderInterface:
        """Return the evaluation provider for this category."""
        return self._evaluation_provider
