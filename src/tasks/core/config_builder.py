"""
Configuration builder for framework-compatible task configurations.
Generates JSON configs that comply with the framework schema.
"""

from typing import Dict, List, Any

from .utils import validate_test_example


class ConfigBuilder:
    """
    Builds framework-compatible configuration JSON files.

    Converts task data into the required schema format with:
    - Proper file download configurations
    - Directory creation commands
    - VS Code launch setup
    - Evaluation configurations
    """

    def __init__(self):
        """Initialize ConfigBuilder with category registry support."""
        self._category_registry = None

    def get_available_categories(self) -> Dict[str, Any]:
        """Get available categories from the registry."""
        if not self._category_registry:
            from src.tasks.core.category_registry import category_registry

            self._category_registry = category_registry

        return self._category_registry.get_all_categories()

    def build_config(
        self,
        task_id: str,
        task_example: Dict[str, Any],
        s3_urls: Dict[str, str],
        files_created: Dict[str, str],
        evaluation_mode: str = "compare_text_file",
        category_obj=None,
        config_provider=None,
        evaluation_provider=None,
    ) -> Dict[str, Any]:
        """
        Build complete framework configuration.

        Args:
            task_id: Task UUID
            task_example: Original test example
            s3_urls: S3 URLs for uploaded files
            files_created: Local file information
            evaluation_mode: Evaluation method to use ('compare_text_file' or 'compare_answer')
            category_obj: Legacy category object (for backward compatibility)
            config_provider: Optional config provider for category-specific setup steps
            evaluation_provider: Optional evaluation provider for category-specific evaluation config

        Returns:
            Framework-compatible configuration dictionary
        """
        if not validate_test_example(task_example):
            raise ValueError("Invalid test example provided")

        # Determine snapshot and related apps based on category
        snapshot = "vscode"
        related_apps = ["vscode"]

        # Use category object for backward compatibility, or check task type for category-specific behavior
        if (category_obj and category_obj.get_category_name() == "tabular_data_reporting") or task_example.get("task_type", "").startswith(
            (
                "simple_data_transfer",
                "basic_data_aggregation",
                "simple_calculation_output",
            )
        ):
            snapshot = "os"
            related_apps = ["libreoffice_calc", "os"]

        config = {
            "id": task_id,
            "snapshot": snapshot,
            "instruction": task_example["instructions"],
            "source": "",  # Empty as per examples
            "config": self._build_config_steps(task_example, s3_urls, files_created, config_provider, category_obj),
            "trajectory": "trajectories/",
            "related_apps": related_apps,
            "evaluator": self._build_evaluator_config(
                task_example,
                files_created,
                evaluation_mode,
                s3_urls,
                evaluation_provider,
                category_obj,
            ),
        }

        return config

    def _build_config_steps(
        self,
        task_example: Dict[str, Any],
        s3_urls: Dict[str, str],
        files_created: Dict[str, str],
        config_provider=None,
        category_obj=None,
    ) -> List[Dict[str, Any]]:
        """
        Build configuration steps using category system.

        Args:
            task_example: Original test example
            s3_urls: S3 URLs for uploaded files
            files_created: Local file information
            config_provider: Optional config provider for category-specific setup steps
            category_obj: Category object instance (for backward compatibility)

        Returns:
            List of configuration steps
        """
        # Use config provider if available
        if config_provider and config_provider.supports_task_type(task_example.get("task_type", "")):
            return config_provider.build_setup_steps(task_example, s3_urls, files_created)

        # Fallback to category object for backward compatibility
        if category_obj:
            setup_config = category_obj.get_setup_config()
            return setup_config.build_config_steps(task_example, s3_urls, files_created)

        # Final fallback for backward compatibility
        from src.tasks.categories.debugging_and_refactoring.setup_config import DebuggingAndRefactoringSetupConfig

        setup_config = DebuggingAndRefactoringSetupConfig()
        return setup_config.build_config_steps(task_example, s3_urls, files_created)

    def _build_evaluator_config(
        self,
        task_example: Dict[str, Any],
        files_created: Dict[str, str],
        evaluation_mode: str = "compare_text_file",
        s3_urls: Dict[str, str] = None,
        evaluation_provider=None,
        category_obj=None,
    ) -> Dict[str, Any]:
        """
        Build evaluator configuration using category system.

        Args:
            task_example: Original test example
            files_created: Local file information
            evaluation_mode: Evaluation method to use ('compare_text_file' or 'compare_answer')
            s3_urls: S3 URLs for uploaded files
            evaluation_provider: Optional evaluation provider for category-specific evaluation config
            category_obj: Category object instance (for backward compatibility)

        Returns:
            Evaluator configuration dictionary
        """
        # Use evaluation provider if available
        if evaluation_provider and evaluation_provider.supports_task_type(task_example.get("task_type", "")):
            return evaluation_provider.build_evaluator_config(task_example, files_created, evaluation_mode, s3_urls)

        # Fallback to legacy logic
        task_type = task_example["task_type"]

        # Use category object's evaluators if provided, otherwise fall back to default
        if category_obj:
            evaluators = category_obj.get_evaluators()
        else:
            # Fallback for backward compatibility
            from src.tasks.categories.debugging_and_refactoring.evaluators import DebuggingAndRefactoringEvaluators

            evaluators = DebuggingAndRefactoringEvaluators()

        # Add evaluation_mode to task_example for evaluator generation
        task_example_with_mode = {**task_example, "evaluation_mode": evaluation_mode}

        # Generate evaluator configuration directly
        if evaluators.needs_multi_evaluator(task_type):
            return evaluators.build_multi_evaluator_config(task_type, task_example_with_mode, files_created, s3_urls)
        else:
            return evaluators.build_single_evaluator_config(task_type, task_example_with_mode, files_created, s3_urls)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate generated configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid
        """
        required_fields = ["id", "snapshot", "instruction", "config", "evaluator"]

        # Check required fields
        for field in required_fields:
            if field not in config:
                return False

        # Validate evaluator
        evaluator = config["evaluator"]
        if "func" not in evaluator:
            return False

        # Additional validations can be added here
        return True

    def add_postconfig_steps(self, config: Dict[str, Any], postconfig_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add post-configuration steps to existing configuration.

        Args:
            config: Existing configuration
            postconfig_steps: Additional steps to add

        Returns:
            Updated configuration
        """
        if postconfig_steps:
            config["config"].extend(postconfig_steps)

        return config

    def get_config_summary(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        Get summary of configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Summary dictionary
        """
        return {
            "id": config.get("id", ""),
            "instruction_preview": config.get("instruction", "")[:100] + "..."
            if len(config.get("instruction", "")) > 100
            else config.get("instruction", ""),
            "config_steps": str(len(config.get("config", []))),
            "evaluator_type": config.get("evaluator", {}).get("func", ""),
        }
