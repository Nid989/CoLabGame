"""
Provider interfaces for category-specific functionality.
These interfaces define the contracts that categories must implement to provide
their specific logic to the core framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class FileProviderInterface(ABC):
    """Interface for category-specific file operations."""

    @abstractmethod
    def get_file_placement_path(self, task_type: str, filename: str) -> str:
        """
        Get the file placement path for a specific task type.

        Args:
            task_type: The type of task (e.g., 'fix_missing_quote')
            filename: The filename to be placed

        Returns:
            Full path where the file should be placed
        """
        pass

    @abstractmethod
    def get_directories_to_create(self, task_type: str) -> List[str]:
        """
        Get list of directories that need to be created for a task type.

        Args:
            task_type: The type of task

        Returns:
            List of directory paths to create
        """
        pass

    @abstractmethod
    def create_task_files(self, task_data: Dict[str, Any], task_id: str, temp_dir: str) -> Optional[Dict[str, str]]:
        """
        Create category-specific task files.

        Args:
            task_data: Task data dictionary
            task_id: Unique task identifier
            temp_dir: Temporary directory for file creation

        Returns:
            Dictionary with file information or None if no special files needed
        """
        pass

    @abstractmethod
    def supports_task_type(self, task_type: str) -> bool:
        """
        Check if this provider supports the given task type.

        Args:
            task_type: The type of task

        Returns:
            True if this provider supports the task type
        """
        pass


class ConfigProviderInterface(ABC):
    """Interface for category-specific configuration building."""

    @abstractmethod
    def build_setup_steps(
        self,
        task_data: Dict[str, Any],
        s3_urls: Dict[str, str],
        files_created: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Build setup configuration steps for task initialization.

        Args:
            task_data: Original task data
            s3_urls: S3 URLs for uploaded files
            files_created: Information about created files

        Returns:
            List of setup step configurations
        """
        pass

    @abstractmethod
    def get_evaluation_mode(self, task_type: str, level: int) -> str:
        """
        Get the evaluation mode for a specific task type and level.

        Args:
            task_type: The type of task
            level: The difficulty level

        Returns:
            Evaluation mode string (e.g., 'compare_text_file', 'compare_answer')
        """
        pass

    @abstractmethod
    def supports_task_type(self, task_type: str) -> bool:
        """
        Check if this provider supports the given task type.

        Args:
            task_type: The type of task

        Returns:
            True if this provider supports the task type
        """
        pass


class EvaluationProviderInterface(ABC):
    """Interface for category-specific evaluation configuration."""

    @abstractmethod
    def build_evaluator_config(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        evaluation_mode: str,
        s3_urls: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Build evaluator configuration for the task.

        Args:
            task_data: Original task data
            files_created: Information about created files
            evaluation_mode: The evaluation mode to use
            s3_urls: S3 URLs for uploaded files

        Returns:
            Evaluator configuration dictionary
        """
        pass

    @abstractmethod
    def supports_task_type(self, task_type: str) -> bool:
        """
        Check if this provider supports the given task type.

        Args:
            task_type: The type of task

        Returns:
            True if this provider supports the task type
        """
        pass

    @abstractmethod
    def get_evaluator_instance(self):
        """
        Get the evaluator instance for this category.

        Returns:
            Category-specific evaluator instance
        """
        pass
