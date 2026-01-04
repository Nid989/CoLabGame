"""
Consolidated providers for debugging and refactoring category.
Handles file placement, directory creation, setup configuration, and evaluation logic.
"""

import os
from typing import Dict, List, Any, Optional

from ..base import (
    FileProviderInterface,
    ConfigProviderInterface,
    EvaluationProviderInterface,
)
from .evaluators import DebuggingAndRefactoringEvaluators


class DebuggingAndRefactoringFileProvider(FileProviderInterface):
    """File provider implementation for debugging and refactoring tasks."""

    def __init__(self):
        """Initialize debugging and refactoring file provider."""
        self.supported_tasks = {
            "basic_python_syntax_fix",
            "simple_logic_completion",
            "multi_file_config_update",
        }
        self.file_placement_mapping = {
            "basic_python_syntax_fix": "/home/user/coding_tasks/{filename}",
            "simple_logic_completion": "/home/user/coding_tasks/{filename}",
            "multi_file_config_update": "/home/user/coding_tasks/{filename}",
        }
        self.directory_creation_mapping = {
            "basic_python_syntax_fix": ["/home/user/coding_tasks"],
            "simple_logic_completion": ["/home/user/coding_tasks"],
            "multi_file_config_update": ["/home/user/coding_tasks"],
        }

    def get_file_placement_path(self, task_type: str, filename: str) -> str:
        """Get the file placement path for a specific task type."""
        if not self.supports_task_type(task_type):
            raise ValueError(f"Unsupported task type: {task_type}")
        path_template = self.file_placement_mapping[task_type]
        return path_template.format(filename=filename)

    def get_directories_to_create(self, task_type: str) -> List[str]:
        """Get list of directories that need to be created for a task type."""
        if not self.supports_task_type(task_type):
            return []
        return self.directory_creation_mapping.get(task_type, [])

    def create_task_files(self, task_data: Dict[str, Any], task_id: str, temp_dir: str) -> Optional[Dict[str, str]]:
        """Create debugging and refactoring specific task files."""
        task_type = task_data.get("task_type")
        if not self.supports_task_type(task_type):
            return None
        files_created = {}
        main_filename = task_data.get("file_name", f"task_{task_id[:8]}.py")
        main_file_path = os.path.join(temp_dir, main_filename)
        broken_content = task_data.get("broken_file_content", "")
        with open(main_file_path, "w") as f:
            f.write(broken_content)
        files_created["main_file"] = main_file_path
        files_created["main_filename"] = main_filename
        if task_type == "simple_logic_completion":
            ground_truth_files = self._create_simple_logic_completion_ground_truth(task_data, task_id, temp_dir)
            if ground_truth_files:
                files_created.update(ground_truth_files)
                files_created["ground_truth_files"] = ground_truth_files
        elif task_type == "multi_file_config_update":
            ground_truth_files = self._create_multi_file_config_update_ground_truth(task_data, task_id, temp_dir)
            if ground_truth_files:
                files_created.update(ground_truth_files)
                files_created["ground_truth_files"] = ground_truth_files
        return files_created

    def supports_task_type(self, task_type: str) -> bool:
        """Check if this provider supports the given task type."""
        return task_type in self.supported_tasks

    def _create_simple_logic_completion_ground_truth(self, task_data: Dict[str, Any], task_id: str, temp_dir: str) -> Optional[Dict[str, str]]:
        """Create ground truth files for simple logic completion task."""
        ground_truth_files = {}
        ground_truth_dir = os.path.join(temp_dir, "ground_truth")
        os.makedirs(ground_truth_dir, exist_ok=True)
        expected_file_content = task_data.get("expected_file_content", "")
        if expected_file_content:
            output_filename = task_data.get("output_filename", "results.txt")
            ground_truth_filename = f"expected_{output_filename}"
            ground_truth_path = os.path.join(ground_truth_dir, ground_truth_filename)
            with open(ground_truth_path, "w") as f:
                f.write(expected_file_content)
            ground_truth_files["expected_output_file"] = ground_truth_path
            ground_truth_files["expected_output_filename"] = ground_truth_filename
        return ground_truth_files if ground_truth_files else None

    def _create_multi_file_config_update_ground_truth(self, task_data: Dict[str, Any], task_id: str, temp_dir: str) -> Optional[Dict[str, str]]:
        """Create ground truth files for multi-file config update task."""
        ground_truth_files = {}
        ground_truth_dir = os.path.join(temp_dir, "ground_truth")
        os.makedirs(ground_truth_dir, exist_ok=True)
        expected_config_content = task_data.get("expected_config_content", "")
        if expected_config_content:
            config_filename = task_data.get("config_filename", "app_config.json")
            expected_config_filename = f"expected_{config_filename}"
            expected_config_path = os.path.join(ground_truth_dir, expected_config_filename)
            with open(expected_config_path, "w") as f:
                f.write(expected_config_content)
            ground_truth_files["expected_config_file"] = expected_config_path
            ground_truth_files["expected_config_filename"] = expected_config_filename
        expected_log_content = task_data.get("expected_log_content", "")
        if expected_log_content:
            log_filename = task_data.get("log_filename", "system_log.txt")
            expected_log_filename = f"expected_{log_filename}"
            expected_log_path = os.path.join(ground_truth_dir, expected_log_filename)
            with open(expected_log_path, "w") as f:
                f.write(expected_log_content)
            ground_truth_files["expected_log_file"] = expected_log_path
            ground_truth_files["expected_log_filename"] = expected_log_filename
        return ground_truth_files if ground_truth_files else None


class DebuggingAndRefactoringConfigProvider(ConfigProviderInterface):
    """Config provider implementation for debugging and refactoring tasks."""

    def __init__(self):
        """Initialize debugging and refactoring config provider."""
        self.supported_tasks = {
            "basic_python_syntax_fix",
            "simple_logic_completion",
            "multi_file_config_update",
        }
        self.evaluation_mode_mapping = {
            "basic_python_syntax_fix": "compare_answer",
            "simple_logic_completion": "multi_evaluator",
            "multi_file_config_update": "multi_evaluator",
        }

    def build_setup_steps(
        self,
        task_data: Dict[str, Any],
        s3_urls: Dict[str, str],
        files_created: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Build setup configuration steps for task initialization."""
        task_type = task_data["task_type"]
        if not self.supports_task_type(task_type):
            return []
        steps = []
        file_provider = DebuggingAndRefactoringFileProvider()
        directories = file_provider.get_directories_to_create(task_type)
        for directory in directories:
            steps.append(
                {
                    "type": "command",
                    "parameters": {"command": ["mkdir", "-p", directory]},
                }
            )
        if s3_urls.get("main_file"):
            main_filename = files_created.get("main_filename", "main.py")
            file_path = file_provider.get_file_placement_path(task_type, main_filename)
            steps.append(
                {
                    "type": "download",
                    "parameters": {"files": [{"url": s3_urls["main_file"], "path": file_path}]},
                }
            )
        if "additional_files" in files_created and s3_urls.get("additional_files"):
            additional_downloads = []
            for orig_filename, file_info in files_created["additional_files"].items():
                if orig_filename in s3_urls["additional_files"]:
                    file_path = file_provider.get_file_placement_path(task_type, file_info["filename"])
                    additional_downloads.append(
                        {
                            "url": s3_urls["additional_files"][orig_filename]["url"],
                            "path": file_path,
                        }
                    )
            if additional_downloads:
                steps.append({"type": "download", "parameters": {"files": additional_downloads}})
        steps.append({"type": "sleep", "parameters": {"seconds": 10.0}})
        return steps

    def get_evaluation_mode(self, task_type: str, level: int) -> str:
        """Get the evaluation mode for a specific task type and level."""
        if not self.supports_task_type(task_type):
            return "compare_text_file" if level == 1 else "compare_answer"
        return self.evaluation_mode_mapping.get(task_type, "compare_answer")

    def supports_task_type(self, task_type: str) -> bool:
        """Check if this provider supports the given task type."""
        return task_type in self.supported_tasks


class DebuggingAndRefactoringEvaluationProvider(EvaluationProviderInterface):
    """Evaluation provider implementation for debugging and refactoring tasks."""

    def __init__(self):
        """Initialize debugging and refactoring evaluation provider."""
        self.supported_tasks = {
            "basic_python_syntax_fix",
            "simple_logic_completion",
            "multi_file_config_update",
        }
        self.evaluators = DebuggingAndRefactoringEvaluators()

    def build_evaluator_config(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        evaluation_mode: str,
        s3_urls: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build evaluator configuration for the task."""
        task_type = task_data.get("task_type")
        if not self.supports_task_type(task_type):
            return {}
        task_data_with_mode = {**task_data, "evaluation_mode": evaluation_mode}
        if self.evaluators.needs_multi_evaluator(task_type):
            return self.evaluators.build_multi_evaluator_config(task_type, task_data_with_mode, files_created, s3_urls)
        else:
            return self.evaluators.build_single_evaluator_config(task_type, task_data_with_mode, files_created, s3_urls)

    def supports_task_type(self, task_type: str) -> bool:
        """Check if this provider supports the given task type."""
        return task_type in self.supported_tasks

    def get_evaluator_instance(self):
        """Get the evaluator instance for this category."""
        return self.evaluators


__all__ = [
    "DebuggingAndRefactoringFileProvider",
    "DebuggingAndRefactoringConfigProvider",
    "DebuggingAndRefactoringEvaluationProvider",
]
