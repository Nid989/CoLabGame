"""
Setup configuration logic for debugging and refactoring category.
Contains the logic for building setup configuration steps for tasks.
"""

from typing import Dict, Any, List
from src.tasks.core.utils import (
    get_directories_to_create,
    get_file_placement_path,
)


class DebuggingAndRefactoringSetupConfig:
    """Configuration builder for task setup steps in the debugging and refactoring category."""

    def __init__(self):
        """Initialize setup configuration builder."""
        pass

    def build_config_steps(
        self,
        task_example: Dict[str, Any],
        s3_urls: Dict[str, str],
        files_created: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Build configuration steps for task setup."""
        steps = []
        directories_to_create = get_directories_to_create(task_example["task_type"])
        for directory in directories_to_create:
            steps.append(
                {
                    "type": "command",
                    "parameters": {"command": ["mkdir", "-p", directory]},
                }
            )
        download_files = []
        main_file_vm_path = get_file_placement_path(task_example["task_type"], files_created["main_filename"])
        download_files.append({"url": s3_urls["main_file"], "path": main_file_vm_path})
        if "additional_files" in s3_urls:
            for orig_filename, file_info in s3_urls["additional_files"].items():
                additional_vm_path = get_file_placement_path(task_example["task_type"], file_info["filename"])
                download_files.append({"url": file_info["url"], "path": additional_vm_path})
        steps.append({"type": "download", "parameters": {"files": download_files}})
        return steps
