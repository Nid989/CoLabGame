"""
Consolidated providers for research synthesis category.
Handles file placement, directory creation, setup configuration, and evaluation logic.
"""

import os
from typing import Dict, List, Any, Optional

from ..base import (
    FileProviderInterface,
    ConfigProviderInterface,
    EvaluationProviderInterface,
)
from .evaluators import ResearchSynthesisEvaluators


class ResearchSynthesisFileProvider(FileProviderInterface):
    """File provider implementation for research synthesis tasks."""

    def __init__(self):
        """Initialize file provider."""
        pass

    def create_main_file(self, task_data: Dict[str, Any], task_id: str, temp_dir: str) -> str:
        """Create the main HTML webpage file for the task."""
        webpage_filename = task_data["webpage_filename"]
        webpage_content = task_data["webpage_content"]
        filepath = os.path.join(temp_dir, webpage_filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(webpage_content)
        return filepath

    def create_additional_files(self, task_data: Dict[str, Any], task_id: str, temp_dir: str) -> Dict[str, Dict[str, str]]:
        """Create additional files needed for the task."""
        additional_files = {}
        if task_data.get("level") == 3 and "file_content" in task_data:
            download_filename = task_data["download_filename"]
            file_content = task_data["file_content"]
            files_dir = os.path.join(temp_dir, "files")
            os.makedirs(files_dir, exist_ok=True)
            filepath = os.path.join(files_dir, download_filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(file_content)
            additional_files[download_filename] = {
                "local_path": filepath,
                "filename": download_filename,
                "content_type": "text/plain",
            }
        return additional_files

    def create_gold_standard_file(
        self,
        task_data: Dict[str, Any],
        task_id: str,
        temp_dir: str,
        evaluation_mode: str,
    ) -> Optional[str]:
        """Create expected presentation specification JSON file for evaluation."""
        from .tasks.base_task import BaseResearchSynthesisTaskGenerator

        temp_generator = BaseResearchSynthesisTaskGenerator(task_data["task_type"], task_data["level"])
        expected_filepath = temp_generator.create_expected_spec(task_data, temp_dir)
        return expected_filepath

    def get_main_filename(self, task_data: Dict[str, Any]) -> str:
        """Get the main file filename."""
        return task_data["webpage_filename"]

    def supports_task_type(self, task_type: str) -> bool:
        """Check if this provider supports the given task type."""
        supported_types = {
            "basic_web_extraction",
            "multi_point_summary",
            "file_download_integration",
        }
        return task_type in supported_types

    def create_task_files(self, task_data: Dict[str, Any], task_id: str, temp_dir: str) -> Dict[str, str]:
        """Create all task files for the given task."""
        files_created = {}
        main_file_path = self.create_main_file(task_data, task_id, temp_dir)
        files_created["main_file"] = main_file_path
        files_created["main_filename"] = task_data["webpage_filename"]
        additional_files = self.create_additional_files(task_data, task_id, temp_dir)
        if additional_files:
            files_created["additional_files"] = additional_files
        gold_standard_path = self.create_gold_standard_file(task_data, task_id, temp_dir, "multi_evaluator")
        if gold_standard_path:
            files_created["gold_standard_file"] = gold_standard_path
        return files_created

    def get_directories_to_create(self, task_type: str) -> list:
        """Get directories that need to be created for this task type."""
        return []

    def get_file_placement_path(self, task_type: str, filename: str) -> str:
        """Get the file placement path for a given task type and filename."""
        if filename.endswith(".pptx"):
            return f"/home/user/Desktop/{filename}"
        else:
            return f"/tmp/{filename}"


class ResearchSynthesisConfigProvider(ConfigProviderInterface):
    """Config provider implementation for research synthesis tasks."""

    def __init__(self):
        """Initialize config provider."""
        self.supported_tasks = {
            "basic_web_extraction",
            "multi_point_summary",
            "file_download_integration",
        }
        self.evaluation_mode_mapping = {
            "basic_web_extraction": "multi_evaluator",
            "multi_point_summary": "multi_evaluator",
            "file_download_integration": "multi_evaluator",
        }

    def build_setup_steps(
        self,
        task_data: Dict[str, Any],
        s3_urls: Dict[str, str],
        files_created: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Build setup configuration steps for task initialization."""
        task_type = task_data["task_type"]
        level = task_data.get("level", 1)
        if not self.supports_task_type(task_type):
            return []
        steps = []
        if "main_file" in s3_urls:
            steps.append(
                {
                    "type": "download",
                    "parameters": {
                        "files": [
                            {
                                "url": s3_urls["main_file"],
                                "path": f"/tmp/{task_data['webpage_filename']}",
                            }
                        ]
                    },
                }
            )
        if level == 3 and "additional_files" in s3_urls:
            additional_downloads = []
            for orig_filename, file_info in s3_urls["additional_files"].items():
                additional_downloads.append(
                    {
                        "url": file_info["url"],
                        "path": f"/tmp/files/{file_info['filename']}",
                    }
                )
            if additional_downloads:
                steps.append(
                    {
                        "type": "command",
                        "parameters": {"command": ["mkdir", "-p", "/tmp/files"]},
                    }
                )
                steps.append({"type": "download", "parameters": {"files": additional_downloads}})
        steps.append(
            {
                "type": "execute",
                "parameters": {
                    "command": [
                        "python3",
                        "-c",
                        "import subprocess, os; "
                        "log=open('/tmp/http_server.log','a'); "
                        "p=subprocess.Popen(['python3','-m','http.server','8080','--directory','/tmp'], stdout=log, stderr=log, preexec_fn=os.setsid); "
                        "open('/tmp/http_server.pid','w').write(str(p.pid)); "
                        "print('Server started on port 8080')",
                    ]
                },
            }
        )
        steps.append({"type": "sleep", "parameters": {"seconds": 2}})
        target_url = f"http://localhost:8080/{task_data['webpage_filename']}"
        steps.append(
            {
                "type": "launch",
                "parameters": {"command": ["google-chrome", "--new-window", target_url]},
            }
        )
        steps.append({"type": "launch", "parameters": {"command": ["libreoffice", "--impress"]}})
        steps.append(
            {
                "type": "activate_window",
                "parameters": {"window_name": "Google Chrome", "strict": False},
            }
        )
        steps.append({"type": "sleep", "parameters": {"seconds": 10.0}})
        return steps

    def get_evaluation_mode(self, task_type: str, level: int) -> str:
        """Get evaluation mode for the given task type and level."""
        return self.evaluation_mode_mapping.get(task_type, "multi_evaluator")

    def supports_task_type(self, task_type: str) -> bool:
        """Check if this provider supports the given task type."""
        return task_type in self.supported_tasks


class ResearchSynthesisEvaluationProvider(EvaluationProviderInterface):
    """Evaluation provider implementation for research synthesis tasks."""

    def __init__(self):
        """Initialize evaluation provider."""
        self.supported_tasks = {
            "basic_web_extraction",
            "multi_point_summary",
            "file_download_integration",
        }
        self.evaluators = ResearchSynthesisEvaluators()

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
        return self.evaluators.build_multi_evaluator_config(task_type, task_data_with_mode, files_created, s3_urls)

    def supports_task_type(self, task_type: str) -> bool:
        """Check if this provider supports the given task type."""
        return task_type in self.supported_tasks

    def get_evaluator_instance(self):
        """Get the evaluator instance for this category."""
        return self.evaluators


__all__ = [
    "ResearchSynthesisFileProvider",
    "ResearchSynthesisConfigProvider",
    "ResearchSynthesisEvaluationProvider",
]
