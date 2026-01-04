"""
Evaluator configuration logic for Information Synthesis & Presentation category.
Contains the logic for building multi-evaluator configurations, not running evaluations.
"""

from typing import Dict, Any


class ResearchSynthesisEvaluators:
    """Evaluator configuration builders for Information Synthesis & Presentation tasks."""

    def __init__(self):
        """Initialize evaluator configuration builder."""
        pass

    def build_multi_evaluator_config(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Build multi-evaluator configuration.

        Args:
            task_type: Type of task
            task_data: Task data from test example
            files_created: Information about created files
            s3_urls: S3 URLs for uploaded files

        Returns:
            Multi-evaluator configuration
        """
        if task_type == "basic_web_extraction":
            return self._build_level1_evaluator(task_data, files_created, s3_urls)
        elif task_type == "multi_point_summary":
            return self._build_level2_evaluator(task_data, files_created, s3_urls)
        elif task_type == "file_download_integration":
            return self._build_level3_evaluator(task_data, files_created, s3_urls)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _build_level1_evaluator(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build evaluator for Level 1: basic_web_extraction task."""
        presentation_file = task_data["presentation_file"]
        presentation_path = f"/home/user/Documents/{presentation_file}"

        config = {
            "postconfig": [
                {
                    "type": "activate_window",
                    "parameters": {
                        "window_name": "LibreOffice Impress",
                        "strict": False,
                    },
                },
                {
                    "type": "execute",
                    "parameters": {
                        "command": [
                            "python3",
                            "-c",
                            "import pyautogui; pyautogui.hotkey('ctrl', 's');",
                        ]
                    },
                },
                {"type": "sleep", "parameters": {"seconds": 2}},
            ],
            "func": "evaluate_presentation_against_spec",
            "result": {
                "type": "vm_file",
                "path": presentation_path,
                "dest": "result_presentation.pptx",
            },
            "expected": {
                "type": "cloud_file",
                "path": s3_urls.get("gold_standard_url", ""),
                "dest": "expected_presentation.json",
            },
            "options": {
                "text_threshold": 0.8,
            },
        }

        return config

    def _build_level2_evaluator(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build evaluator for Level 2: multi_point_summary task."""
        presentation_file = task_data["presentation_file"]
        presentation_path = f"/home/user/Documents/{presentation_file}"

        config = {
            "postconfig": [
                {
                    "type": "activate_window",
                    "parameters": {
                        "window_name": "LibreOffice Impress",
                        "strict": False,
                    },
                },
                {
                    "type": "execute",
                    "parameters": {
                        "command": [
                            "python3",
                            "-c",
                            "import pyautogui; pyautogui.hotkey('ctrl', 's');",
                        ]
                    },
                },
                {"type": "sleep", "parameters": {"seconds": 2}},
            ],
            "func": "evaluate_presentation_against_spec",
            "result": {
                "type": "vm_file",
                "path": presentation_path,
                "dest": "result_presentation.pptx",
            },
            "expected": {
                "type": "cloud_file",
                "path": s3_urls.get("gold_standard_url", ""),
                "dest": "expected_presentation.json",
            },
            "options": {
                "text_threshold": 0.8,
            },
        }

        return config

    def _build_level3_evaluator(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build evaluator for Level 3: file_download_integration task."""
        presentation_file = task_data["presentation_file"]
        presentation_path = f"/home/user/Documents/{presentation_file}"

        config = {
            "postconfig": [
                {
                    "type": "activate_window",
                    "parameters": {
                        "window_name": "LibreOffice Impress",
                        "strict": False,
                    },
                },
                {
                    "type": "execute",
                    "parameters": {
                        "command": [
                            "python3",
                            "-c",
                            "import pyautogui; pyautogui.hotkey('ctrl', 's');",
                        ]
                    },
                },
                {"type": "sleep", "parameters": {"seconds": 2}},
            ],
            "func": "evaluate_presentation_against_spec",
            "result": {
                "type": "vm_file",
                "path": presentation_path,
                "dest": "result_presentation.pptx",
            },
            "expected": {
                "type": "cloud_file",
                "path": s3_urls.get("gold_standard_url", ""),
                "dest": "expected_presentation.json",
            },
            "options": {
                "text_threshold": 0.8,
            },
        }

        return config

    def needs_multi_evaluator(self, task_type: str) -> bool:
        """
        Determine if a task type needs multi-evaluator validation.
        All Information Synthesis & Presentation tasks use multi-evaluator.

        Args:
            task_type: Type of task

        Returns:
            True if multi-evaluator needed, False otherwise
        """
        supported_types = {
            "basic_web_extraction",
            "multi_point_summary",
            "file_download_integration",
        }
        return task_type in supported_types
