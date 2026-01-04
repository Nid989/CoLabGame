"""
Evaluator configuration logic for integrated content workflow category.
Contains the logic for building evaluator configurations, not running evaluations.
"""

from typing import Dict, Any, List


class WorkflowOrchestrationEvaluators:
    """Evaluator configuration builders for integrated workflow tasks."""

    def __init__(self):
        """Initialize evaluator configuration builder."""
        pass

    def build_single_evaluator_config(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Build single evaluator configuration.

        Args:
            task_type: Type of task
            task_data: Task data from test example
            files_created: Information about created files
            s3_urls: S3 URLs for uploaded files

        Returns:
            Single evaluator configuration
        """
        # For integrated workflow, we primarily use multi-evaluator configurations
        # This method serves as a fallback for simple cases
        return self._build_document_content_config(task_data, files_created, s3_urls)

    def build_multi_evaluator_config(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Build multi-evaluator configuration for integrated workflow tasks.

        Args:
            task_type: Type of task
            task_data: Task data from test example
            files_created: Information about created files
            s3_urls: S3 URLs for uploaded files

        Returns:
            Multi-evaluator configuration
        """
        if task_type == "basic_info_gathering":
            return self._build_basic_info_gathering_multi_evaluator(task_data, files_created, s3_urls)

        # Fallback to single evaluator
        return self.build_single_evaluator_config(task_type, task_data, files_created, s3_urls)

    def _build_basic_info_gathering_multi_evaluator(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build multi-evaluator config for basic information gathering task."""

        # Get expected file URLs
        expected_document_url = s3_urls.get("expected_document", "") if s3_urls else ""

        # Build the multi-evaluator configuration
        config = {
            "postconfig": [
                {
                    "type": "activate_window",
                    "parameters": {"window_name": "LibreOffice Writer", "strict": False},
                },
                {
                    "type": "execute",
                    "parameters": {
                        "command": [
                            "python3",
                            "-c",
                            "import pyautogui; pyautogui.hotkey('ctrl', 's')",
                        ]
                    },
                },
                {"type": "sleep", "parameters": {"seconds": 2}},
            ],
            "func": [
                "compare_docx_images",
                "compare_answer",
                "check_image_size",
                "evaluate_document_against_spec",
            ],
            "result": [
                {  # result document for compare_docx_images
                    "type": "vm_file",
                    "path": task_data.get("expected_files", {}).get(
                        "document_path",
                        "/home/user/Desktop/" + task_data.get("output_document", "report.docx"),
                    ),
                    "dest": "result_document.docx",
                },
                {  # expected answer string for compare_answer
                    "type": "rule",
                    "rules": {"answer": "Exists"},
                },
                {  # processed image for size check
                    "type": "vm_file",
                    "path": task_data.get("expected_files", {}).get(
                        "image_path",
                        "/home/user/Desktop/" + task_data.get("processed_image", "processed_image.png"),
                    ),
                    "dest": "result_image.png",
                },
                {  # document for spec evaluation
                    "type": "vm_file",
                    "path": task_data.get("expected_files", {}).get(
                        "document_path",
                        "/home/user/Desktop/" + task_data.get("output_document", "report.docx"),
                    ),
                    "dest": "eval_document.docx",
                },
            ],
            "expected": [
                {  # cloud docx for compare_docx_images
                    "type": "cloud_file",
                    "path": expected_document_url,
                    "dest": "expected_document.docx",
                },
                {  # vm command to produce the answer
                    "type": "vm_command_line",
                    "command": f"test -f {task_data.get('expected_files', {}).get('image_path', '/home/user/Desktop/' + task_data.get('processed_image', 'processed_image.png'))} && echo 'Exists' || echo 'Missing'",
                },
                {  # size rule
                    "type": "rule",
                    "rules": {
                        "width": task_data.get("target_width", 800),
                        "height": task_data.get("target_height", 600),
                    },
                },
                {  # cloud json spec for evaluate_document_against_spec
                    "type": "cloud_file",
                    "path": s3_urls.get("json_specification", "") if s3_urls else "",
                    "dest": "expected_spec.json",
                },
            ],
            "options": [
                {"debug": True},  # compare_docx_images
                {},  # check_file_exists
                {},  # check_image_size
                {"debug": True, "threshold": 0.75},  # evaluate_document_against_spec
            ],
            "conj": "and",
        }

        return config

    def _build_document_content_config(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build simple document content comparison configuration."""
        expected_document_url = s3_urls.get("expected_document", "") if s3_urls else ""

        return {
            "func": "compare_docx_files",
            "expected": {"type": "cloud_file", "path": expected_document_url, "dest": "expected_document.docx"},
            "result": {
                "type": "vm_file",
                "path": task_data.get("expected_files", {}).get("document_path", "/home/user/Desktop/report.docx"),
                "dest": "result_document.docx",
            },
            "options": {"ignore_blanks": True, "examine_embedded_images": True},
        }

    def get_supported_metrics(self) -> List[str]:
        """Get list of supported evaluation metrics for this category."""
        return [
            "compare_docx_images",
            "check_file_exists",
            "check_image_size",
            "evaluate_document_against_spec",
        ]

    def get_metrics_for_task_type(self, task_type: str) -> List[str]:
        """Get specific metrics used for a task type."""
        metrics_mapping = {
            "basic_info_gathering": [
                "compare_docx_images",
                "compare_answer",
                "check_image_size",
                "evaluate_document_against_spec",
            ]
        }

        return metrics_mapping.get(task_type, [])

    def validate_evaluator_config(self, config: Dict[str, Any]) -> bool:
        """Validate an evaluator configuration."""
        # Check for multi-evaluator format
        if "func" in config and isinstance(config["func"], list):
            required_fields = ["func", "result", "expected"]
            for field in required_fields:
                if field not in config:
                    return False

            # Check array lengths match
            if len(config["func"]) != len(config["result"]) or len(config["func"]) != len(config["expected"]):
                return False

        # Check for single evaluator format
        elif "func" in config and isinstance(config["func"], str):
            if "result" not in config or "expected" not in config:
                return False

        else:
            return False

        return True

    def get_postconfig_actions(self, task_type: str) -> List[Dict[str, Any]]:
        """Get post-configuration actions for a task type."""
        actions = {
            "basic_info_gathering": [
                {"type": "activate_window", "parameters": {"window_name": "LibreOffice Writer", "strict": False}},
                {"type": "execute", "parameters": {"command": ["python3", "-c", "import pyautogui; pyautogui.hotkey('ctrl', 's')"]}},
                {"type": "sleep", "parameters": {"seconds": 2}},
            ]
        }

        return actions.get(task_type, [])
