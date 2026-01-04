"""
Evaluator configuration logic for visual content integration category.
Contains the logic for building evaluator configurations, not running evaluations.
"""

from typing import Dict, Any


class ImageProcessingEvaluators:
    """Evaluator configuration builders for visual content integration tasks."""

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
        Build single evaluator configuration (not used for visual content tasks).
        All visual content tasks use multi-evaluator approach.
        """
        # Visual content integration always uses multi-evaluator
        return self.build_multi_evaluator_config(task_type, task_data, files_created, s3_urls)

    def build_multi_evaluator_config(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Build multi-evaluator configuration for visual content tasks.
        """
        if task_type == "basic_image_insertion":
            return self._build_basic_image_insertion_evaluator(task_data, files_created, s3_urls)
        elif task_type == "image_resize_insertion":
            return self._build_image_resize_insertion_evaluator(task_data, files_created, s3_urls)
        elif task_type == "image_modify_caption":
            return self._build_image_modify_caption_evaluator(task_data, files_created, s3_urls)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def needs_multi_evaluator(self, task_type: str) -> bool:
        """Check if task type requires multi-evaluator approach."""
        # All visual content tasks require multi-evaluator
        supported_tasks = {
            "basic_image_insertion",
            "image_resize_insertion",
            "image_modify_caption",
        }
        return task_type in supported_tasks

    def _build_basic_image_insertion_evaluator(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build evaluator for Level 1: Basic Image Insertion."""
        evaluation_data = task_data.get("evaluation_data", {})
        document_filename = evaluation_data.get("document_filename", "document.docx")

        config = {
            "postconfig": [
                {
                    "type": "activate_window",
                    "parameters": {
                        "window_name": "LibreOffice Writer",
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
                {"type": "sleep", "parameters": {"seconds": 1}},
            ],
            "func": "compare_docx_images",
            "expected": {
                "type": "cloud_file",
                "path": s3_urls.get("expected_document", "") if s3_urls else "",
                "dest": f"expected_{document_filename}",
            },
            "result": {
                "type": "vm_file",
                "path": f"/home/user/Desktop/{document_filename}",
                "dest": f"result_{document_filename}",
            },
            "options": {"debug": True},
        }

        return config

    def _build_image_resize_insertion_evaluator(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build evaluator for Level 2: Image Resize + Insertion."""
        evaluation_data = task_data.get("evaluation_data", {})
        resized_filename = evaluation_data.get("resized_filename", "resized.png")
        document_filename = evaluation_data.get("document_filename", "document.docx")
        expected_width = evaluation_data.get("expected_width", 400)
        expected_height = evaluation_data.get("expected_height", 300)

        config = {
            "postconfig": [
                {
                    "type": "activate_window",
                    "parameters": {
                        "window_name": "LibreOffice Writer",
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
                {"type": "sleep", "parameters": {"seconds": 1}},
            ],
            "func": ["check_image_size", "compare_docx_images"],
            "expected": [
                {
                    "type": "rule",
                    "rules": {"height": expected_height, "width": expected_width},
                },
                {
                    "type": "cloud_file",
                    "path": s3_urls.get("expected_document", "") if s3_urls else "",
                    "dest": f"expected_{document_filename}",
                },
            ],
            "result": [
                {
                    "type": "vm_file",
                    "path": f"/home/user/Desktop/{resized_filename}",
                    "dest": f"result_{resized_filename}",
                },
                {
                    "type": "vm_file",
                    "path": f"/home/user/Desktop/{document_filename}",
                    "dest": f"result_{document_filename}",
                },
            ],
            "options": [
                {},
                {"debug": True},
            ],
            "conj": "and",
        }

        return config

    def _build_image_modify_caption_evaluator(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build evaluator for Level 3: Image Modification + Caption."""
        evaluation_data = task_data.get("evaluation_data", {})
        modified_filename = evaluation_data.get("modified_filename", "modified.png")
        document_filename = evaluation_data.get("document_filename", "document.docx")
        modification_type = evaluation_data.get("modification_type", "resize")

        # Choose appropriate function based on modification type
        if modification_type == "grayscale":
            eval_function = "decrease_brightness"  # For grayscale detection
        else:  # resize
            eval_function = "check_image_size"  # For size validation

        # Generate JSON spec filename from document filename
        base_name = document_filename.replace(".docx", "")
        spec_filename = f"{base_name}_spec.json"

        config = {
            "postconfig": [
                {
                    "type": "activate_window",
                    "parameters": {
                        "window_name": "LibreOffice Writer",
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
                {"type": "sleep", "parameters": {"seconds": 1}},
            ],
            "func": [eval_function, "compare_docx_images", "evaluate_document_against_spec"],
            "expected": [
                {
                    "type": "rule" if modification_type == "resize" else "vm_file",
                    "rules": evaluation_data.get("modification_params", {}) if modification_type == "resize" else None,
                    "path": f"/home/user/Desktop/{task_data.get('source_image', 'source.jpg')}" if modification_type == "grayscale" else None,
                    "dest": f"original_{task_data.get('source_image', 'source.jpg')}" if modification_type == "grayscale" else None,
                },
                {
                    "type": "cloud_file",
                    "path": s3_urls.get("expected_document", "") if s3_urls else "",
                    "dest": f"expected_{document_filename}",
                },
                {
                    "type": "cloud_file",
                    "path": s3_urls.get("json_specification", "") if s3_urls else "",
                    "dest": f"spec_{spec_filename}",
                },
            ],
            "result": [
                {
                    "type": "vm_file",
                    "path": f"/home/user/Desktop/{modified_filename}",
                    "dest": f"result_{modified_filename}",
                },
                {
                    "type": "vm_file",
                    "path": f"/home/user/Desktop/{document_filename}",
                    "dest": f"result_{document_filename}",
                },
                {
                    "type": "vm_file",
                    "path": f"/home/user/Desktop/{document_filename}",
                    "dest": f"eval_{document_filename}",
                },
            ],
            "options": [
                {},
                {"debug": True},
                {"debug": True, "threshold": 0.75},
            ],
            "conj": "and",
        }

        return config
