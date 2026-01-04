"""
Evaluator configuration logic for debugging and refactoring category.
Contains the logic for building evaluator configurations, not running evaluations.
"""

from typing import Dict, Any


class DebuggingAndRefactoringEvaluators:
    """Evaluator configuration builders for debugging and refactoring tasks."""

    def __init__(self):
        """Initialize evaluator configuration builder."""
        self.file_placement_mapping = {
            "basic_python_syntax_fix": "/home/user/coding_tasks/{filename}",
            "simple_logic_completion": "/home/user/coding_tasks/{filename}",
            "multi_file_config_update": "/home/user/coding_tasks/{filename}",
        }

    def _get_file_placement_path(self, task_type: str, filename: str) -> str:
        """Get the file placement path for a specific task type and filename."""
        path_template = self.file_placement_mapping.get(task_type)
        if path_template:
            return path_template.format(filename=filename)
        return f"/home/user/Desktop/{filename}"

    def build_single_evaluator_config(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build single evaluator configuration."""
        evaluation_mode = task_data.get("evaluation_mode", "compare_text_file")
        if evaluation_mode == "compare_text_file":
            return self._build_compare_text_file_config(task_data, files_created, s3_urls)
        else:
            return self._build_compare_answer_config(task_data, files_created, s3_urls)

    def build_multi_evaluator_config(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build multi-evaluator configuration."""
        if task_type == "simple_logic_completion":
            return self._build_simple_logic_completion_evaluator(task_data, files_created, s3_urls)
        elif task_type == "multi_file_config_update":
            return self._build_multi_file_config_update_evaluator(task_data, files_created, s3_urls)
        else:
            raise ValueError(f"Unknown multi-evaluator task type: {task_type}")

    def _build_compare_text_file_config(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build compare_text_file evaluator configuration with output comparison."""
        level_2_tasks = [
            "fix_multiple_errors",
            "complete_simple_function",
            "update_config_file",
            "add_output_statement",
        ]
        is_level_2 = task_data["task_type"] in level_2_tasks
        if is_level_2:
            command = [
                "/bin/bash",
                "-c",
                f"cd /home/user/Desktop/coding_tasks && python3 {files_created['main_filename']} > output.txt 2>&1",
            ]
            output_path = "/home/user/Desktop/coding_tasks/output.txt"
        else:
            command = [
                "/bin/bash",
                "-c",
                f"cd /home/user/Desktop && python3 {files_created['main_filename']} > output.txt 2>&1",
            ]
            output_path = "/home/user/Desktop/output.txt"
        return {
            "postconfig": [{"type": "execute", "parameters": {"command": command}}],
            "func": "compare_text_file",
            "expected": {
                "type": "cloud_file",
                "path": s3_urls.get("gold_standard_url", "") if s3_urls else "",
                "dest": "expected_output.txt",
            },
            "result": {
                "type": "vm_file",
                "path": output_path,
                "dest": "actual_output.txt",
            },
        }

    def _build_compare_answer_config(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build compare_answer evaluator configuration."""
        main_file_path = self._get_file_placement_path(task_data["task_type"], files_created["main_filename"])
        if task_data["task_type"] in [
            "basic_python_syntax_fix",
            "basic_shell_syntax_fix",
        ]:
            if task_data["task_type"] == "basic_shell_syntax_fix":
                command = [
                    "/bin/bash",
                    "-c",
                    f"cd /home/user/coding_tasks && bash {files_created['main_filename']}",
                ]
            else:
                command = [
                    "/bin/bash",
                    "-c",
                    f"cd /home/user/coding_tasks && python {files_created['main_filename']}",
                ]
        else:
            command = ["python", main_file_path]
        return {
            "func": "compare_answer",
            "expected": {
                "type": "rule",
                "rules": {"expected": task_data["expected_output"]},
            },
            "result": {"type": "vm_command_line", "command": command},
        }

    def _build_simple_logic_completion_evaluator(
        self, task_data: Dict[str, Any], files_created: Dict[str, str], s3_urls: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Build evaluator for simple_logic_completion task: exact_match + compare_text_file."""
        output_filename = task_data.get("output_filename", "results.txt")
        output_file_path = self._get_file_placement_path(task_data["task_type"], output_filename)
        expected_output_url = s3_urls.get("expected_output_file", "") if s3_urls else ""
        expected_output_filename = files_created.get("expected_output_filename", "expected_output.txt")
        return {
            "func": ["exact_match", "compare_files"],
            "result": [
                {
                    "type": "vm_command_line",
                    "command": f"test -f {output_file_path} && echo 'Exists' || echo 'Not Found'",
                    "shell": True,
                },
                {
                    "type": "vm_file",
                    "path": output_file_path,
                    "dest": "actual_output.txt",
                },
            ],
            "expected": [
                {"type": "rule", "rules": {"expected": "Exists"}},
                {
                    "type": "cloud_file",
                    "path": expected_output_url,
                    "dest": expected_output_filename,
                },
            ],
            "conj": "and",
        }

    def _build_multi_file_config_update_evaluator(
        self, task_data: Dict[str, Any], files_created: Dict[str, str], s3_urls: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Build evaluator for multi_file_config_update task: config verification + execution + log file."""
        config_filename = task_data.get("config_filename", "app_config.json")
        log_filename = task_data.get("log_filename", "system_log.txt")
        config_file_path = self._get_file_placement_path(task_data["task_type"], config_filename)
        log_file_path = self._get_file_placement_path(task_data["task_type"], log_filename)
        expected_config_url = s3_urls.get("expected_config_file", "") if s3_urls else ""
        expected_log_url = s3_urls.get("expected_log_file", "") if s3_urls else ""
        expected_config_filename = files_created.get("expected_config_filename", "expected_config.json")
        expected_log_filename = files_created.get("expected_log_filename", "expected_log.txt")
        config_file_type = "json" if config_filename.endswith(".json") else "ini"
        return {
            "func": ["compare_files", "exact_match", "compare_files"],
            "result": [
                {
                    "type": "vm_file",
                    "path": config_file_path,
                    "dest": "actual_config.json",
                },
                {
                    "type": "vm_command_line",
                    "command": f"test -f {log_file_path} && echo 'Exists' || echo 'Not Found'",
                    "shell": True,
                },
                {"type": "vm_file", "path": log_file_path, "dest": "actual_log.txt"},
            ],
            "expected": [
                {
                    "type": "cloud_file",
                    "path": expected_config_url,
                    "dest": expected_config_filename,
                },
                {"type": "rule", "rules": {"expected": "Exists"}},
                {
                    "type": "cloud_file",
                    "path": expected_log_url,
                    "dest": expected_log_filename,
                },
            ],
            "options": [
                {
                    "file_type": config_file_type,
                },
                {},
                {"file_type": "text", "ignore_blanks": True},
            ],
            "conj": "and",
        }

    def needs_multi_evaluator(self, task_type: str) -> bool:
        """Determine if a task type needs multi-evaluator validation."""
        multi_evaluator_tasks = [
            "simple_logic_completion",
            "multi_file_config_update",
        ]
        return task_type in multi_evaluator_tasks
