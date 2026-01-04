"""
Evaluator configuration logic for tabular data reporting category.
Contains the logic for building evaluator configurations, not running evaluations.
"""

from typing import Dict, Any


class TabularDataReportingEvaluators:
    """Evaluator configuration builders for tabular data reporting tasks."""

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
        Build single evaluator configuration for data analysis tasks.

        Args:
            task_type: Type of task
            task_data: Task data from test example
            files_created: Information about created files
            s3_urls: S3 URLs for uploaded files

        Returns:
            Single evaluator configuration
        """
        if task_type == "simple_data_transfer":
            return self._build_level1_evaluator(task_data, files_created, s3_urls)
        elif task_type == "basic_data_aggregation":
            return self._build_level2_evaluator(task_data, files_created, s3_urls)
        elif task_type == "simple_calculation_output":
            return self._build_level3_evaluator(task_data, files_created, s3_urls)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _build_level1_evaluator(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build evaluator for Level 1: Simple Data Transfer."""
        evaluation_data = task_data["evaluation_data"]

        return {
            "postconfig": [{"type": "sleep", "parameters": {"seconds": 2.0}}],
            "func": "compare_table",
            "result": {
                "type": "vm_file",
                "path": "/home/user/Desktop/result.xlsx",
                "dest": "result.xlsx",
            },
            "expected": {
                "type": "cloud_file",
                "path": s3_urls["expected_spreadsheet"] if s3_urls else "",
                "dest": "expected_result.xlsx",
            },
            "options": {
                "debug": True,
                "rules": [
                    {
                        "type": "check_cell",
                        "sheet_idx": 0,
                        "coordinate": evaluation_data["target_cell"],
                        "props": {
                            "value": {
                                "method": "eq",
                                "ref": evaluation_data["expected_value"],
                            }
                        },
                    }
                ],
            },
        }

    def _build_level2_evaluator(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build evaluator for Level 2: Basic Data Aggregation."""
        evaluation_data = task_data["evaluation_data"]

        return {
            "postconfig": [{"type": "sleep", "parameters": {"seconds": 2.0}}],
            "func": "compare_table",
            "result": {
                "type": "vm_file",
                "path": "/home/user/Desktop/result.xlsx",
                "dest": "result.xlsx",
            },
            "expected": {
                "type": "cloud_file",
                "path": s3_urls["expected_spreadsheet"] if s3_urls else "",
                "dest": "expected_result.xlsx",
            },
            "options": {
                "debug": True,
                "rules": [
                    {
                        "type": "check_cell",
                        "sheet_idx": 0,
                        "coordinate": evaluation_data["target_cell"],
                        "props": {
                            "value": {
                                "method": "approx:0.01",
                                "ref": evaluation_data["expected_value"],
                            }
                        },
                    }
                ],
            },
        }

    def _build_level3_evaluator(
        self,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """Build evaluator for Level 3: Simple Calculation and Output."""
        evaluation_data = task_data["evaluation_data"]

        return {
            "postconfig": [{"type": "sleep", "parameters": {"seconds": 2.0}}],
            "func": ["compare_table", "exact_match"],
            "result": [
                {
                    "type": "vm_file",
                    "path": "/home/user/Desktop/result.xlsx",
                    "dest": "result.xlsx",
                },
                {
                    "type": "vm_command_line",
                    "command": "cat /home/user/Desktop/output.txt",
                    "shell": True,
                },
            ],
            "expected": [
                {
                    "type": "cloud_file",
                    "path": s3_urls["expected_spreadsheet"] if s3_urls else "",
                    "dest": "expected_result.xlsx",
                },
                {
                    "type": "rule",
                    "rules": {"expected": f"{evaluation_data['spreadsheet_expected_value']}\n"},
                },
            ],
            "options": [
                {
                    "debug": True,
                    "rules": [
                        {
                            "type": "check_cell",
                            "sheet_idx": 0,
                            "coordinate": evaluation_data["spreadsheet_target_cell"],
                            "props": {
                                "value": {
                                    "method": "approx:0.1",
                                    "ref": evaluation_data["spreadsheet_expected_value"],
                                }
                            },
                        }
                    ],
                },
                {},
            ],
            "conj": "and",
        }

    def build_multi_evaluator_config(
        self,
        task_type: str,
        task_data: Dict[str, Any],
        files_created: Dict[str, str],
        s3_urls: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Build multi-evaluator configuration.
        Only Level 3 uses multi-evaluator in data analysis category.
        """
        if task_type == "simple_calculation_output":
            return self._build_level3_evaluator(task_data, files_created, s3_urls)
        else:
            # Fallback to single evaluator
            return self.build_single_evaluator_config(task_type, task_data, files_created, s3_urls)

    def needs_multi_evaluator(self, task_type: str) -> bool:
        """
        Determine if a task type needs multi-evaluator validation.

        Args:
            task_type: Type of task

        Returns:
            True if multi-evaluator needed, False otherwise
        """
        # Only Level 3 task needs multi-evaluator (spreadsheet + output file)
        level_3_multi = ["simple_calculation_output"]

        return task_type in level_3_multi
