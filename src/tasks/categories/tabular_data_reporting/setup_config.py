"""
Setup configuration logic for tabular data reporting category.
Contains the logic for building setup configuration steps for tasks.
"""

from typing import Dict, Any, List


class TabularDataReportingSetupConfig:
    """Configuration builder for task setup steps in the tabular data reporting category."""

    def __init__(self):
        """Initialize setup configuration builder."""
        pass

    def build_config_steps(
        self,
        task_example: Dict[str, Any],
        s3_urls: Dict[str, str],
        files_created: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Build configuration steps for task setup.

        Args:
            task_example: Original test example
            s3_urls: S3 URLs for uploaded files
            files_created: Local file information

        Returns:
            List of configuration steps
        """
        steps = []
        task_type = task_example["task_type"]

        if task_type == "simple_data_transfer":
            steps = self._build_level1_setup(task_example, s3_urls, files_created)
        elif task_type == "basic_data_aggregation":
            steps = self._build_level2_setup(task_example, s3_urls, files_created)
        elif task_type == "simple_calculation_output":
            steps = self._build_level3_setup(task_example, s3_urls, files_created)

        return steps

    def _build_level1_setup(
        self,
        task_example: Dict[str, Any],
        s3_urls: Dict[str, str],
        files_created: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Build setup steps for Level 1: Simple Data Transfer."""
        steps = []

        # Step 1: Create directory
        directory_path = task_example["directory_path"]
        steps.append(
            {
                "type": "command",
                "parameters": {"command": ["mkdir", "-p", directory_path]},
            }
        )

        # Step 2: Download data file (main file contains the data content)
        steps.append(
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {
                            "url": s3_urls["main_file"],
                            "path": f"{directory_path}/{task_example['data_filename']}",
                        }
                    ]
                },
            }
        )

        # Step 3: Launch LibreOffice Calc with blank spreadsheet
        steps.append({"type": "launch", "parameters": {"command": ["libreoffice", "--calc"]}})

        return steps

    def _build_level2_setup(
        self,
        task_example: Dict[str, Any],
        s3_urls: Dict[str, str],
        files_created: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Build setup steps for Level 2: Basic Data Aggregation."""
        steps = []

        # Step 1: Create directory
        directory_path = task_example["directory_path"]
        steps.append(
            {
                "type": "command",
                "parameters": {"command": ["mkdir", "-p", directory_path]},
            }
        )

        # Step 2: Download numbers file (main file contains the numbers content)
        steps.append(
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {
                            "url": s3_urls["main_file"],
                            "path": f"{directory_path}/{task_example['data_filename']}",
                        }
                    ]
                },
            }
        )

        # Step 3: Launch LibreOffice Calc with blank spreadsheet
        steps.append({"type": "launch", "parameters": {"command": ["libreoffice", "--calc"]}})

        return steps

    def _build_level3_setup(
        self,
        task_example: Dict[str, Any],
        s3_urls: Dict[str, str],
        files_created: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Build setup steps for Level 3: Simple Calculation and Output."""
        steps = []

        # Step 1: Download the main Excel data file
        main_filename = files_created.get("main_filename", "task_data.xlsx")
        steps.append(
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {
                            "url": s3_urls["main_file"],
                            "path": f"/home/user/Desktop/{main_filename}",
                        }
                    ]
                },
            }
        )

        # Step 2: Launch LibreOffice Calc with the Excel file
        steps.append(
            {
                "type": "launch",
                "parameters": {
                    "command": [
                        "libreoffice",
                        "--calc",
                        "--infilter=CSV:44,34,0,1,1",
                        f"/home/user/Desktop/{main_filename}",
                    ]
                },
            }
        )

        return steps
