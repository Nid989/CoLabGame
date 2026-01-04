"""
Main framework task generator.
Orchestrates the conversion of test examples to framework-compatible configurations.
"""

import json
import os
from typing import Dict, List, Any, Optional

from .file_manager import FileManager
from .config_builder import ConfigBuilder
from .utils import (
    generate_task_instance_id,
    validate_test_example,
    generate_timestamp_prefix,
)


class FrameworkTaskGenerator:
    """
    Main class for generating framework-compatible task configurations.

    Converts test examples from test_examples.json format to framework schema format,
    handles file creation and S3 upload, and generates complete configuration files.
    """

    def __init__(self, s3_manager, bucket_name: str, base_s3_prefix: Optional[str] = None):
        """
        Initialize FrameworkTaskGenerator.

        Args:
            s3_manager: S3Manager instance for file uploads
            bucket_name: S3 bucket name
            base_s3_prefix: Base S3 prefix, auto-generated if None
        """
        self.s3_manager = s3_manager
        self.bucket_name = bucket_name
        self.base_s3_prefix = base_s3_prefix or generate_timestamp_prefix()

        # Initialize components
        self.file_manager = FileManager(s3_manager, bucket_name, self.base_s3_prefix)
        self.config_builder = ConfigBuilder()

        # Track generated tasks
        self.generated_tasks: List[Dict[str, Any]] = []

    def generate_dynamic_task(
        self,
        task_type: str,
        category: str = "debugging_and_refactoring",
        level: int = None,
        seed: Optional[int] = None,
        upload_to_s3: bool = True,
        evaluation_mode: str = "compare_text_file",
    ) -> Dict[str, Any]:
        """
        Generate a dynamic task using task implementations.

        Args:
            task_type: Type of task to generate
            category: Category name (defaults to "debugging_and_refactoring")
            level: Difficulty level (optional, for validation)
            seed: Optional seed for reproducible generation
            upload_to_s3: Whether to upload files to S3
            evaluation_mode: Evaluation method to use

        Returns:
            Dictionary containing task configuration and metadata
        """
        # Import category registry
        from src.tasks.core.category_registry import category_registry

        try:
            # Get the category
            category_obj = category_registry.get_category(category)
            if not category_obj:
                raise ValueError(f"Unknown category: {category}")

            # Validate task type exists in category
            task_types = category_obj.get_task_types()
            if task_type not in task_types:
                raise ValueError(f"Task type '{task_type}' not found in category '{category}'")

            # Get the appropriate task implementation
            generator = category_obj.get_task_implementation(task_type)

            # Generate dynamic task data
            task_data = generator.generate_task_data(seed)

            # Generate the task using the internal pipeline
            return self._generate_task_from_data(task_data, upload_to_s3, evaluation_mode, category_obj)

        except Exception as e:
            raise Exception(f"Failed to generate dynamic task {task_type} in category {category}: {str(e)}")

    def _generate_task_from_data(
        self,
        task_data: Dict[str, Any],
        upload_to_s3: bool = True,
        evaluation_mode: str = "compare_text_file",
        category_obj=None,
    ) -> Dict[str, Any]:
        """
        Internal method to generate framework task configuration from task data.

        Args:
            task_data: Task data dictionary (from dynamic task implementations)
            upload_to_s3: Whether to upload files to S3
            evaluation_mode: Evaluation method to use ('compare_text_file' or 'compare_answer')
            category_obj: Category object for evaluators (optional for backward compatibility)

        Returns:
            Dictionary containing task configuration and metadata
        """
        # Get category providers if category_obj is available
        file_provider = None
        config_provider = None
        evaluation_provider = None

        if category_obj:
            try:
                file_provider = category_obj.get_file_provider()
                config_provider = category_obj.get_config_provider()
                evaluation_provider = category_obj.get_evaluation_provider()
            except AttributeError:
                # Fallback for categories that don't support providers yet
                pass
        # Use evaluation mode from task data if available, otherwise use provided default
        if "evaluation_mode" in task_data:
            evaluation_mode = task_data["evaluation_mode"]
        # Validate input
        if not validate_test_example(task_data):
            raise ValueError("Invalid task data: missing required fields")

        # Generate task ID
        task_id = generate_task_instance_id()

        try:
            # Create temporary directory for files
            temp_dir = self.file_manager.create_temp_directory(task_id)

            # Create all task files using file provider
            files_created = self.file_manager.create_task_files(task_data, task_id, temp_dir, file_provider)

            # Create gold standard file for evaluation using file provider
            gold_standard_path = self.file_manager.create_gold_standard_file(task_data, task_id, temp_dir, evaluation_mode, file_provider)
            if gold_standard_path:
                files_created["gold_standard_file"] = gold_standard_path

            # Category-specific file creation is now handled by file providers
            # Legacy code for data analysis expected files has been moved to providers

            # Upload files to S3 if requested
            s3_urls = {}
            if upload_to_s3:
                s3_urls = self.file_manager.upload_task_files(task_id, files_created)
            else:
                # Generate local file URLs for testing
                s3_urls = self._generate_local_urls(files_created)

            # Build framework configuration using providers
            framework_config = self.config_builder.build_config(
                task_id,
                task_data,
                s3_urls,
                files_created,
                evaluation_mode,
                category_obj,
                config_provider,
                evaluation_provider,
            )

            # Validate generated configuration
            if not self.config_builder.validate_config(framework_config):
                raise ValueError("Generated configuration failed validation")

            # Upload configuration to S3 if requested
            config_url = ""
            if upload_to_s3:
                config_url = self.file_manager.upload_config_file(task_id, framework_config)

            # Create task metadata
            task_metadata = self.file_manager.create_task_metadata(task_id, task_data, s3_urls, config_url)

            # Create result object
            result = {
                "task_id": task_id,
                "framework_config": framework_config,
                "metadata": task_metadata,
                "files_created": files_created,
                "s3_urls": s3_urls,
                "config_url": config_url,
                "success": True,
            }

            # Track generated task
            self.generated_tasks.append(result)

            return result

        except Exception as e:
            # Clean up on error
            self.file_manager.cleanup_temp_directories()
            raise Exception(f"Failed to generate task {task_id}: {str(e)}")

    def save_generated_configs(self, output_dir: str, results: List[Dict[str, Any]]) -> str:
        """
        Save generated configurations to local files.

        Args:
            output_dir: Directory to save configurations
            results: List of task generation results

        Returns:
            Path to summary file
        """
        os.makedirs(output_dir, exist_ok=True)

        configs_dir = os.path.join(output_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)

        summary_data = {
            "generation_info": {
                "total_tasks": len(results),
                "s3_bucket": self.bucket_name,
                "s3_prefix": self.base_s3_prefix,
                "generated_at": generate_timestamp_prefix(),
            },
            "tasks": [],
        }

        # Save individual configurations
        for result in results:
            if result["success"]:
                task_id = result["task_id"]
                config_filename = f"{task_id}.json"
                config_path = os.path.join(configs_dir, config_filename)

                # Save framework config
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(result["framework_config"], f, indent=2)

                # Add to summary
                summary_data["tasks"].append(
                    {
                        "task_id": task_id,
                        "config_file": config_filename,
                        "task_type": result["metadata"]["original_task_type"],
                        "level": result["metadata"]["level"],
                        "config_url": result["config_url"],
                    }
                )

        # Save summary
        summary_path = os.path.join(output_dir, "generation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        print(f"Saved {len(results)} configurations to {output_dir}")
        return summary_path

    def get_generation_summary(self) -> Dict[str, Any]:
        """
        Get summary of current generation session.

        Returns:
            Summary information dictionary
        """
        return {
            "total_generated": len(self.generated_tasks),
            "successful": len([t for t in self.generated_tasks if t["success"]]),
            "s3_info": self.file_manager.get_s3_structure_info(),
            "task_types": list(set([t["metadata"]["original_task_type"] for t in self.generated_tasks if t["success"]])),
            "levels": list(set([t["metadata"]["level"] for t in self.generated_tasks if t["success"]])),
        }

    def validate_s3_setup(self) -> bool:
        """
        Validate S3 setup and connectivity.

        Returns:
            True if S3 setup is valid
        """
        return self.file_manager.validate_s3_connection()

    def cleanup(self):
        """Clean up temporary resources."""
        self.file_manager.cleanup_temp_directories()

    def _generate_local_urls(self, files_created: Dict[str, str]) -> Dict[str, str]:
        """
        Generate local file URLs for testing (when not uploading to S3).

        Args:
            files_created: Dictionary of created files

        Returns:
            Dictionary with local file URLs
        """
        urls = {}

        # Main file
        urls["main_file"] = f"file://{files_created['main_file']}"

        # Category-specific primary inputs if present
        if "source_image" in files_created:
            urls["source_image"] = f"file://{files_created['source_image']}"
        if "document_template" in files_created:
            urls["document_template"] = f"file://{files_created['document_template']}"

        # Additional files
        if "additional_files" in files_created:
            urls["additional_files"] = {}
            for orig_filename, file_info in files_created["additional_files"].items():
                urls["additional_files"][orig_filename] = {
                    "url": f"file://{file_info['local_path']}",
                    "filename": file_info["filename"],
                }

        # Gold standard file
        if "gold_standard_file" in files_created:
            urls["gold_standard_url"] = f"file://{files_created['gold_standard_file']}"
        # Expected document (for evaluators that expect this key)
        if "expected_document" in files_created:
            urls["expected_document"] = f"file://{files_created['expected_document']}"

        # Expected spreadsheet file for data analysis tasks
        if "expected_spreadsheet_file" in files_created:
            urls["expected_spreadsheet"] = f"file://{files_created['expected_spreadsheet_file']}"

        # Expected output file for Level 3 data analysis tasks
        if "expected_output_file" in files_created:
            urls["expected_output_file"] = f"file://{files_created['expected_output_file']}"

        return urls

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
