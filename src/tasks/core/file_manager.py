"""
File management for framework task generation.
Handles temporary file creation, S3 uploads, and file organization.
"""

import os
import tempfile
import shutil
import json
import subprocess
from typing import Dict, List, Any, Optional

from .utils import generate_s3_key, generate_timestamp_prefix


class FileManager:
    """
    Manages file operations for framework task generation.

    Handles:
    - Temporary file creation
    - File naming with UUID suffixes
    - S3 upload coordination
    - File cleanup
    """

    def __init__(self, s3_manager, bucket_name: str, base_s3_prefix: Optional[str] = None):
        """
        Initialize FileManager.

        Args:
            s3_manager: S3Manager instance for uploads
            bucket_name: S3 bucket name
            base_s3_prefix: Base S3 prefix, auto-generated if None
        """
        self.s3_manager = s3_manager
        self.bucket_name = bucket_name
        self.base_s3_prefix = base_s3_prefix or generate_timestamp_prefix()
        self.temp_dirs: List[str] = []

    def create_temp_directory(self, task_id: str) -> str:
        """
        Create a temporary directory for task files.

        Args:
            task_id: Task UUID

        Returns:
            Path to temporary directory
        """
        temp_dir = tempfile.mkdtemp(prefix=f"task_{task_id[:8]}_")
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def create_task_files(
        self,
        task_example: Dict[str, Any],
        task_id: str,
        temp_dir: str,
        file_provider=None,
    ) -> Dict[str, str]:
        """
        Create all files for a task in temporary directory.

        Args:
            task_example: Test example dictionary
            task_id: Task UUID
            temp_dir: Temporary directory path
            file_provider: Optional file provider for category-specific file creation

        Returns:
            Dictionary mapping file types to local file paths
        """
        files_created = {}

        # Use file provider for category-specific file creation if available
        if file_provider and file_provider.supports_task_type(task_example.get("task_type", "")):
            provider_files = file_provider.create_task_files(task_example, task_id, temp_dir)
            if provider_files:
                files_created.update(provider_files)
                # Still create additional files if needed
                additional_files = self._create_additional_files_if_needed(task_example, temp_dir)
                if additional_files:
                    files_created["additional_files"] = additional_files
                return files_created

        # Fallback to generic file creation logic (category-agnostic)
        # Create main file using generic text file approach
        main_filename = task_example["file_name"]
        main_file_path = os.path.join(temp_dir, main_filename)

        # Generic file creation (works for all categories)
        with open(main_file_path, "w", encoding="utf-8") as f:
            f.write(task_example.get("broken_file_content", ""))

        files_created["main_file"] = main_file_path
        files_created["main_filename"] = main_filename

        # Create correct file for evaluation reference (not uploaded)
        correct_filename = f"{os.path.splitext(main_filename)[0]}_correct{os.path.splitext(main_filename)[1]}"
        correct_file_path = os.path.join(temp_dir, correct_filename)

        with open(correct_file_path, "w", encoding="utf-8") as f:
            f.write(task_example["correct_file_content"])

        files_created["correct_file"] = correct_file_path
        files_created["correct_filename"] = correct_filename

        # Create additional files if present
        if "additional_files" in task_example:
            additional_files = {}
            for orig_filename, content in task_example["additional_files"].items():
                additional_file_path = os.path.join(temp_dir, orig_filename)

                with open(additional_file_path, "w", encoding="utf-8") as f:
                    f.write(content)

                additional_files[orig_filename] = {
                    "local_path": additional_file_path,
                    "filename": orig_filename,
                }

            files_created["additional_files"] = additional_files

        return files_created

    def _create_additional_files_if_needed(self, task_example: Dict[str, Any], temp_dir: str) -> Optional[Dict[str, Any]]:
        """Create additional files if present in task example."""
        if "additional_files" not in task_example:
            return None

        additional_files = {}
        for orig_filename, content in task_example["additional_files"].items():
            additional_file_path = os.path.join(temp_dir, orig_filename)

            with open(additional_file_path, "w", encoding="utf-8") as f:
                f.write(content)

            additional_files[orig_filename] = {
                "local_path": additional_file_path,
                "filename": orig_filename,
            }

        return additional_files

    # Note: _create_data_analysis_expected_files method has been removed.
    # Category-specific file creation is now handled by category file providers.

    def create_gold_standard_file(
        self,
        task_example: Dict[str, Any],
        task_id: str,
        temp_dir: str,
        evaluation_mode: str,
        file_provider=None,
    ) -> Optional[str]:
        """
        Create gold standard file for evaluation.

        Args:
            task_example: Test example dictionary
            task_id: Task UUID
            temp_dir: Temporary directory path
            evaluation_mode: Evaluation mode ('compare_text_file', 'compare_answer', 'compare_table')
            file_provider: Optional file provider for category-specific file creation

        Returns:
            Path to gold standard file if created, None otherwise
        """
        # Use file provider for category-specific file creation if available
        if file_provider and file_provider.supports_task_type(task_example.get("task_type", "")):
            files_created = file_provider.create_task_files(task_example, task_id, temp_dir)
            if files_created and "gold_standard_file" in files_created:
                return files_created["gold_standard_file"]

        if evaluation_mode != "compare_text_file":
            return None

        # Create correct Python file
        correct_filename = f"correct_{task_id[:8]}.py"
        correct_file_path = os.path.join(temp_dir, correct_filename)

        with open(correct_file_path, "w", encoding="utf-8") as f:
            f.write(task_example["correct_file_content"])

        # Run the correct file and capture output
        gold_standard_filename = "expected_output.txt"
        gold_standard_path = os.path.join(temp_dir, gold_standard_filename)

        try:
            # Change to temp directory and run Python file
            result = subprocess.run(
                [f"cd {temp_dir} && python3 {correct_filename}"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Write output (both stdout and stderr) to gold standard file
            output_content = result.stdout
            if result.stderr:
                output_content += result.stderr

            with open(gold_standard_path, "w", encoding="utf-8") as f:
                f.write(output_content)

            # Clean up the temporary correct file
            os.remove(correct_file_path)

            return gold_standard_path

        except subprocess.TimeoutExpired:
            print(f"Warning: Gold standard generation timed out for task {task_id}")
            return None
        except Exception as e:
            print(f"Warning: Failed to generate gold standard for task {task_id}: {e}")
            return None

    def upload_task_files(self, task_id: str, files_created: Dict[str, str]) -> Dict[str, str]:
        """
        Upload task files to S3.

        Args:
            task_id: Task UUID
            files_created: Dictionary of created files

        Returns:
            Dictionary mapping file types to S3 URLs
        """
        s3_urls = {}

        # Upload main file
        main_s3_key = generate_s3_key(self.base_s3_prefix, task_id, files_created["main_filename"])
        try:
            self.s3_manager.upload_file(
                Filename=files_created["main_file"],
                Bucket=self.bucket_name,
                Key=main_s3_key,
            )
            s3_urls["main_file"] = self.s3_manager.get_wget_link(self.bucket_name, main_s3_key)
        except Exception as e:
            raise Exception(f"Failed to upload main file: {e}")

        # Upload additional files if present
        if "additional_files" in files_created:
            s3_urls["additional_files"] = {}
            for orig_filename, file_info in files_created["additional_files"].items():
                additional_s3_key = generate_s3_key(self.base_s3_prefix, task_id, file_info["filename"])
                try:
                    self.s3_manager.upload_file(
                        Filename=file_info["local_path"],
                        Bucket=self.bucket_name,
                        Key=additional_s3_key,
                    )
                    s3_urls["additional_files"][orig_filename] = {
                        "url": self.s3_manager.get_wget_link(self.bucket_name, additional_s3_key),
                        "filename": file_info["filename"],
                    }
                except Exception as e:
                    raise Exception(f"Failed to upload additional file {orig_filename}: {e}")

        # Upload any additional files generically (category-agnostic approach)
        self._upload_additional_category_files(files_created, task_id, s3_urls)

        return s3_urls

    def _upload_additional_category_files(self, files_created: Dict[str, str], task_id: str, s3_urls: Dict[str, str]):
        """
        Upload any additional category-specific files generically.

        Args:
            files_created: Dictionary of created files
            task_id: Task UUID
            s3_urls: Dictionary to update with S3 URLs
        """
        # Define known additional file types that might be created by providers
        additional_file_types = [
            "expected_spreadsheet_file",
            "expected_output_file",
            "gold_standard_file",
            "expected_filename",
            "main_expected_file",
            "expected_converted_image",
            "expected_resized_image",
            "expected_filtered_image",
            "expected_document",
            "source_image",
            "document_template",
            # Integrated workflow specific files
            "research_html",
            "expected_processed_image",
            # Code ops specific ground truth files
            "expected_config_file",
            "expected_log_file",
            # Graphic document specific files
            "json_specification",
        ]

        for file_type in additional_file_types:
            if file_type in files_created:
                # Generate appropriate filename based on file type
                if file_type == "expected_spreadsheet_file":
                    filename = "expected_result.xlsx"
                    url_key = "expected_spreadsheet"
                elif file_type == "expected_output_file":
                    filename = "expected_output.txt"
                    url_key = "expected_output_file"
                elif file_type == "gold_standard_file":
                    # Preserve original file extension
                    local_file_path = files_created[file_type]
                    import os

                    original_extension = os.path.splitext(local_file_path)[1]
                    if original_extension == ".json":
                        filename = f"expected_presentation{original_extension}"
                    else:
                        # For document files, preserve the extension (.docx, .xlsx, etc.)
                        filename = f"expected_document{original_extension}"
                    url_key = "gold_standard_url"
                elif file_type == "main_expected_file":
                    filename = "main_expected.xlsx"
                    url_key = "main_expected"
                elif file_type == "expected_converted_image":
                    filename = "expected_converted.png"
                    url_key = "expected_converted_image"
                elif file_type == "expected_resized_image":
                    filename = "expected_resized.png"
                    url_key = "expected_resized_image"
                elif file_type == "expected_filtered_image":
                    filename = "expected_filtered.png"
                    url_key = "expected_filtered_image"
                elif file_type == "expected_document":
                    filename = "expected_document.docx"
                    url_key = "expected_document"
                elif file_type == "source_image":
                    # Preserve original image extension
                    local_file_path = files_created[file_type]
                    import os

                    original_extension = os.path.splitext(local_file_path)[1]
                    filename = f"source_image{original_extension}"
                    url_key = "source_image"
                elif file_type == "document_template":
                    filename = "document_template.docx"
                    url_key = "document_template"
                elif file_type == "research_html":
                    filename = "research.html"
                    url_key = "research_html"
                elif file_type == "expected_processed_image":
                    filename = "expected_processed_image.png"
                    url_key = "expected_processed_image"
                elif file_type == "expected_config_file":
                    filename = files_created.get("expected_config_filename", "expected_config.json")
                    url_key = "expected_config_file"
                elif file_type == "expected_log_file":
                    filename = files_created.get("expected_log_filename", "expected_log.txt")
                    url_key = "expected_log_file"
                elif file_type == "json_specification":
                    # Preserve original JSON filename for spec files
                    local_file_path = files_created[file_type]
                    import os

                    filename = os.path.basename(local_file_path)
                    url_key = "json_specification"
                else:
                    # Generic handling for unknown file types
                    filename = f"{file_type}_{task_id[:8]}.txt"
                    url_key = file_type

                s3_key = generate_s3_key(self.base_s3_prefix, task_id, filename)
                try:
                    self.s3_manager.upload_file(
                        Filename=files_created[file_type],
                        Bucket=self.bucket_name,
                        Key=s3_key,
                    )
                    s3_urls[url_key] = self.s3_manager.get_wget_link(self.bucket_name, s3_key)
                except Exception as e:
                    print(f"Warning: Failed to upload {file_type}: {e}")
                    # Don't raise exception for optional files

    def upload_config_file(self, task_id: str, config_data: Dict[str, Any]) -> str:
        """
        Upload framework configuration JSON to S3.

        Args:
            task_id: Task UUID
            config_data: Framework configuration dictionary

        Returns:
            S3 URL for configuration file
        """
        config_filename = f"task_config_{task_id[:8]}.json"

        # Create temporary config file
        temp_config_path = os.path.join(tempfile.gettempdir(), config_filename)
        with open(temp_config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        try:
            # Upload to S3
            config_s3_key = generate_s3_key(self.base_s3_prefix, task_id, config_filename)
            self.s3_manager.upload_file(Filename=temp_config_path, Bucket=self.bucket_name, Key=config_s3_key)

            # Get S3 URL
            s3_url = self.s3_manager.get_wget_link(self.bucket_name, config_s3_key)

            # Clean up temporary config file
            os.remove(temp_config_path)

            return s3_url

        except Exception as e:
            # Clean up temporary config file on error
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
            raise Exception(f"Failed to upload config file: {e}")

    def create_task_metadata(
        self,
        task_id: str,
        task_example: Dict[str, Any],
        s3_urls: Dict[str, str],
        config_url: str,
    ) -> Dict[str, Any]:
        """
        Create metadata for task generation tracking.

        Args:
            task_id: Task UUID
            task_example: Original test example
            s3_urls: S3 URLs for uploaded files
            config_url: S3 URL for configuration file

        Returns:
            Metadata dictionary
        """
        return {
            "task_id": task_id,
            "original_task_type": task_example["task_type"],
            "level": task_example["level"],
            "seed": task_example["seed"],
            "original_filename": task_example["file_name"],
            "s3_base_prefix": self.base_s3_prefix,
            "s3_urls": s3_urls,
            "config_url": config_url,
            "created_at": generate_timestamp_prefix(),
        }

    def cleanup_temp_directories(self):
        """Clean up all temporary directories created during session."""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up temp directory {temp_dir}: {e}")

        self.temp_dirs.clear()

    def get_s3_structure_info(self) -> Dict[str, str]:
        """
        Get information about S3 structure being used.

        Returns:
            Dictionary with S3 structure information
        """
        return {
            "bucket_name": self.bucket_name,
            "base_s3_prefix": self.base_s3_prefix,
            "full_s3_path": f"s3://{self.bucket_name}/{self.base_s3_prefix}/",
            "example_task_path": f"s3://{self.bucket_name}/{self.base_s3_prefix}/[task-uuid]/",
        }

    def validate_s3_connection(self) -> bool:
        """
        Validate S3 connection and permissions.

        Returns:
            True if S3 connection is valid
        """
        try:
            # Try to list objects in the bucket (limited to 1)
            self.s3_manager.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
            return True
        except Exception as e:
            print(f"S3 connection validation failed: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_temp_directories()
