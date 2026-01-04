"""
Setup configuration logic for visual content integration category.
Contains the logic for building setup configuration steps for tasks.
"""

from typing import Dict, Any, List


class ImageProcessingSetupConfig:
    """Configuration builder for task setup steps in the visual content integration category."""

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

        # Step 1: Download files (Desktop directory already exists in Ubuntu)
        download_files = []

        # Download source image
        if "source_image" in s3_urls:
            source_image_filename = task_example.get("source_image", "source_image.png")
            source_image_path = f"/home/user/Desktop/{source_image_filename}"
            download_files.append({"url": s3_urls["source_image"], "path": source_image_path})

        # Download document template
        if "document_template" in s3_urls:
            document_filename = task_example.get("document_file", "document.docx")
            document_path = f"/home/user/Desktop/{document_filename}"
            download_files.append({"url": s3_urls["document_template"], "path": document_path})

        if download_files:
            steps.append({"type": "download", "parameters": {"files": download_files}})

        # Step 2: Launch GIMP with source image
        source_image_filename = task_example.get("source_image", "source_image.png")
        source_image_path = f"/home/user/Desktop/{source_image_filename}"

        steps.append({"type": "launch", "parameters": {"command": ["gimp", source_image_path]}})

        # Step 3: Launch LibreOffice Writer with document
        document_filename = task_example.get("document_file", "document.docx")
        document_path = f"/home/user/Desktop/{document_filename}"

        steps.append(
            {
                "type": "launch",
                "parameters": {"command": ["libreoffice", "--writer", document_path]},
            }
        )

        return steps
