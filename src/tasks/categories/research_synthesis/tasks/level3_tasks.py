"""
Level 3 task implementations for Information Synthesis & Presentation.
"""

from typing import Dict, Any, Optional
from .base_task import BaseResearchSynthesisTaskGenerator


class FileDownloadIntegrationGenerator(BaseResearchSynthesisTaskGenerator):
    """Generate dynamic file_download_integration tasks."""

    def __init__(self):
        super().__init__("file_download_integration", 3)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic file download integration task."""
        task_data = self.generate_basic_task_structure(seed)

        # Generate webpage content for Level 3
        webpage_data = self.generate_webpage_content(3, seed)

        # Generate webpage filename
        webpage_filename = f"webpage_{task_data['seed']}.html"

        # Create collaborative instructions
        instructions = self._generate_collaborative_instructions(
            task_data["presentation_file"],
            webpage_data["download_filename"],
            webpage_data["file_content"],
        )

        task_data.update(
            {
                "webpage_filename": webpage_filename,
                "webpage_content": webpage_data["html_content"],
                "download_filename": webpage_data["download_filename"],
                "file_content": webpage_data["file_content"],
                "file_description": webpage_data.get("file_description", webpage_data["download_filename"]),
                "expected_file_content": webpage_data["file_content"],
                "template_type": webpage_data["template_type"],
                "domain": webpage_data.get("domain", "professional_services"),
                "instructions": instructions,
                "target_url": f"http://localhost:8080/{webpage_filename}",
                "download_url": f"http://localhost:8080/files/{webpage_data['download_filename']}",
                "evaluation_data": {
                    "download_filename": webpage_data["download_filename"],
                    "file_content": webpage_data["file_content"],
                    "presentation_file": task_data["presentation_file"],
                },
                # Required by framework validation
                "broken_file_content": webpage_data["html_content"],
                "correct_file_content": webpage_data["html_content"],
                "evaluation_method": "multi_evaluator",
            }
        )

        return task_data

    def _generate_collaborative_instructions(self, presentation_file: str, download_filename: str, file_content: str) -> str:
        """Generate collaborative instructions for Level 3 tasks."""
        return f"Create a presentation slide containing the complete content from the text file available on the provided webpage. Save the completed presentation as '{presentation_file}'."
