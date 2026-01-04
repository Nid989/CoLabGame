"""
Level 1 task implementations for visual content integration: Basic Image Insertion.
"""

from typing import Dict, Any, Optional
from .base_task import ImageProcessingBaseTask


class BasicImageInsertionGenerator(ImageProcessingBaseTask):
    """Generate dynamic basic image insertion tasks."""

    def __init__(self):
        super().__init__("basic_image_insertion", 1)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic basic image insertion task."""
        task_data = self.generate_enhanced_task_structure(seed)

        # Extract scenario for variety but keep it simple
        professional_scenario = task_data["professional_scenario"]
        domain_context = task_data["domain_context"]

        # Use simplified filenames - keep variety but make them simple
        source_filename = task_data["source_image"]
        document_filename = task_data["document_file"]

        # Create simple instruction - no GIMP, just LibreOffice Writer
        instruction = (
            f"Create a document that contains the image '{source_filename}' by updating '{document_filename}' in LibreOffice Writer. "
            f"All required files are available on the Desktop. "
            f"Ensure the document is saved with the image properly inserted."
        )

        task_data.update(
            {
                "source_image": source_filename,
                "document_file": document_filename,
                "image_type": professional_scenario["professional_type"],
                "instructions": instruction,
                "image_size": (800, 600),  # Standard resolution
                "document_template_type": "simple_empty",  # Simple document template
                # Required fields for validation
                "file_name": document_filename,
                "broken_file_content": f"Empty document: {document_filename}",  # Source state
                "correct_file_content": f"Document with inserted image: {source_filename}",  # Expected result
                "evaluation_method": "multi_evaluator",
                "evaluation_data": {
                    "source_filename": source_filename,
                    "document_filename": document_filename,
                    "has_image_in_doc": True,
                    "requires_gimp": False,  # Level 1 doesn't need GIMP
                    "domain": domain_context["domain"],
                    "company": domain_context["company"],
                },
            }
        )

        return task_data
