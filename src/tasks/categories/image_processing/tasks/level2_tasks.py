"""
Level 2 task implementations for visual content integration: Image Resizing + Insertion.
"""

from typing import Dict, Any, Optional
from .base_task import ImageProcessingBaseTask


class ImageResizeInsertionGenerator(ImageProcessingBaseTask):
    """Generate dynamic image resizing and insertion tasks."""

    def __init__(self):
        super().__init__("image_resize_insertion", 2)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic image resizing and insertion task."""
        task_data = self.generate_enhanced_task_structure(seed)

        # Extract scenario for variety but keep it simple
        professional_scenario = task_data["professional_scenario"]
        domain_context = task_data["domain_context"]

        # Use simple, round number dimensions for easier completion
        dimension_options = [(400, 300), (600, 400), (800, 600), (500, 350)]
        target_width, target_height = self.random.choice(dimension_options)

        # Use simplified filenames
        source_filename = task_data["source_image"]

        # Generate simple resized filename - keep some variety but simple
        base_name = source_filename.split(".")[0]
        resized_filename = f"{base_name}_resized.png"
        document_filename = task_data["document_file"]

        # Create simple instruction
        instruction = (
            f"Create a document containing a properly sized image by processing '{source_filename}' to {target_width}×{target_height} pixels using GIMP, "
            f"saving it as '{resized_filename}', then incorporating this resized image into '{document_filename}' using LibreOffice Writer. "
            f"All required files are available on the Desktop. Ensure the document is saved with the resized image properly inserted."
        )

        task_data.update(
            {
                "source_image": source_filename,
                "resized_image": resized_filename,
                "document_file": document_filename,
                "target_width": target_width,
                "target_height": target_height,
                "image_type": professional_scenario["professional_type"],
                "instructions": instruction,
                "source_image_size": (1200, 900),  # Source resolution
                "document_template_type": "level2_with_content",  # Template with existing content
                # Required fields for validation
                "file_name": document_filename,
                "broken_file_content": f"Original image: {source_filename} (1200x900), Document with existing content: {document_filename}",  # Source state
                "correct_file_content": f"Resized image: {resized_filename} ({target_width}x{target_height}), Document with resized image",  # Expected result
                "evaluation_method": "multi_evaluator",
                "evaluation_data": {
                    "expected_width": target_width,
                    "expected_height": target_height,
                    "resized_filename": resized_filename,
                    "document_filename": document_filename,
                    "has_image_in_doc": True,
                    "requires_gimp": True,  # Level 2 needs GIMP for resizing
                    "domain": domain_context["domain"],
                    "company": domain_context["company"],
                },
            }
        )

        return task_data
