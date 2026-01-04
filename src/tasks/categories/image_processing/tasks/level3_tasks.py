"""
Level 3 task implementations for visual content integration: Image Modification + Caption.
"""

from typing import Dict, Any, Optional
from .base_task import ImageProcessingBaseTask


class ImageModifyCaptionGenerator(ImageProcessingBaseTask):
    """Generate dynamic image modification and caption tasks."""

    def __init__(self):
        super().__init__("image_modify_caption", 3)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic image modification and caption task."""
        task_data = self.generate_enhanced_task_structure(seed)

        # Extract scenario for variety but keep it simple
        professional_scenario = task_data["professional_scenario"]
        domain_context = task_data["domain_context"]

        # Choose ONE simple modification: Resize OR Grayscale
        modification_options = ["resize", "grayscale"]
        modification_type = self.random.choice(modification_options)

        source_filename = task_data["source_image"]
        base_name = source_filename.split(".")[0]

        # Generate simple modified filename and parameters
        if modification_type == "resize":
            # Simple resize dimensions
            dimension_options = [(400, 300), (600, 400), (500, 350)]
            target_width, target_height = self.random.choice(dimension_options)
            modified_filename = f"{base_name}_resized.png"
            modification_instruction = f"sizing it to {target_width}×{target_height} pixels"
            modification_params = {"width": target_width, "height": target_height}
        else:  # grayscale
            modified_filename = f"{base_name}_grayscale.png"
            modification_instruction = "converting it to grayscale"
            modification_params = {"filter": "grayscale"}

        document_filename = task_data["document_file"]

        # Generate professional caption text using domain context
        caption_text = self.generate_caption_text(domain=domain_context["domain"], context=domain_context)

        # Create simple instruction
        instruction = (
            f"Create a complete document with a modified image and caption by processing '{source_filename}' in GIMP to {modification_instruction}, "
            f"saving it as '{modified_filename}', then incorporating this modified image into '{document_filename}' using LibreOffice Writer with the caption text '{caption_text}' below the image. "
            f"All required files are available on the Desktop. Ensure the document is saved with the modified image and caption properly placed."
        )

        task_data.update(
            {
                "source_image": source_filename,
                "modified_image": modified_filename,
                "document_file": document_filename,
                "modification_type": modification_type,
                "modification_params": modification_params,
                "caption_text": caption_text,
                "image_type": professional_scenario["professional_type"],
                "instructions": instruction,
                "source_image_size": (1200, 900),  # Source resolution
                "document_template_type": "level3_with_content",  # Template with existing content
                # Required fields for validation
                "file_name": document_filename,
                "broken_file_content": f"Original image: {source_filename} (1200x900), Document with existing content: {document_filename}",  # Source state
                "correct_file_content": f"Modified image: {modified_filename} ({modification_type}), Document with image and caption: {caption_text}",  # Expected result
                "evaluation_method": "multi_evaluator",
                "evaluation_data": {
                    "modification_type": modification_type,
                    "modification_params": modification_params,
                    "expected_caption": caption_text,
                    "modified_filename": modified_filename,
                    "document_filename": document_filename,
                    "has_image_in_doc": True,
                    "has_caption": True,
                    "modification_applied": True,
                    "requires_gimp": True,  # Level 3 needs GIMP for modification
                    "domain": domain_context["domain"],
                    "company": domain_context["company"],
                },
            }
        )

        return task_data
