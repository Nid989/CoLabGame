"""
Level 2 task implementations for Information Synthesis & Presentation.
"""

from typing import Dict, Any, Optional
from .base_task import BaseResearchSynthesisTaskGenerator


class MultiPointSummaryGenerator(BaseResearchSynthesisTaskGenerator):
    """Generate dynamic multi_point_summary tasks."""

    def __init__(self):
        super().__init__("multi_point_summary", 2)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic multi-point summary task."""
        task_data = self.generate_basic_task_structure(seed)

        # Generate webpage content for Level 2
        webpage_data = self.generate_webpage_content(2, seed)

        # Generate webpage filename
        webpage_filename = f"webpage_{task_data['seed']}.html"

        # Create collaborative instructions
        instructions = self._generate_collaborative_instructions(
            task_data["presentation_file"], webpage_data["target_texts"], webpage_data["template_type"]
        )

        task_data.update(
            {
                "webpage_filename": webpage_filename,
                "webpage_content": webpage_data["html_content"],
                "target_texts": webpage_data["target_texts"],
                "expected_texts": webpage_data["target_texts"],
                "template_type": webpage_data["template_type"],
                "domain": webpage_data.get("domain", "professional_services"),
                "instructions": instructions,
                "target_url": f"http://localhost:8080/{webpage_filename}",
                "evaluation_data": {
                    "target_texts": webpage_data["target_texts"],
                    "presentation_file": task_data["presentation_file"],
                    "expected_slide_count": 2,
                },
                # Required by framework validation
                "broken_file_content": webpage_data["html_content"],
                "correct_file_content": webpage_data["html_content"],
                "evaluation_method": "multi_evaluator",
            }
        )

        return task_data

    def _generate_collaborative_instructions(self, presentation_file: str, target_texts: list, template_type: str = None) -> str:
        """Generate collaborative instructions for Level 2 tasks."""
        # Create task-specific questions based on target text types and template
        if template_type == "product_catalog" or ("Product:" in target_texts[0] and "Price:" in target_texts[1]):
            task_desc = "identify the product name and price, creating two slides: placing the product name on the first slide and the price on the second slide"
        elif template_type == "contact_info" or ("Phone:" in target_texts[0] and "Email:" in target_texts[1]):
            task_desc = "identify the contact phone number and email address, creating two slides: placing the phone number on the first slide and the email address on the second slide"
        else:
            task_desc = "identify the two key pieces of information shown on the webpage, creating two slides: placing the first piece of information on the first slide and the second piece on the second slide"

        return f"Create a two-slide presentation where you {task_desc} from the provided webpage. Save the completed presentation as '{presentation_file}'."
