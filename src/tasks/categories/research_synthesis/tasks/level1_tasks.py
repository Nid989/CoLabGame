"""
Level 1 task implementations for Information Synthesis & Presentation.
"""

from typing import Dict, Any, Optional
from .base_task import BaseResearchSynthesisTaskGenerator


class BasicWebExtractionGenerator(BaseResearchSynthesisTaskGenerator):
    """Generate dynamic basic_web_extraction tasks."""

    def __init__(self):
        super().__init__("basic_web_extraction", 1)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic basic web extraction task."""
        task_data = self.generate_basic_task_structure(seed)

        # Generate webpage content for Level 1
        webpage_data = self.generate_webpage_content(1, seed)

        # Generate webpage filename
        webpage_filename = f"webpage_{task_data['seed']}.html"

        # Create collaborative instructions
        instructions = self._generate_collaborative_instructions(
            task_data["presentation_file"], webpage_data["target_text"], webpage_data["template_type"]
        )

        task_data.update(
            {
                "webpage_filename": webpage_filename,
                "webpage_content": webpage_data["html_content"],
                "target_text": webpage_data["target_text"],
                "expected_text": webpage_data["target_text"],
                "template_type": webpage_data["template_type"],
                "domain": webpage_data.get("domain", "professional_services"),
                "instructions": instructions,
                "target_url": f"http://localhost:8080/{webpage_filename}",
                "evaluation_data": {
                    "target_text": webpage_data["target_text"],
                    "presentation_file": task_data["presentation_file"],
                },
                # Required by framework validation
                "broken_file_content": webpage_data["html_content"],
                "correct_file_content": webpage_data["html_content"],
                "evaluation_method": "multi_evaluator",
            }
        )

        return task_data

    def _generate_collaborative_instructions(self, presentation_file: str, target_text: str, template_type: str = None) -> str:
        """Generate collaborative instructions for Level 1 tasks."""
        # Create task-specific questions based on target text type and template
        if "Founded" in target_text or template_type == "company_info":
            question = "What year was the company founded?"
        elif "Price:" in target_text or template_type == "product_info":
            question = "What is the product price?"
        elif "Date:" in target_text or template_type == "event_info":
            question = "What is the event date?"
        elif "Phone:" in target_text:
            question = "What is the phone number?"
        elif "Email:" in target_text:
            question = "What is the email address?"
        else:
            question = "Find the highlighted information on the webpage"

        return f"Create a presentation slide that answers '{question}' using information from the provided webpage. Save the completed presentation as '{presentation_file}'."
