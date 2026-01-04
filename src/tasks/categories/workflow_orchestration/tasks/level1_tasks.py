"""
Level 1 task implementations for integrated content workflow.
Level 1: Enhanced Professional Information Gathering & Document Creation
Enhanced with domain-based content generation for realistic professional scenarios.
"""

from typing import Dict, Any, Optional
from .base_task import WorkflowOrchestrationBaseTask


class BasicInfoGatheringTaskGenerator(WorkflowOrchestrationBaseTask):
    """
    Level 1: Enhanced Professional Information Gathering & Document Creation

    Applications: Chrome, LibreOffice Writer, GIMP
    Task: Research professional content online, extract key findings, process images for specific purposes,
          and create domain-appropriate documents with realistic business context.

    Enhanced Features:
    - 6 professional domains (corporate, academic, journalism, marketing, healthcare, financial)
    - 20+ companies per domain with realistic business contexts
    - 10+ research areas per domain
    - Professional HTML templates with authentic layouts
    - Contextual image processing requirements
    - Domain-appropriate document structures
    - Realistic file naming conventions
    - Professional instruction language
    """

    def __init__(self):
        super().__init__("basic_info_gathering", 1)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate enhanced Level 1 task data with professional domain context and massive variability."""
        if seed is not None:
            self.set_seed(seed)

        # Generate enhanced task structure with professional context
        enhanced_data = self.generate_enhanced_task_structure(seed)

        # Extract components from enhanced data
        domain_context = enhanced_data["domain_context"]
        scenario = enhanced_data["professional_scenario"]
        research_content = enhanced_data["research_content"]
        image_scenario = enhanced_data["image_scenario"]
        document_requirements = enhanced_data["document_requirements"]
        html_content = enhanced_data["html_content"]

        # Generate contextual filenames
        filenames = self.generate_contextual_filenames(domain_context, self.random.randint(1, 1000))

        # Generate professional instruction
        instruction_text = self.generate_professional_instruction(enhanced_data, filenames)

        # Generate task ID
        task_id = self.generate_task_id(f"L1_{domain_context['domain']}")

        # Create comprehensive task data structure
        task_data = {
            "task_id": task_id,
            "task_type": self.task_type,
            "level": self.level,
            "seed": seed or 0,
            # Required fields for framework validation
            "instructions": instruction_text,
            "file_name": filenames["research_filename"],
            "broken_file_content": "",  # Not applicable for integrated workflow
            "correct_file_content": "",  # Not applicable for integrated workflow
            "evaluation_method": "multi_evaluator",
            # Enhanced professional context
            "domain": domain_context["domain"],
            "company": domain_context["company"],
            "business_context": scenario["business_timeline"],
            "professional_scenario": scenario,
            "context_description": self.generate_professional_context_description(domain_context, scenario),
            # Research content with professional HTML
            "research_area": domain_context["research_area"],
            "research_content": research_content,
            "research_filename": filenames["research_filename"],
            "html_content": html_content,
            "key_finding": research_content["key_finding"],
            # Enhanced image processing with realistic context
            "image_scenario": image_scenario,
            "source_image_filename": filenames["image_filename"],
            "target_width": image_scenario["width"],
            "target_height": image_scenario["height"],
            "target_dimensions": f"{image_scenario['width']}x{image_scenario['height']}",
            "image_purpose": image_scenario["purpose"],
            "processing_context": image_scenario["context"],
            # Professional document requirements
            "document_requirements": document_requirements,
            "output_document": document_requirements["output_filename"],
            "processed_image": filenames["processed_image_filename"],
            "output_directory": "/home/user/Desktop",
            "document_structure": document_requirements["document_structure"],
            # Applications involved
            "applications": ["google-chrome", "libreoffice", "gimp"],
            # Enhanced evaluation data
            "expected_files": {
                "document_path": f"/home/user/Desktop/{document_requirements['output_filename']}",
                "image_path": f"/home/user/Desktop/{filenames['processed_image_filename']}",
            },
            # Professional metadata for tracking and analysis
            "domain_metadata": {
                "domain": domain_context["domain"],
                "company": domain_context["company"],
                "research_area": domain_context["research_area"],
                "document_type": domain_context["document_type"],
                "urgency_level": scenario["urgency_level"],
                "scope_complexity": scenario["scope_complexity"],
                "color_scheme": domain_context["color_scheme"],
            },
            # Content generation metrics (for analysis)
            "content_variety_metrics": {
                "estimated_variants": self.content_generator.get_content_variation_count(),
                "domain_combinations": len(self.content_generator.domains),
                "template_types": len(self.content_generator.html_templates),
                "image_scenarios": len(self.content_generator.image_scenarios),
            },
        }

        return task_data

    def get_enhanced_instruction_template(self, domain: str) -> str:
        """Get domain-specific instruction template for professional context."""
        templates = {
            "corporate_research": (
                "You are conducting a {project_type} for {company}. Extract the key research finding "
                "from the provided {research_area} analysis page, resize the source image to "
                "{target_dimensions} pixels for {image_purpose}, and create a professional "
                "{document_type} containing both the extracted finding and the processed image. "
                "This deliverable is for {business_timeline} and has {urgency_level} priority. "
                "Save the document as '{output_document}' and the processed image as '{processed_image}' on the Desktop."
            ),
            "academic_research": (
                "You are preparing a {project_type} for {institution}. Extract the key research finding "
                "from the provided {research_area} study page, resize the source image to "
                "{target_dimensions} pixels for {image_purpose}, and create an academic "
                "{document_type} containing both the extracted finding and the processed image. "
                "This work is for {business_timeline} submission. "
                "Save the document as '{output_document}' and the processed image as '{processed_image}' on the Desktop."
            ),
            "journalism_media": (
                "You are developing a {project_type} for {publication}. Extract the key finding "
                "from the provided {research_area} research page, resize the source image to "
                "{target_dimensions} pixels for {image_purpose}, and create a journalistic "
                "{document_type} containing both the extracted finding and the processed image. "
                "This piece is for {business_timeline} publication. "
                "Save the document as '{output_document}' and the processed image as '{processed_image}' on the Desktop."
            ),
            "marketing_communications": (
                "You are creating a {project_type} for {agency}. Extract the key insight "
                "from the provided {research_area} research page, resize the source image to "
                "{target_dimensions} pixels for {image_purpose}, and create a marketing "
                "{document_type} containing both the extracted insight and the processed image. "
                "This deliverable is for {business_timeline}. "
                "Save the document as '{output_document}' and the processed image as '{processed_image}' on the Desktop."
            ),
            "healthcare_research": (
                "You are preparing a {project_type} for {organization}. Extract the key medical finding "
                "from the provided {research_area} research page, resize the source image to "
                "{target_dimensions} pixels for {image_purpose}, and create a healthcare "
                "{document_type} containing both the extracted finding and the processed image. "
                "This documentation is for {business_timeline}. "
                "Save the document as '{output_document}' and the processed image as '{processed_image}' on the Desktop."
            ),
            "financial_services": (
                "You are conducting a {project_type} for {organization}. Extract the key financial finding "
                "from the provided {research_area} analysis page, resize the source image to "
                "{target_dimensions} pixels for {image_purpose}, and create a financial "
                "{document_type} containing both the extracted finding and the processed image. "
                "This analysis is for {business_timeline}. "
                "Save the document as '{output_document}' and the processed image as '{processed_image}' on the Desktop."
            ),
        }

        return templates.get(domain, templates["corporate_research"])

    def get_content_variability_summary(self) -> Dict[str, int]:
        """Get summary of content variability capabilities."""
        return {
            "total_domains": len(self.content_generator.domains),
            "companies_per_domain": sum(len(domain["companies"]) for domain in self.content_generator.domains.values())
            // len(self.content_generator.domains),
            "research_areas_per_domain": sum(len(domain["research_areas"]) for domain in self.content_generator.domains.values())
            // len(self.content_generator.domains),
            "document_types_per_domain": sum(len(domain["document_types"]) for domain in self.content_generator.domains.values())
            // len(self.content_generator.domains),
            "html_templates": len(self.content_generator.html_templates),
            "image_scenarios": len(self.content_generator.image_scenarios),
            "estimated_unique_variants": self.content_generator.get_content_variation_count(),
        }
