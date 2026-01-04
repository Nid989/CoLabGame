"""
Base generator class for integrated content workflow tasks.
Enhanced with domain-based content generation for realistic professional scenarios.
"""

import random
from abc import abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from ...base import BaseTask
from ..enhanced_content_generator import WorkflowOrchestrationContentGenerator


class WorkflowOrchestrationBaseTask(BaseTask):
    """Base class for integrated content workflow task implementations with enhanced content generation."""

    def __init__(self, task_type: str, level: int):
        super().__init__()
        self.task_type = task_type
        self.level = level
        self.random = random.Random()

        # Initialize enhanced content generator
        self.content_generator = WorkflowOrchestrationContentGenerator()

    def set_seed(self, seed: int):
        """Set random seed for reproducible generation."""
        self.random.seed(seed)
        self.content_generator.set_seed(seed)

    def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate a task instance (implemented by calling generate_task_data)."""
        seed = kwargs.get("seed")
        if seed is not None:
            self.set_seed(seed)
        return self.generate_task_data(seed)

    @abstractmethod
    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic task data."""
        pass

    def get_task_type(self) -> str:
        """Return the task type identifier."""
        return self.task_type

    def get_level(self) -> int:
        """Return the difficulty level of this task."""
        return self.level

    def generate_random_number(self, min_val: int = 1, max_val: int = 100, seed_offset: int = 0) -> int:
        """Generate random number with optional seed offset for deterministic variety."""
        if seed_offset:
            # Create a temporary random generator with modified seed
            temp_random = random.Random()
            temp_random.seed(self.random.getstate()[1][0] + seed_offset)
            return temp_random.randint(min_val, max_val)
        return self.random.randint(min_val, max_val)

    def generate_task_id(self, prefix: str = "task") -> str:
        """Generate unique task identifier."""
        random_suffix = self.generate_random_number(10000, 99999)
        return f"{prefix}_{random_suffix}"

    def generate_filename(self, prefix: str, extension: str = ".html", seed_offset: int = 0) -> str:
        """Generate random filename with deterministic variety."""
        suffix = self.generate_random_number(100000, 999999, seed_offset)
        return f"{prefix}_{suffix:06d}{extension}"

    # Research page generation helpers
    def get_research_topics(self) -> List[str]:
        """Get available research topics."""
        return ["renewable_energy", "space_exploration", "artificial_intelligence", "climate_change", "quantum_computing"]

    def get_fact_types(self) -> List[str]:
        """Get available fact types for research."""
        return ["breakthrough_discovery", "efficiency_improvement", "cost_reduction", "environmental_impact", "future_applications"]

    def generate_research_fact(self, topic: str, fact_type: str, seed_offset: int = 0) -> str:
        """Generate a deterministic research fact based on topic and fact type."""
        fact_templates = {
            "renewable_energy": {
                "breakthrough_discovery": "Scientists discovered a new solar cell material with {percentage}% efficiency in {year}",
                "efficiency_improvement": "Solar panel efficiency improved by {percentage}% using advanced coating technology",
                "cost_reduction": "Wind energy costs reduced by {percentage}% through improved turbine design",
                "environmental_impact": "Renewable energy prevented {amount} tons of CO2 emissions in {year}",
                "future_applications": "Next-generation solar panels will achieve {percentage}% efficiency by {year}",
            },
            "space_exploration": {
                "breakthrough_discovery": "New exoplanet discovered {distance} light-years away with potential for life",
                "efficiency_improvement": "Rocket fuel efficiency improved by {percentage}% with new propulsion system",
                "cost_reduction": "Space launch costs reduced by {percentage}% through reusable rocket technology",
                "environmental_impact": "Space debris reduction program removed {amount} pieces of orbital waste",
                "future_applications": "Mars mission planned for {year} with {crew_size} crew members",
            },
            "artificial_intelligence": {
                "breakthrough_discovery": "AI system achieved {percentage}% accuracy in medical diagnosis",
                "efficiency_improvement": "Machine learning model processing speed improved by {percentage}%",
                "cost_reduction": "AI automation reduced operational costs by {percentage}% in manufacturing",
                "environmental_impact": "AI optimization reduced energy consumption by {percentage}% in data centers",
                "future_applications": "AI assistants will handle {percentage}% of customer service by {year}",
            },
            "climate_change": {
                "breakthrough_discovery": "New carbon capture technology removes {amount} tons CO2 per year",
                "efficiency_improvement": "Climate monitoring satellites improved prediction accuracy by {percentage}%",
                "cost_reduction": "Green technology adoption reduced costs by {percentage}% for businesses",
                "environmental_impact": "Reforestation project planted {amount} million trees in {year}",
                "future_applications": "Carbon neutrality target set for {year} with {percentage}% emission reduction",
            },
            "quantum_computing": {
                "breakthrough_discovery": "Quantum computer achieved {number}-qubit processing capability",
                "efficiency_improvement": "Quantum algorithm solved optimization problem {percentage}% faster",
                "cost_reduction": "Quantum computing development costs reduced by {percentage}% through new methods",
                "environmental_impact": "Quantum simulation accelerated clean energy research by {percentage}%",
                "future_applications": "Quantum computers will revolutionize {field} by {year}",
            },
        }

        template = fact_templates.get(topic, {}).get(fact_type, "Research shows significant progress in {topic}")

        # Generate deterministic values based on seed
        percentage = self.generate_random_number(15, 85, seed_offset)
        year = self.generate_random_number(2025, 2030, seed_offset + 1)
        amount = self.generate_random_number(100, 9999, seed_offset + 2)
        distance = self.generate_random_number(10, 500, seed_offset + 3)
        crew_size = self.generate_random_number(3, 8, seed_offset + 4)
        number = self.generate_random_number(50, 1000, seed_offset + 5)

        fields = ["medicine", "finance", "logistics", "communications", "research"]
        field = fields[self.generate_random_number(0, len(fields) - 1, seed_offset + 6)]

        return template.format(
            percentage=percentage,
            year=year,
            amount=amount,
            distance=distance,
            crew_size=crew_size,
            number=number,
            field=field,
            topic=topic.replace("_", " "),
        )

    # Image generation helpers
    def get_image_types(self) -> List[str]:
        """Get available image types for processing."""
        return ["tech_diagram", "data_chart", "process_flow", "infographic", "concept_art"]

    def generate_target_dimensions(self, level: int = 1) -> Tuple[int, int]:
        """Generate target image dimensions based on task level."""
        dimensions_by_level = {
            1: [(800, 600), (640, 480), (1024, 768)],
            2: [(1200, 300), (800, 200), (1000, 250)],
            3: [(200, 100), (150, 150), (300, 200)],
        }

        options = dimensions_by_level.get(level, [(800, 600)])
        return self.random.choice(options)

    # Enhanced content generation methods
    def generate_enhanced_task_structure(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate enhanced task structure with domain awareness and professional context."""
        if seed is not None:
            self.set_seed(seed)

        # Generate domain context and professional scenario
        domain_context = self.content_generator.select_domain_context(self.random.randint(1, 1000))
        scenario = self.content_generator.generate_professional_scenario(domain_context, self.random.randint(1, 1000))
        research_content = self.content_generator.generate_research_content(domain_context, scenario, self.random.randint(1, 1000))
        image_scenario = self.content_generator.generate_image_scenario(domain_context, self.random.randint(1, 1000))
        document_requirements = self.content_generator.generate_document_requirements(domain_context, scenario, self.random.randint(1, 1000))

        # Generate professional HTML content
        html_content = self.content_generator.generate_professional_html_content(domain_context, research_content, self.random.randint(1, 1000))

        return {
            "task_type": self.task_type,
            "level": self.level,
            "seed": seed or self.random.randint(1, 10000),
            "domain_context": domain_context,
            "professional_scenario": scenario,
            "research_content": research_content,
            "image_scenario": image_scenario,
            "document_requirements": document_requirements,
            "html_content": html_content,
            "evaluation_method": "multi_evaluator",
            "example_id": f"L{self.level}_{self.task_type}_{domain_context['domain']}_{self.random.randint(1, 10000)}",
        }

    def generate_professional_instruction(self, enhanced_data: Dict[str, Any], filenames: Optional[Dict[str, str]] = None) -> str:
        """Generate professional instruction text based on enhanced task data."""
        domain_context = enhanced_data["domain_context"]
        scenario = enhanced_data["professional_scenario"]
        image_scenario = enhanced_data["image_scenario"]
        document_req = enhanced_data["document_requirements"]

        # Create contextual instruction
        # Deterministic, explicit instruction including exact filenames and locations
        research_filename = filenames.get("research_filename") if filenames else "research.html"
        source_image_name = filenames.get("image_filename") if filenames else "source_image.png"
        processed_image_name = image_scenario["processed_filename"]
        document_name = document_req["output_filename"]

        instruction = (
            f"Complete a {scenario['project_type']} for {domain_context['company']} that combines research findings with processed imagery. "
            f"Extract the key finding text from the research page (accessible in Chromium or via http://localhost:8080/{research_filename}). "
            f"Process '{source_image_name}' to {image_scenario['width']}x{image_scenario['height']} pixels for {image_scenario['purpose']}. "
            f"Create a {domain_context['document_type'].replace('_', ' ')} containing both the extracted key finding text and the processed image. "
            f"Save the document as '{document_name}' and the processed image as '{processed_image_name}', both on the Desktop."
        )

        return instruction

    def generate_contextual_filenames(self, domain_context: Dict[str, Any], seed_offset: int = 0) -> Dict[str, str]:
        """Generate domain-appropriate filenames."""
        temp_random = random.Random()
        temp_random.seed(self.random.getstate()[1][0] + seed_offset)

        domain = domain_context["domain"]
        research_area = domain_context["research_area"]

        # Domain-specific filename prefixes
        filename_prefixes = {
            "corporate_research": ["business", "market", "strategy", "corporate", "analysis"],
            "academic_research": ["research", "study", "academic", "scientific", "paper"],
            "journalism_media": ["news", "article", "story", "report", "feature"],
            "marketing_communications": ["campaign", "brand", "marketing", "creative", "strategy"],
            "healthcare_research": ["medical", "health", "clinical", "patient", "study"],
            "financial_services": ["financial", "investment", "market", "economic", "portfolio"],
        }

        prefixes = filename_prefixes.get(domain, filename_prefixes["corporate_research"])
        prefix = temp_random.choice(prefixes)
        suffix = temp_random.randint(1000, 9999)

        return {
            "research_filename": f"{prefix}_{research_area}_{suffix}.html",
            "document_filename": f"{research_area}_report_{suffix}.docx",
            "image_filename": f"{prefix}_visual_{suffix}.png",
            "processed_image_filename": f"{prefix}_processed_{suffix}.png",
        }

    def generate_professional_context_description(self, domain_context: Dict[str, Any], scenario: Dict[str, Any]) -> str:
        """Generate professional context description for the task."""
        return (
            f"{domain_context['domain'].replace('_', ' ').title()} {domain_context['document_type'].replace('_', ' ')} "
            f"for {scenario['business_timeline']} ({scenario['urgency_level']} priority)"
        )

    # Legacy document generation helpers (maintained for backward compatibility)
    def generate_document_structure(self, task_type: str) -> Dict[str, Any]:
        """Generate document structure requirements."""
        structures = {
            "basic_info_gathering": {
                "title": "Research Report",
                "sections": ["Summary", "Key Findings", "Visual Analysis"],
                "required_elements": ["researched_fact", "processed_image"],
            }
        }
        return structures.get(task_type, {"title": "Document", "sections": [], "required_elements": []})
