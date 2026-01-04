"""
Base generator class for visual content integration tasks.
Enhanced with professional domain-specific content generation for realistic scenarios.
"""

import random
from abc import abstractmethod
from typing import Dict, Any, Optional, Tuple
from ...base import BaseTask


class ImageProcessingBaseTask(BaseTask):
    """Base class for visual content integration task implementations with enhanced content generation."""

    def __init__(self, task_type: str, level: int):
        super().__init__()
        self.task_type = task_type
        self.level = level
        self.random = random.Random()

        # Initialize professional domain pools for enhanced variability
        self._init_domain_pools()

    def set_seed(self, seed: int):
        """Set random seed for reproducible generation."""
        self.random.seed(seed)

    def _init_domain_pools(self):
        """Initialize professional domain pools for enhanced content variability."""
        # Professional domains with realistic business contexts
        self.professional_domains = {
            "marketing_design": {
                "companies": [
                    "BrandCraft Studios",
                    "VisualEdge Marketing",
                    "Creative Pulse Agency",
                    "Design Forward Inc",
                    "Visual Impact Solutions",
                    "Brand Matrix Creative",
                    "Pixel Perfect Marketing",
                    "Concept Forge Studio",
                    "Dynamic Design House",
                    "Creative Vision Labs",
                    "Art Direction Pro",
                    "Brand Strategy Collective",
                    "Visual Communications Hub",
                    "Design Innovation Group",
                    "Creative Content Studio",
                    "Marketing Visual Solutions",
                    "Brand Experience Design",
                    "Creative Media Collective",
                    "Visual Identity Studios",
                    "Design Excellence Group",
                    "Creative Brand Solutions",
                    "Visual Marketing Pro",
                    "Design Strategy Studio",
                    "Brand Creative Lab",
                    "Image Processing Masters",
                ],
                "image_scenarios": [
                    "social_media_campaign",
                    "product_photography",
                    "brand_identity_design",
                    "marketing_collateral",
                    "advertising_creative",
                    "website_graphics",
                    "promotional_materials",
                    "event_branding",
                    "packaging_design",
                    "digital_marketing_assets",
                    "print_advertisements",
                    "corporate_branding",
                    "campaign_visuals",
                    "brand_guidelines",
                    "marketing_presentations",
                ],
                "document_types": [
                    "campaign_brief",
                    "brand_guidelines",
                    "creative_proposal",
                    "design_specification",
                    "marketing_presentation",
                    "visual_identity_guide",
                    "creative_brief",
                    "project_overview",
                    "brand_strategy_document",
                    "design_proposal",
                    "marketing_report",
                    "creative_summary",
                    "brand_analysis",
                    "campaign_strategy",
                    "visual_content_plan",
                ],
                "image_purposes": [
                    "social_media_optimization",
                    "print_advertising",
                    "web_banner_creation",
                    "product_showcase",
                    "brand_presentation",
                    "marketing_campaign",
                    "promotional_graphics",
                    "corporate_identity",
                    "digital_advertising",
                    "content_marketing",
                    "brand_storytelling",
                    "visual_communication",
                ],
                "color_schemes": ["#e74c3c", "#3498db", "#f39c12", "#2ecc71", "#9b59b6"],
                "contexts": [
                    "product launch campaign",
                    "brand refresh project",
                    "social media strategy",
                    "advertising campaign",
                    "corporate rebranding",
                    "marketing material update",
                    "promotional campaign",
                    "brand identity development",
                ],
            },
            "scientific_publication": {
                "companies": [
                    "Research Institute",
                    "Scientific Publications Lab",
                    "Academic Research Center",
                    "Innovation Studies Institute",
                    "Scientific Data Analytics",
                    "Research Methodology Group",
                    "Laboratory Sciences Division",
                    "Academic Excellence Center",
                    "Scientific Research Foundation",
                    "Knowledge Discovery Institute",
                    "Research Development Office",
                    "Scientific Investigation Unit",
                    "Advanced Research Laboratory",
                    "Scientific Publication Services",
                    "Research Documentation Center",
                    "Academic Publishing Solutions",
                    "Scientific Communication Hub",
                    "Research Analysis Institute",
                    "Laboratory Data Solutions",
                    "Scientific Content Services",
                    "Research Excellence Group",
                    "Academic Research Solutions",
                    "Scientific Study Center",
                    "Research Innovation Lab",
                    "Scientific Documentation Services",
                ],
                "image_scenarios": [
                    "data_visualization",
                    "experimental_results",
                    "scientific_diagrams",
                    "research_charts",
                    "laboratory_equipment",
                    "methodology_flowcharts",
                    "statistical_analysis",
                    "research_findings",
                    "experimental_setup",
                    "scientific_illustrations",
                    "research_infographics",
                    "academic_presentations",
                    "study_results",
                    "research_methodology",
                    "scientific_documentation",
                ],
                "document_types": [
                    "research_paper",
                    "scientific_report",
                    "study_findings",
                    "experimental_analysis",
                    "research_methodology",
                    "academic_publication",
                    "scientific_documentation",
                    "laboratory_report",
                    "research_summary",
                    "study_overview",
                    "scientific_analysis",
                    "research_findings_report",
                    "academic_study",
                    "scientific_investigation",
                    "research_documentation",
                ],
                "image_purposes": [
                    "data_presentation",
                    "scientific_illustration",
                    "research_documentation",
                    "academic_publication",
                    "study_visualization",
                    "experimental_evidence",
                    "research_communication",
                    "scientific_reporting",
                    "academic_presentation",
                    "research_analysis",
                    "scientific_education",
                    "study_documentation",
                ],
                "color_schemes": ["#2c3e50", "#34495e", "#7f8c8d", "#95a5a6", "#ecf0f1"],
                "contexts": [
                    "peer review submission",
                    "grant application",
                    "conference presentation",
                    "academic publication",
                    "research collaboration",
                    "scientific documentation",
                    "study publication",
                    "research dissemination",
                ],
            },
            "business_presentation": {
                "companies": [
                    "Corporate Solutions Inc",
                    "Business Strategy Group",
                    "Executive Consulting",
                    "Professional Services Corp",
                    "Strategic Business Solutions",
                    "Corporate Development Group",
                    "Business Excellence Institute",
                    "Executive Strategy Firm",
                    "Management Consulting Services",
                    "Business Innovation Lab",
                    "Corporate Strategy Solutions",
                    "Professional Development Group",
                    "Business Analysis Corporation",
                    "Strategic Planning Services",
                    "Corporate Consulting Group",
                    "Business Process Solutions",
                    "Executive Advisory Services",
                    "Corporate Excellence Center",
                    "Business Strategy Institute",
                    "Professional Consulting Firm",
                    "Strategic Business Development",
                    "Corporate Innovation Group",
                    "Business Solutions Provider",
                    "Executive Services Corporation",
                    "Strategic Management Consulting",
                ],
                "image_scenarios": [
                    "quarterly_reports",
                    "business_analytics",
                    "performance_metrics",
                    "strategic_planning",
                    "financial_presentations",
                    "market_analysis",
                    "business_proposals",
                    "executive_summaries",
                    "corporate_overviews",
                    "business_development",
                    "strategic_initiatives",
                    "performance_dashboards",
                    "market_research",
                    "business_intelligence",
                    "corporate_communications",
                ],
                "document_types": [
                    "executive_summary",
                    "business_proposal",
                    "strategic_plan",
                    "quarterly_report",
                    "market_analysis",
                    "business_overview",
                    "corporate_presentation",
                    "performance_report",
                    "strategic_initiative",
                    "business_case",
                    "investment_proposal",
                    "company_overview",
                    "business_strategy",
                    "corporate_update",
                    "executive_briefing",
                ],
                "image_purposes": [
                    "executive_presentation",
                    "board_meeting",
                    "investor_pitch",
                    "strategic_planning",
                    "performance_review",
                    "business_communication",
                    "corporate_reporting",
                    "strategic_overview",
                    "business_analysis",
                    "market_presentation",
                    "financial_reporting",
                    "corporate_strategy",
                ],
                "color_schemes": ["#1f2937", "#374151", "#4b5563", "#6b7280", "#9ca3af"],
                "contexts": [
                    "board meeting presentation",
                    "investor pitch",
                    "quarterly business review",
                    "strategic planning session",
                    "executive briefing",
                    "corporate presentation",
                    "business proposal",
                    "performance review",
                ],
            },
            "educational_content": {
                "companies": [
                    "Educational Excellence Institute",
                    "Learning Solutions Center",
                    "Academic Development Group",
                    "Educational Technology Lab",
                    "Learning Innovation Institute",
                    "Educational Content Services",
                    "Academic Support Center",
                    "Learning Development Corporation",
                    "Educational Research Institute",
                    "Academic Excellence Group",
                    "Learning Technology Solutions",
                    "Educational Innovation Lab",
                    "Academic Content Development",
                    "Learning Resources Center",
                    "Educational Services Corporation",
                    "Academic Solutions Provider",
                    "Learning Enhancement Institute",
                    "Educational Development Group",
                    "Academic Innovation Center",
                    "Learning Excellence Services",
                    "Educational Quality Institute",
                    "Academic Resource Center",
                    "Learning Support Services",
                    "Educational Technology Group",
                    "Academic Development Solutions",
                ],
                "image_scenarios": [
                    "educational_diagrams",
                    "learning_materials",
                    "instructional_graphics",
                    "educational_presentations",
                    "course_materials",
                    "training_visuals",
                    "educational_infographics",
                    "learning_aids",
                    "instructional_design",
                    "educational_illustrations",
                    "academic_content",
                    "learning_resources",
                    "educational_media",
                    "training_materials",
                    "academic_visuals",
                ],
                "document_types": [
                    "course_material",
                    "educational_guide",
                    "learning_module",
                    "instructional_document",
                    "training_manual",
                    "educational_resource",
                    "academic_guide",
                    "learning_material",
                    "educational_handbook",
                    "course_outline",
                    "instructional_guide",
                    "educational_overview",
                    "learning_plan",
                    "academic_resource",
                    "educational_summary",
                ],
                "image_purposes": [
                    "educational_illustration",
                    "learning_enhancement",
                    "instructional_support",
                    "academic_presentation",
                    "educational_communication",
                    "learning_visualization",
                    "instructional_design",
                    "educational_documentation",
                    "academic_illustration",
                    "learning_aid",
                    "educational_material",
                    "instructional_content",
                ],
                "color_schemes": ["#3498db", "#2980b9", "#5dade2", "#85c1e9", "#aed6f1"],
                "contexts": [
                    "course development",
                    "educational program",
                    "training initiative",
                    "academic project",
                    "learning resource creation",
                    "educational content development",
                    "instructional design",
                    "curriculum development",
                ],
            },
            "media_journalism": {
                "companies": [
                    "Digital News Network",
                    "Media Communications Group",
                    "Journalism Excellence Institute",
                    "News Content Services",
                    "Media Production Studio",
                    "Digital Media Solutions",
                    "News Broadcasting Corporation",
                    "Media Development Group",
                    "Journalism Innovation Lab",
                    "News Content Creation",
                    "Media Strategy Group",
                    "Digital Journalism Services",
                    "News Media Corporation",
                    "Media Content Solutions",
                    "Journalism Technology Institute",
                    "News Production Services",
                    "Media Excellence Center",
                    "Digital News Solutions",
                    "Journalism Development Group",
                    "Media Innovation Lab",
                    "News Content Network",
                    "Media Communication Services",
                    "Journalism Solutions Corporation",
                    "Digital Media Group",
                    "News Excellence Institute",
                ],
                "image_scenarios": [
                    "news_graphics",
                    "media_presentations",
                    "journalistic_content",
                    "news_visuals",
                    "media_illustrations",
                    "news_infographics",
                    "editorial_graphics",
                    "media_documentation",
                    "news_photography",
                    "media_design",
                    "journalistic_visuals",
                    "news_content",
                    "media_storytelling",
                    "editorial_design",
                    "news_presentation",
                ],
                "document_types": [
                    "news_article",
                    "media_report",
                    "editorial_piece",
                    "news_brief",
                    "media_analysis",
                    "journalistic_report",
                    "news_summary",
                    "media_overview",
                    "editorial_document",
                    "news_feature",
                    "media_story",
                    "journalism_piece",
                    "news_documentation",
                    "media_content",
                    "editorial_analysis",
                ],
                "image_purposes": [
                    "news_illustration",
                    "media_storytelling",
                    "editorial_support",
                    "journalistic_documentation",
                    "news_presentation",
                    "media_communication",
                    "editorial_enhancement",
                    "news_visualization",
                    "media_content_creation",
                    "journalistic_illustration",
                    "news_graphics",
                    "media_design",
                ],
                "color_schemes": ["#e74c3c", "#c0392b", "#ec7063", "#f1948a", "#fadbd8"],
                "contexts": [
                    "breaking news coverage",
                    "feature story development",
                    "editorial project",
                    "media campaign",
                    "news documentation",
                    "journalistic investigation",
                    "media presentation",
                    "editorial enhancement",
                ],
            },
            "healthcare_communication": {
                "companies": [
                    "Healthcare Communications Group",
                    "Medical Content Services",
                    "Health Information Institute",
                    "Healthcare Media Solutions",
                    "Medical Communications Corporation",
                    "Healthcare Content Development",
                    "Health Education Services",
                    "Medical Information Group",
                    "Healthcare Documentation Center",
                    "Medical Content Creation",
                    "Health Communication Solutions",
                    "Healthcare Media Group",
                    "Medical Information Services",
                    "Healthcare Excellence Institute",
                    "Health Content Corporation",
                    "Medical Communications Lab",
                    "Healthcare Innovation Group",
                    "Medical Documentation Services",
                    "Health Media Solutions",
                    "Healthcare Content Services",
                    "Medical Communication Institute",
                    "Healthcare Information Group",
                    "Health Content Development",
                    "Medical Media Corporation",
                    "Healthcare Communication Services",
                ],
                "image_scenarios": [
                    "medical_illustrations",
                    "healthcare_infographics",
                    "health_education_materials",
                    "medical_presentations",
                    "healthcare_documentation",
                    "medical_diagrams",
                    "health_communication_visuals",
                    "healthcare_graphics",
                    "medical_content",
                    "health_information_design",
                    "healthcare_presentations",
                    "medical_visuals",
                    "health_education_graphics",
                    "healthcare_materials",
                    "medical_communication_content",
                ],
                "document_types": [
                    "health_information",
                    "medical_report",
                    "healthcare_guide",
                    "health_education_material",
                    "medical_documentation",
                    "healthcare_overview",
                    "health_summary",
                    "medical_analysis",
                    "healthcare_presentation",
                    "health_brief",
                    "medical_information",
                    "healthcare_document",
                    "health_communication",
                    "medical_overview",
                    "healthcare_material",
                ],
                "image_purposes": [
                    "health_education",
                    "medical_illustration",
                    "healthcare_communication",
                    "health_information_design",
                    "medical_documentation",
                    "healthcare_presentation",
                    "health_awareness",
                    "medical_communication",
                    "healthcare_education",
                    "health_visualization",
                    "medical_information_design",
                    "healthcare_illustration",
                ],
                "color_schemes": ["#27ae60", "#2ecc71", "#58d68d", "#82e0aa", "#a9dfbf"],
                "contexts": [
                    "health education program",
                    "medical communication",
                    "healthcare presentation",
                    "health awareness campaign",
                    "medical documentation",
                    "healthcare training",
                    "health information dissemination",
                    "medical education",
                ],
            },
        }

        # Enhanced image generation parameters for professional contexts
        self.professional_image_types = {
            "corporate_presentation": {
                "layouts": ["slide_format", "report_layout", "dashboard_design", "infographic_style"],
                "elements": ["charts", "diagrams", "data_visualization", "professional_graphics"],
                "styles": ["modern_clean", "executive_formal", "business_casual", "corporate_elegant"],
            },
            "marketing_visual": {
                "layouts": ["social_media", "web_banner", "print_ready", "promotional_design"],
                "elements": ["brand_elements", "product_focus", "call_to_action", "visual_hierarchy"],
                "styles": ["vibrant_engaging", "minimalist_modern", "bold_creative", "professional_approachable"],
            },
            "scientific_diagram": {
                "layouts": ["research_chart", "data_graph", "process_diagram", "technical_illustration"],
                "elements": ["data_points", "scientific_notation", "measurement_scales", "research_elements"],
                "styles": ["academic_formal", "research_precise", "scientific_clear", "technical_detailed"],
            },
            "educational_material": {
                "layouts": ["instructional_design", "learning_aid", "educational_poster", "course_material"],
                "elements": ["learning_objectives", "educational_graphics", "instructional_elements", "knowledge_visualization"],
                "styles": ["friendly_approachable", "clear_informative", "engaging_interactive", "educational_professional"],
            },
        }

        # Professional caption templates based on domains
        self.professional_captions = {
            "marketing_design": [
                "Brand identity visual showcasing {company} marketing strategy",
                "Creative campaign asset for {project_type} initiative",
                "Professional marketing graphic optimized for {context}",
                "Brand communication visual supporting {campaign_goal}",
                "Strategic marketing asset designed for {target_audience}",
                "Creative brand expression for {business_objective}",
                "Marketing communication graphic enhancing {brand_message}",
                "Professional campaign visual for {market_segment}",
                "Brand strategy illustration supporting {marketing_goal}",
                "Creative marketing content for {promotional_context}",
            ],
            "scientific_publication": [
                "Research data visualization for {study_type} analysis",
                "Scientific illustration supporting {research_finding}",
                "Experimental results diagram from {research_context}",
                "Academic research graphic illustrating {scientific_concept}",
                "Study methodology visualization for {research_area}",
                "Scientific data representation of {experimental_results}",
                "Research documentation graphic for {academic_purpose}",
                "Scientific analysis illustration supporting {study_conclusion}",
                "Academic research visual for {publication_context}",
                "Laboratory results diagram from {scientific_investigation}",
            ],
            "business_presentation": [
                "Executive summary visual for {business_context}",
                "Strategic planning graphic supporting {business_objective}",
                "Corporate performance illustration for {reporting_period}",
                "Business analytics visualization of {key_metrics}",
                "Strategic initiative diagram for {business_goal}",
                "Executive briefing graphic illustrating {business_insight}",
                "Corporate communication visual for {stakeholder_group}",
                "Business development illustration supporting {strategic_plan}",
                "Performance metrics visualization for {business_review}",
                "Strategic overview graphic for {corporate_initiative}",
            ],
            "educational_content": [
                "Educational illustration enhancing {learning_objective}",
                "Instructional graphic supporting {course_topic}",
                "Learning aid visualization for {educational_concept}",
                "Academic content illustration for {subject_area}",
                "Educational resource graphic explaining {learning_material}",
                "Instructional design visual for {educational_purpose}",
                "Learning enhancement illustration for {course_module}",
                "Educational communication graphic for {target_learners}",
                "Academic illustration supporting {instructional_goal}",
                "Learning resource visual for {educational_context}",
            ],
            "media_journalism": [
                "Editorial graphic supporting {news_story}",
                "Journalistic illustration for {media_coverage}",
                "News content visual enhancing {editorial_piece}",
                "Media communication graphic for {news_context}",
                "Editorial enhancement illustration for {story_theme}",
                "Journalism visual supporting {news_analysis}",
                "Media content graphic for {editorial_purpose}",
                "News illustration enhancing {journalistic_content}",
                "Editorial design visual for {media_presentation}",
                "Journalism graphic supporting {news_documentation}",
            ],
            "healthcare_communication": [
                "Health education illustration for {medical_topic}",
                "Medical communication graphic supporting {health_information}",
                "Healthcare visual enhancing {patient_education}",
                "Health information illustration for {medical_context}",
                "Medical documentation graphic for {healthcare_purpose}",
                "Health awareness visual supporting {medical_communication}",
                "Healthcare education illustration for {health_topic}",
                "Medical information graphic enhancing {healthcare_content}",
                "Health communication visual for {medical_education}",
                "Healthcare illustration supporting {health_awareness}",
            ],
        }

    def generate(self, **kwargs) -> Dict[str, Any]:
        """Generate a task instance (implemented by calling generate_task_data)."""
        seed = kwargs.get("seed")
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

    def generate_random_number(self, min_val: int = 1, max_val: int = 100) -> int:
        """Generate random number."""
        return self.random.randint(min_val, max_val)

    def generate_image_filename(self, prefix: str = "image", extension: str = ".png") -> str:
        """Generate random image filename."""
        suffix = self.random.choice(["_photo", "_picture", "_graphic", "_sample", "_test"])
        number = self.random.randint(1, 99)
        return f"{prefix}{suffix}_{number:02d}{extension}"

    def generate_document_filename(self, prefix: str = "document", extension: str = ".docx") -> str:
        """Generate random document filename."""
        suffix = self.random.choice(["_report", "_template", "_draft", "_analysis", "_notes"])
        number = self.random.randint(1, 99)
        return f"{prefix}{suffix}_{number:02d}{extension}"

    def generate_caption_text(self, domain: str = None, context: Dict[str, Any] = None) -> str:
        """Generate professional caption text based on domain and context."""
        if domain and domain in self.professional_captions:
            caption_template = self.random.choice(self.professional_captions[domain])

            # Fill in template variables based on context
            if context:
                try:
                    return caption_template.format(**context)
                except KeyError:
                    # Fallback to basic substitution if context doesn't have all keys
                    return self._fill_caption_template(caption_template, domain, context)
            else:
                return self._fill_caption_template(caption_template, domain)
        else:
            # Fallback to enhanced generic captions
            professional_captions = [
                "Professional visual content for business presentation",
                "Enhanced graphic design for marketing materials",
                "Optimized image for digital communication",
                "Professional illustration for corporate use",
                "Strategic visual content for business objectives",
                "Enhanced graphic for professional documentation",
                "Optimized visual asset for marketing campaign",
                "Professional image for business communication",
                "Strategic graphic design for corporate presentation",
                "Enhanced visual content for professional use",
            ]
            return self.random.choice(professional_captions)

    def _fill_caption_template(self, template: str, domain: str, context: Dict[str, Any] = None) -> str:
        """Fill caption template with appropriate values."""
        # Create default context values based on domain
        domain_data = self.professional_domains.get(domain, {})

        fill_values = {
            "company": self.random.choice(domain_data.get("companies", ["Professional Company"])),
            "project_type": self.random.choice(domain_data.get("image_scenarios", ["business project"])),
            "context": self.random.choice(domain_data.get("contexts", ["professional context"])),
            "campaign_goal": "brand awareness",
            "target_audience": "professional stakeholders",
            "business_objective": "strategic growth",
            "brand_message": "corporate excellence",
            "market_segment": "target market",
            "marketing_goal": "engagement",
            "promotional_context": "business promotion",
            "study_type": "research",
            "research_finding": "significant discovery",
            "research_context": "academic study",
            "scientific_concept": "research methodology",
            "research_area": "scientific investigation",
            "experimental_results": "study outcomes",
            "academic_purpose": "research documentation",
            "study_conclusion": "research findings",
            "publication_context": "academic publication",
            "scientific_investigation": "laboratory study",
            "business_context": "corporate environment",
            "reporting_period": "quarterly review",
            "key_metrics": "performance indicators",
            "business_goal": "strategic objective",
            "business_insight": "market analysis",
            "stakeholder_group": "executive team",
            "strategic_plan": "business strategy",
            "business_review": "performance assessment",
            "corporate_initiative": "strategic project",
            "learning_objective": "educational goal",
            "course_topic": "academic subject",
            "educational_concept": "learning principle",
            "subject_area": "academic discipline",
            "learning_material": "educational content",
            "educational_purpose": "instructional goal",
            "course_module": "learning unit",
            "target_learners": "students",
            "instructional_goal": "educational objective",
            "educational_context": "learning environment",
            "news_story": "current events",
            "media_coverage": "news reporting",
            "editorial_piece": "journalistic content",
            "news_context": "media environment",
            "story_theme": "editorial focus",
            "news_analysis": "media investigation",
            "editorial_purpose": "journalistic goal",
            "journalistic_content": "news material",
            "media_presentation": "news delivery",
            "news_documentation": "media record",
            "medical_topic": "health subject",
            "health_information": "medical content",
            "patient_education": "health awareness",
            "medical_context": "healthcare environment",
            "healthcare_purpose": "medical objective",
            "medical_communication": "health information",
            "health_topic": "medical subject",
            "healthcare_content": "health material",
            "medical_education": "health learning",
            "health_awareness": "medical understanding",
        }

        # Override with context values if provided
        if context:
            fill_values.update(context)

        # Try to format with available values
        try:
            return template.format(**fill_values)
        except KeyError:
            # If template has unknown variables, replace them with generic terms
            result = template
            for key in ["company", "project_type", "context"]:
                if "{" + key + "}" in result and key not in fill_values:
                    result = result.replace("{" + key + "}", f"professional {key}")
            return result

    def generate_image_dimensions(self) -> Tuple[int, int]:
        """Generate random image dimensions for resizing tasks."""
        dimension_options = [
            (200, 150),
            (300, 200),
            (250, 180),
            (150, 100),
            (400, 300),
            (180, 120),
        ]
        return self.random.choice(dimension_options)

    def generate_image_format_pair(self) -> Tuple[str, str]:
        """Generate source and target format pair for conversion tasks (DocX-compatible only)."""
        format_pairs = [
            ("png", "jpg"),
            ("jpg", "png"),
        ]
        return self.random.choice(format_pairs)

    def generate_filter_type(self) -> str:
        """Generate random filter type for Level 3 tasks."""
        filters = ["Grayscale", "Sepia", "Invert Colors", "Blur"]
        return self.random.choice(filters)

    def generate_image_type(self) -> str:
        """Generate image type for programmatic generation."""
        image_types = ["geometric", "colorful", "text_based", "gradient"]
        return self.random.choice(image_types)

    def generate_professional_scenario(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate professional scenario with domain-specific context."""
        if seed is not None:
            self.set_seed(seed)

        # Select professional domain
        domain = self.random.choice(list(self.professional_domains.keys()))
        domain_data = self.professional_domains[domain]

        # Generate professional context
        company = self.random.choice(domain_data["companies"])
        image_scenario = self.random.choice(domain_data["image_scenarios"])
        document_type = self.random.choice(domain_data["document_types"])
        image_purpose = self.random.choice(domain_data["image_purposes"])
        context = self.random.choice(domain_data["contexts"])
        color_scheme = self.random.choice(domain_data["color_schemes"])

        # Generate professional image type based on domain
        if domain == "marketing_design":
            professional_type = "marketing_visual"
        elif domain == "scientific_publication":
            professional_type = "scientific_diagram"
        elif domain == "business_presentation":
            professional_type = "corporate_presentation"
        else:
            professional_type = "educational_material"

        image_type_data = self.professional_image_types[professional_type]
        layout = self.random.choice(image_type_data["layouts"])
        elements = self.random.choice(image_type_data["elements"])
        style = self.random.choice(image_type_data["styles"])

        return {
            "domain": domain,
            "company": company,
            "image_scenario": image_scenario,
            "document_type": document_type,
            "image_purpose": image_purpose,
            "context": context,
            "color_scheme": color_scheme,
            "professional_type": professional_type,
            "layout": layout,
            "elements": elements,
            "style": style,
        }

    def generate_enhanced_task_structure(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate enhanced task structure with professional domain context."""
        if seed is not None:
            self.set_seed(seed)

        # Generate professional scenario
        professional_scenario = self.generate_professional_scenario(seed)

        # Generate domain-aware filenames
        domain = professional_scenario["domain"]
        company = professional_scenario["company"]
        scenario = professional_scenario["image_scenario"]

        # Create professional filename patterns
        company_code = company.lower().replace(" ", "_").replace(",", "")[:12]
        scenario_code = scenario.replace("_", "-")

        # Professional image filename (DocX-compatible formats only)
        image_formats = ["jpg", "png"]
        source_format = self.random.choice(image_formats)
        source_filename = f"{company_code}_{scenario_code}_source.{source_format}"

        # Professional document filename
        doc_name = professional_scenario["document_type"].replace("_", "-")
        document_filename = f"{company_code}_{doc_name}.docx"

        return {
            "task_type": self.task_type,
            "level": self.level,
            "seed": seed or self.random.randint(1, 10000),
            "category": "image_processing",
            "evaluation_mode": "multi_evaluator",
            "example_id": f"VC_L{self.level}_{self.task_type}_{self.random.randint(1, 1000)}",
            "professional_scenario": professional_scenario,
            "source_image": source_filename,
            "document_file": document_filename,
            "image_type": professional_scenario["professional_type"],
            "domain_context": {
                "domain": domain,
                "company": company,
                "image_scenario": scenario,
                "project_type": professional_scenario["image_purpose"],
                "context": professional_scenario["context"],
            },
        }

    def generate_basic_task_structure(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate basic task structure (enhanced version with backward compatibility)."""
        return self.generate_enhanced_task_structure(seed)
