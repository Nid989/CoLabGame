"""
Enhanced Content Generator for Integrated Workflow tasks.
Provides domain-based content generation with professional contexts, realistic scenarios,
and comprehensive HTML templates for multi-application workflows.
"""

import random
from typing import Dict, Any
from datetime import datetime


class WorkflowOrchestrationContentGenerator:
    """Enhanced content generator with domain-based variability and professional scenarios."""

    def __init__(self):
        self.random = random.Random()

        # Professional domains with realistic business contexts
        self.domains = {
            "corporate_research": {
                "companies": [
                    "TechFlow Solutions",
                    "DataCorp Industries",
                    "InnovateAI Systems",
                    "CodeCraft Technologies",
                    "ByteStream Analytics",
                    "CloudTech Enterprises",
                    "DevTools Pro",
                    "SmartCode Solutions",
                    "TechVision Corp",
                    "DigitalEdge Systems",
                    "CyberSoft Industries",
                    "NetLogic Technologies",
                    "SystemCore Solutions",
                    "DataMind Analytics",
                    "TechForge Labs",
                    "CodeWorks International",
                    "DigitalPro Services",
                    "InfoTech Solutions",
                    "SoftLine Systems",
                    "TechBase Corporation",
                    "DataFlow Technologies",
                    "CloudCore Solutions",
                    "WebTech Innovations",
                    "CodeLab Dynamics",
                    "TechSuite Pro",
                ],
                "research_areas": [
                    "market_analysis",
                    "competitor_research",
                    "technology_trends",
                    "customer_analytics",
                    "business_intelligence",
                    "digital_transformation",
                    "industry_benchmarking",
                    "product_development",
                    "strategic_planning",
                    "innovation_tracking",
                ],
                "document_types": [
                    "executive_summary",
                    "market_report",
                    "competitive_analysis",
                    "business_proposal",
                    "strategy_document",
                    "research_brief",
                    "industry_overview",
                    "trend_analysis",
                    "feasibility_study",
                    "investment_report",
                ],
                "image_requirements": [
                    "chart_resizing",
                    "infographic_editing",
                    "logo_placement",
                    "banner_creation",
                    "diagram_formatting",
                    "presentation_graphics",
                    "report_visuals",
                    "data_visualization",
                    "process_diagrams",
                    "organizational_charts",
                ],
                "contexts": [
                    "quarterly business review",
                    "strategic planning session",
                    "investor presentation",
                    "board meeting report",
                    "market entry analysis",
                    "product launch research",
                    "competitive landscape study",
                    "digital strategy review",
                ],
                "color_schemes": ["#1f2937", "#3b82f6", "#059669"],  # Corporate blue/gray
            },
            "academic_research": {
                "companies": [
                    "Research University",
                    "Institute of Technology",
                    "Science Research Center",
                    "Academic Innovation Lab",
                    "University Studies Department",
                    "Graduate Research Institute",
                    "Advanced Studies Center",
                    "Scientific Research Foundation",
                    "Educational Technology Lab",
                    "Knowledge Discovery Institute",
                    "Scholarly Research Group",
                    "University Innovation Hub",
                    "Academic Excellence Center",
                    "Research Development Office",
                    "Scientific Investigation Unit",
                    "Higher Education Research",
                    "Academic Research Consortium",
                    "University Science Department",
                    "Graduate Studies Institute",
                    "Research Innovation Center",
                ],
                "research_areas": [
                    "climate_science",
                    "biotechnology",
                    "quantum_physics",
                    "computer_science",
                    "materials_engineering",
                    "environmental_studies",
                    "biomedical_research",
                    "artificial_intelligence",
                    "renewable_energy",
                    "space_exploration",
                    "neuroscience",
                    "genetics_research",
                    "sustainable_technology",
                    "data_science",
                    "robotics_engineering",
                ],
                "document_types": [
                    "research_proposal",
                    "literature_review",
                    "findings_report",
                    "grant_application",
                    "conference_paper",
                    "research_summary",
                    "methodology_document",
                    "progress_report",
                    "thesis_chapter",
                    "research_overview",
                ],
                "image_requirements": [
                    "scientific_diagrams",
                    "data_visualizations",
                    "research_charts",
                    "methodology_flowcharts",
                    "experimental_setups",
                    "statistical_graphs",
                    "scientific_illustrations",
                    "lab_equipment_photos",
                    "research_infographics",
                    "academic_presentations",
                ],
                "contexts": [
                    "grant funding application",
                    "peer review submission",
                    "conference presentation",
                    "research collaboration",
                    "academic publication",
                    "doctoral defense",
                    "research symposium",
                    "scientific workshop",
                ],
                "color_schemes": ["#7c3aed", "#8b5cf6", "#a78bfa"],  # Academic purple
            },
            "journalism_media": {
                "companies": [
                    "Tech Today Media",
                    "Science Weekly News",
                    "Business Insight Journal",
                    "Digital Times",
                    "Innovation Reporter",
                    "Future Focus News",
                    "Industry Watch Media",
                    "Technology Tribune",
                    "Business Chronicle",
                    "Science Explorer",
                    "Digital Trends Report",
                    "Innovation Daily",
                    "Tech Pulse Media",
                    "Business Spotlight",
                    "Science Digest",
                    "Technology Review",
                    "Industry Insider",
                    "Digital Herald",
                    "Innovation News Network",
                    "Tech Analysis Today",
                ],
                "research_areas": [
                    "breaking_news",
                    "investigative_stories",
                    "technology_features",
                    "business_analysis",
                    "science_reporting",
                    "industry_coverage",
                    "startup_profiles",
                    "innovation_tracking",
                    "market_developments",
                    "expert_interviews",
                    "trend_analysis",
                    "product_reviews",
                    "company_profiles",
                    "sector_analysis",
                    "research_coverage",
                ],
                "document_types": [
                    "news_article",
                    "feature_story",
                    "press_release",
                    "editorial_piece",
                    "investigative_report",
                    "interview_transcript",
                    "news_brief",
                    "media_advisory",
                    "story_pitch",
                    "journalism_report",
                ],
                "image_requirements": [
                    "featured_images",
                    "article_headers",
                    "photo_editing",
                    "media_graphics",
                    "news_visuals",
                    "infographic_creation",
                    "social_media_images",
                    "press_photos",
                    "editorial_graphics",
                    "story_illustrations",
                ],
                "contexts": [
                    "breaking news coverage",
                    "feature article development",
                    "press conference report",
                    "exclusive interview",
                    "investigative journalism",
                    "industry event coverage",
                    "product launch story",
                    "expert analysis piece",
                ],
                "color_schemes": ["#dc2626", "#ef4444", "#f87171"],  # Media red
            },
            "marketing_communications": {
                "companies": [
                    "Creative Marketing Solutions",
                    "Brand Strategy Group",
                    "Digital Marketing Pro",
                    "Content Creation Studio",
                    "Marketing Innovation Lab",
                    "Brand Development Agency",
                    "Creative Communications",
                    "Marketing Dynamics",
                    "Content Strategy Studio",
                    "Brand Experience Group",
                    "Digital Content Agency",
                    "Marketing Excellence Center",
                    "Creative Strategy Solutions",
                    "Brand Communication Studio",
                    "Content Marketing Pro",
                    "Marketing Innovation Studio",
                    "Creative Brand Solutions",
                    "Digital Strategy Group",
                    "Content Development Agency",
                    "Marketing Creative Lab",
                ],
                "research_areas": [
                    "brand_analysis",
                    "market_research",
                    "consumer_behavior",
                    "campaign_effectiveness",
                    "digital_trends",
                    "social_media_analysis",
                    "content_strategy",
                    "audience_research",
                    "competitor_analysis",
                    "marketing_analytics",
                    "brand_positioning",
                    "customer_insights",
                    "market_segmentation",
                    "promotional_strategies",
                    "engagement_metrics",
                ],
                "document_types": [
                    "campaign_brief",
                    "brand_guidelines",
                    "marketing_proposal",
                    "content_strategy",
                    "creative_brief",
                    "marketing_report",
                    "brand_analysis",
                    "campaign_summary",
                    "strategy_document",
                    "marketing_overview",
                ],
                "image_requirements": [
                    "brand_visuals",
                    "campaign_graphics",
                    "social_media_content",
                    "marketing_materials",
                    "promotional_images",
                    "logo_variants",
                    "advertisement_graphics",
                    "presentation_slides",
                    "infographic_design",
                    "visual_branding",
                ],
                "contexts": [
                    "campaign development",
                    "brand strategy session",
                    "client presentation",
                    "creative review meeting",
                    "marketing planning",
                    "brand audit",
                    "campaign launch",
                    "strategy workshop",
                ],
                "color_schemes": ["#f59e0b", "#d97706", "#b45309"],  # Marketing orange
            },
            "healthcare_research": {
                "companies": [
                    "MedResearch Institute",
                    "Healthcare Innovation Lab",
                    "Clinical Studies Center",
                    "Medical Research Foundation",
                    "Health Technology Institute",
                    "Biomedical Research Center",
                    "Healthcare Analytics Group",
                    "Medical Innovation Lab",
                    "Clinical Research Institute",
                    "Health Science Center",
                    "Medical Technology Research",
                    "Healthcare Development Institute",
                    "Clinical Innovation Center",
                    "Medical Studies Foundation",
                    "Health Research Lab",
                    "Biomedical Innovation Institute",
                    "Healthcare Research Group",
                    "Medical Science Center",
                    "Clinical Development Lab",
                    "Health Technology Research",
                ],
                "research_areas": [
                    "clinical_trials",
                    "medical_devices",
                    "healthcare_technology",
                    "patient_outcomes",
                    "treatment_efficacy",
                    "diagnostic_methods",
                    "therapeutic_research",
                    "medical_innovation",
                    "health_informatics",
                    "preventive_care",
                    "pharmaceutical_research",
                    "medical_imaging",
                    "health_analytics",
                    "disease_prevention",
                    "patient_care",
                ],
                "document_types": [
                    "clinical_report",
                    "research_findings",
                    "medical_summary",
                    "health_assessment",
                    "treatment_protocol",
                    "research_proposal",
                    "clinical_overview",
                    "medical_analysis",
                    "health_study",
                    "research_documentation",
                ],
                "image_requirements": [
                    "medical_charts",
                    "diagnostic_images",
                    "health_infographics",
                    "clinical_diagrams",
                    "research_visuals",
                    "medical_illustrations",
                    "data_charts",
                    "treatment_flows",
                    "health_graphics",
                    "research_presentations",
                ],
                "contexts": [
                    "clinical study review",
                    "medical research presentation",
                    "health assessment report",
                    "treatment analysis",
                    "research grant application",
                    "medical conference",
                    "clinical trial documentation",
                    "health technology evaluation",
                ],
                "color_schemes": ["#059669", "#10b981", "#34d399"],  # Healthcare green
            },
            "financial_services": {
                "companies": [
                    "Financial Analytics Corp",
                    "Investment Research Group",
                    "Market Intelligence Solutions",
                    "Financial Strategy Institute",
                    "Capital Markets Research",
                    "Investment Analysis Center",
                    "Financial Innovation Lab",
                    "Market Research Institute",
                    "Capital Strategy Group",
                    "Financial Intelligence Solutions",
                    "Investment Technology Research",
                    "Market Analytics Corporation",
                    "Financial Development Institute",
                    "Capital Research Center",
                    "Investment Innovation Lab",
                    "Financial Research Solutions",
                    "Market Strategy Institute",
                    "Investment Analytics Group",
                    "Capital Intelligence Center",
                    "Financial Technology Research",
                ],
                "research_areas": [
                    "market_analysis",
                    "investment_research",
                    "financial_modeling",
                    "risk_assessment",
                    "portfolio_analysis",
                    "economic_trends",
                    "financial_planning",
                    "investment_strategies",
                    "market_forecasting",
                    "financial_technology",
                    "regulatory_analysis",
                    "credit_research",
                    "asset_management",
                    "financial_analytics",
                    "market_intelligence",
                ],
                "document_types": [
                    "investment_report",
                    "financial_analysis",
                    "market_outlook",
                    "risk_assessment",
                    "portfolio_review",
                    "financial_summary",
                    "investment_proposal",
                    "market_research",
                    "economic_report",
                    "financial_overview",
                ],
                "image_requirements": [
                    "financial_charts",
                    "market_graphs",
                    "investment_visuals",
                    "portfolio_graphics",
                    "economic_diagrams",
                    "financial_infographics",
                    "trend_charts",
                    "performance_graphs",
                    "analysis_visuals",
                    "market_presentations",
                ],
                "contexts": [
                    "investment committee meeting",
                    "client portfolio review",
                    "market analysis presentation",
                    "financial planning session",
                    "risk assessment review",
                    "investment strategy meeting",
                    "market outlook briefing",
                    "financial advisory consultation",
                ],
                "color_schemes": ["#1e40af", "#3b82f6", "#60a5fa"],  # Financial blue
            },
        }

        # Image processing scenarios with realistic business contexts
        self.image_scenarios = {
            "web_optimization": {
                "purposes": ["website_banner", "social_media_post", "blog_header", "landing_page_image", "email_newsletter"],
                "dimensions": [(1200, 400), (1024, 512), (800, 600), (1920, 1080), (1200, 630)],
                "contexts": ["web publication", "social media campaign", "digital marketing", "online presentation", "website redesign"],
            },
            "print_preparation": {
                "purposes": ["report_cover", "presentation_slide", "document_header", "infographic_element", "publication_image"],
                "dimensions": [(2100, 1500), (1800, 1200), (2400, 1600), (3000, 2000), (2560, 1440)],
                "contexts": ["printed report", "presentation material", "publication design", "document formatting", "professional printing"],
            },
            "presentation_ready": {
                "purposes": ["slide_background", "presentation_graphic", "meeting_visual", "conference_slide", "training_material"],
                "dimensions": [(1920, 1080), (1600, 900), (1280, 720), (1366, 768), (1440, 900)],
                "contexts": ["business presentation", "conference talk", "training session", "client meeting", "executive briefing"],
            },
            "social_media": {
                "purposes": ["linkedin_post", "twitter_header", "facebook_cover", "instagram_story", "professional_profile"],
                "dimensions": [(1200, 627), (1500, 500), (1200, 630), (1080, 1920), (400, 400)],
                "contexts": ["social media marketing", "professional networking", "brand promotion", "content marketing", "digital engagement"],
            },
            "document_integration": {
                "purposes": ["inline_graphic", "document_illustration", "report_visual", "text_accompanying_image", "explanatory_diagram"],
                "dimensions": [(800, 600), (640, 480), (1024, 768), (600, 400), (900, 675)],
                "contexts": ["document design", "report creation", "content illustration", "professional documentation", "visual communication"],
            },
        }

        # Professional HTML templates with realistic layouts
        self.html_templates = {
            "corporate_portal": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title} - {company_name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f8fafc; }}
        .header {{ background: linear-gradient(135deg, {primary_color}, {secondary_color}); color: white; padding: 20px 0; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 0 20px; }}
        .nav {{ display: flex; justify-content: space-between; align-items: center; }}
        .logo {{ font-size: 24px; font-weight: bold; }}
        .nav-links {{ display: flex; gap: 20px; }}
        .nav-links a {{ color: white; text-decoration: none; font-weight: 500; }}
        .main-content {{ padding: 40px 20px; }}
        .content-section {{ background: white; border-radius: 8px; padding: 30px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .highlight-box {{ background: {accent_color}; color: white; padding: 20px; border-radius: 6px; margin: 20px 0; }}
        .footer {{ background: #1f2937; color: white; padding: 20px 0; text-align: center; }}
        .key-finding {{ background: #f0f9ff; border-left: 4px solid {primary_color}; padding: 15px; margin: 15px 0; }}
        .research-data {{ font-weight: bold; color: {primary_color}; font-size: 18px; }}
    </style>
</head>
<body>
    <header class="header">
        <div class="container">
            <nav class="nav">
                <div class="logo">{company_name}</div>
                <div class="nav-links">
                    <a href="#research">Research</a>
                    <a href="#insights">Insights</a>
                    <a href="#reports">Reports</a>
                    <a href="#contact">Contact</a>
                </div>
            </nav>
        </div>
    </header>
    
    <main class="main-content">
        <div class="container">
            <div class="content-section">
                <h1>{research_title}</h1>
                <p class="meta">Published: {publish_date} | Category: {research_category}</p>
                
                <div class="key-finding">
                    <h3>Key Research Finding</h3>
                    <p class="research-data">{key_finding}</p>
                </div>
                
                <h2>Research Overview</h2>
                <p>{research_overview}</p>
                
                <h2>Detailed Analysis</h2>
                <p>{detailed_analysis}</p>
                
                <div class="highlight-box">
                    <h3>Critical Insight</h3>
                    <p>{critical_insight}</p>
                </div>
                
                <h2>Methodology</h2>
                <p>{methodology_description}</p>
                
                <h2>Implications</h2>
                <p>{implications}</p>
            </div>
        </div>
    </main>
    
    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 {company_name}. All rights reserved. | Research Department</p>
        </div>
    </footer>
</body>
</html>""",
            "academic_journal": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{research_title} | {institution_name}</title>
    <style>
        body {{ font-family: 'Times New Roman', serif; margin: 0; padding: 0; background-color: #ffffff; line-height: 1.6; }}
        .header {{ background: {primary_color}; color: white; padding: 15px 0; }}
        .container {{ max-width: 900px; margin: 0 auto; padding: 0 20px; }}
        .journal-header {{ text-align: center; border-bottom: 2px solid {primary_color}; padding-bottom: 20px; }}
        .institution {{ font-size: 14px; color: #666; margin-bottom: 10px; }}
        .main-content {{ padding: 30px 20px; }}
        .abstract {{ background: #f9fafb; border-left: 4px solid {primary_color}; padding: 20px; margin: 20px 0; }}
        .section {{ margin: 25px 0; }}
        .finding-highlight {{ background: #fffbeb; border: 1px solid {accent_color}; padding: 15px; margin: 15px 0; border-radius: 4px; }}
        .research-metric {{ font-weight: bold; color: {primary_color}; font-size: 16px; }}
        .citation {{ font-style: italic; color: #666; margin: 10px 0; }}
        .keywords {{ background: #f3f4f6; padding: 10px; border-radius: 4px; margin: 20px 0; }}
    </style>
</head>
<body>
    <header class="header">
        <div class="container">
            <div class="journal-header">
                <div class="institution">{institution_name}</div>
                <h1>{research_title}</h1>
                <p>Research Publication | {publish_date}</p>
            </div>
        </div>
    </header>
    
    <main class="main-content">
        <div class="container">
            <div class="section">
                <h2>Abstract</h2>
                <div class="abstract">
                    <p>{research_abstract}</p>
                </div>
                
                <div class="keywords">
                    <strong>Keywords:</strong> {research_keywords}
                </div>
            </div>
            
            <div class="section">
                <h2>Introduction</h2>
                <p>{introduction_text}</p>
            </div>
            
            <div class="section">
                <h2>Research Findings</h2>
                <div class="finding-highlight">
                    <h3>Primary Discovery</h3>
                    <p class="research-metric">{key_finding}</p>
                </div>
                <p>{findings_detail}</p>
            </div>
            
            <div class="section">
                <h2>Methodology</h2>
                <p>{methodology_description}</p>
            </div>
            
            <div class="section">
                <h2>Discussion</h2>
                <p>{discussion_content}</p>
            </div>
            
            <div class="section">
                <h2>Conclusion</h2>
                <p>{conclusion_text}</p>
            </div>
            
            <div class="citation">
                {citation_format}
            </div>
        </div>
    </main>
</body>
</html>""",
            "news_article": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{article_title} - {publication_name}</title>
    <style>
        body {{ font-family: 'Georgia', serif; margin: 0; padding: 0; background-color: #ffffff; }}
        .header {{ background: {primary_color}; color: white; padding: 10px 0; }}
        .container {{ max-width: 800px; margin: 0 auto; padding: 0 20px; }}
        .masthead {{ display: flex; justify-content: space-between; align-items: center; }}
        .publication-name {{ font-size: 28px; font-weight: bold; }}
        .date {{ font-size: 14px; }}
        .main-content {{ padding: 20px; }}
        .article-header {{ border-bottom: 3px solid {primary_color}; padding-bottom: 15px; margin-bottom: 20px; }}
        .headline {{ font-size: 36px; font-weight: bold; line-height: 1.2; margin-bottom: 10px; }}
        .byline {{ color: #666; font-size: 14px; margin-bottom: 15px; }}
        .lead {{ font-size: 18px; font-weight: 500; color: #333; margin-bottom: 25px; line-height: 1.4; }}
        .article-body {{ font-size: 16px; line-height: 1.7; }}
        .pull-quote {{ background: {accent_color}; color: white; padding: 20px; margin: 25px 0; font-size: 20px; font-style: italic; text-align: center; }}
        .news-highlight {{ background: #fff7ed; border-left: 4px solid {primary_color}; padding: 15px; margin: 20px 0; }}
        .breaking-news {{ background: #dc2626; color: white; padding: 10px; text-align: center; font-weight: bold; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <header class="header">
        <div class="container">
            <div class="masthead">
                <div class="publication-name">{publication_name}</div>
                <div class="date">{publish_date}</div>
            </div>
        </div>
    </header>
    
    <main class="main-content">
        <div class="container">
            <article>
                <div class="article-header">
                    <h1 class="headline">{article_title}</h1>
                    <div class="byline">By {author_name} | {publication_date} | {category}</div>
                </div>
                
                <div class="lead">
                    {article_lead}
                </div>
                
                <div class="article-body">
                    <p>{opening_paragraph}</p>
                    
                    <div class="news-highlight">
                        <strong>Key Development:</strong> {key_finding}
                    </div>
                    
                    <p>{context_paragraph}</p>
                    
                    <div class="pull-quote">
                        "{expert_quote}"
                    </div>
                    
                    <p>{analysis_paragraph}</p>
                    
                    <p>{implications_paragraph}</p>
                    
                    <p>{conclusion_paragraph}</p>
                </div>
            </article>
        </div>
    </main>
</body>
</html>""",
            "marketing_brief": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{campaign_title} - {agency_name}</title>
    <style>
        body {{ font-family: 'Arial', sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); }}
        .header {{ background: {primary_color}; color: white; padding: 20px 0; }}
        .container {{ max-width: 1000px; margin: 0 auto; padding: 0 20px; }}
        .brand-header {{ display: flex; justify-content: space-between; align-items: center; }}
        .agency-name {{ font-size: 24px; font-weight: bold; }}
        .main-content {{ padding: 30px 20px; }}
        .content-card {{ background: white; border-radius: 12px; padding: 30px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .campaign-header {{ text-align: center; margin-bottom: 30px; }}
        .insight-box {{ background: {accent_color}; color: white; padding: 25px; border-radius: 8px; margin: 25px 0; }}
        .metric-highlight {{ background: #f0f9ff; border: 2px solid {primary_color}; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0; }}
        .strategy-point {{ background: #fef3c7; border-left: 4px solid {accent_color}; padding: 15px; margin: 15px 0; }}
        .creative-brief {{ font-size: 18px; font-weight: 500; color: {primary_color}; }}
        .action-item {{ background: #ecfdf5; border-left: 4px solid #10b981; padding: 12px; margin: 10px 0; }}
    </style>
</head>
<body>
    <header class="header">
        <div class="container">
            <div class="brand-header">
                <div class="agency-name">{agency_name}</div>
                <div>Campaign Brief | {creation_date}</div>
            </div>
        </div>
    </header>
    
    <main class="main-content">
        <div class="container">
            <div class="content-card">
                <div class="campaign-header">
                    <h1>{campaign_title}</h1>
                    <p class="creative-brief">{campaign_description}</p>
                </div>
                
                <h2>Market Research Insights</h2>
                <p>{market_research}</p>
                
                <div class="insight-box">
                    <h3>Key Market Finding</h3>
                    <p>{key_finding}</p>
                </div>
                
                <h2>Target Audience Analysis</h2>
                <p>{audience_analysis}</p>
                
                <div class="metric-highlight">
                    <h3>Performance Metric</h3>
                    <div class="creative-brief">{performance_metric}</div>
                </div>
                
                <h2>Strategic Recommendations</h2>
                <div class="strategy-point">
                    <strong>Primary Strategy:</strong> {primary_strategy}
                </div>
                
                <p>{strategy_details}</p>
                
                <h2>Implementation Plan</h2>
                <div class="action-item">
                    <strong>Next Steps:</strong> {implementation_steps}
                </div>
                
                <p>{timeline_details}</p>
            </div>
        </div>
    </main>
</body>
</html>""",
        }

    def set_seed(self, seed: int):
        """Set random seed for reproducible generation."""
        self.random.seed(seed)

    def select_domain_context(self, seed_offset: int = 0) -> Dict[str, Any]:
        """Select and generate domain context with realistic business scenario."""
        # Create temporary random generator for domain selection
        temp_random = random.Random()
        temp_random.seed(self.random.getstate()[1][0] + seed_offset)

        domain_name = temp_random.choice(list(self.domains.keys()))
        domain_info = self.domains[domain_name]

        # Generate comprehensive domain context
        context = {
            "domain": domain_name,
            "company": temp_random.choice(domain_info["companies"]),
            "research_area": temp_random.choice(domain_info["research_areas"]),
            "document_type": temp_random.choice(domain_info["document_types"]),
            "image_requirement": temp_random.choice(domain_info["image_requirements"]),
            "business_context": temp_random.choice(domain_info["contexts"]),
            "color_scheme": domain_info["color_schemes"],
            "primary_color": domain_info["color_schemes"][0],
            "secondary_color": domain_info["color_schemes"][1] if len(domain_info["color_schemes"]) > 1 else domain_info["color_schemes"][0],
            "accent_color": domain_info["color_schemes"][2] if len(domain_info["color_schemes"]) > 2 else domain_info["color_schemes"][0],
        }

        return context

    def generate_professional_scenario(self, domain_context: Dict[str, Any], seed_offset: int = 0) -> Dict[str, Any]:
        """Generate realistic professional scenario based on domain context."""
        temp_random = random.Random()
        temp_random.seed(self.random.getstate()[1][0] + seed_offset)

        domain = domain_context["domain"]

        # Domain-specific scenario generation
        scenarios = {
            "corporate_research": {
                "project_types": [
                    "market expansion study",
                    "competitive analysis project",
                    "technology assessment",
                    "business intelligence initiative",
                    "strategic planning research",
                ],
                "stakeholders": ["executive team", "board members", "investment committee", "strategic planning team", "business development"],
                "deliverables": [
                    "executive presentation",
                    "strategic recommendations",
                    "market analysis report",
                    "competitive landscape overview",
                    "business case document",
                ],
                "timelines": ["quarterly review", "board meeting", "strategic planning session", "investor presentation", "management briefing"],
            },
            "academic_research": {
                "project_types": [
                    "grant-funded study",
                    "collaborative research project",
                    "doctoral dissertation research",
                    "conference publication",
                    "peer-review submission",
                ],
                "stakeholders": ["research committee", "grant reviewers", "academic peers", "department faculty", "research collaborators"],
                "deliverables": ["research paper", "grant proposal", "conference presentation", "literature review", "methodology document"],
                "timelines": ["conference deadline", "grant submission", "thesis defense", "publication review", "research symposium"],
            },
            "journalism_media": {
                "project_types": ["investigative story", "feature article", "breaking news coverage", "industry analysis", "expert interview"],
                "stakeholders": ["editorial team", "readers", "news audience", "industry experts", "publication management"],
                "deliverables": ["news article", "feature story", "investigative report", "interview piece", "analysis article"],
                "timelines": ["publication deadline", "breaking news cycle", "weekly feature slot", "monthly magazine", "daily news coverage"],
            },
            "marketing_communications": {
                "project_types": [
                    "campaign development",
                    "brand strategy project",
                    "market research study",
                    "content strategy initiative",
                    "brand positioning research",
                ],
                "stakeholders": ["client team", "creative department", "account management", "brand stakeholders", "marketing executives"],
                "deliverables": ["campaign brief", "brand strategy document", "creative proposal", "market research report", "content strategy plan"],
                "timelines": ["campaign launch", "brand review meeting", "client presentation", "creative briefing", "strategy workshop"],
            },
            "healthcare_research": {
                "project_types": [
                    "clinical study",
                    "medical research project",
                    "health technology assessment",
                    "patient outcome analysis",
                    "treatment effectiveness study",
                ],
                "stakeholders": ["medical team", "research committee", "healthcare administrators", "regulatory bodies", "clinical staff"],
                "deliverables": ["clinical report", "research summary", "medical analysis", "treatment protocol", "health assessment"],
                "timelines": ["clinical review", "medical conference", "research deadline", "regulatory submission", "treatment evaluation"],
            },
            "financial_services": {
                "project_types": ["investment analysis", "market research study", "risk assessment project", "portfolio review", "economic forecast"],
                "stakeholders": ["investment committee", "portfolio managers", "risk management", "financial advisors", "client base"],
                "deliverables": ["investment report", "market analysis", "risk assessment", "portfolio review", "economic outlook"],
                "timelines": ["quarterly review", "investment meeting", "client presentation", "market briefing", "strategy session"],
            },
        }

        domain_scenarios = scenarios.get(domain, scenarios["corporate_research"])

        scenario = {
            "project_type": temp_random.choice(domain_scenarios["project_types"]),
            "primary_stakeholder": temp_random.choice(domain_scenarios["stakeholders"]),
            "expected_deliverable": temp_random.choice(domain_scenarios["deliverables"]),
            "business_timeline": temp_random.choice(domain_scenarios["timelines"]),
            "urgency_level": temp_random.choice(["standard", "high_priority", "urgent"]),
            "scope_complexity": temp_random.choice(["focused", "comprehensive", "strategic"]),
        }

        return scenario

    def generate_research_content(self, domain_context: Dict[str, Any], scenario: Dict[str, Any], seed_offset: int = 0) -> Dict[str, Any]:
        """Generate comprehensive research content including facts, analysis, and context."""
        temp_random = random.Random()
        temp_random.seed(self.random.getstate()[1][0] + seed_offset)

        domain = domain_context["domain"]
        research_area = domain_context["research_area"]

        # Generate domain-specific research facts with realistic data
        fact_templates = {
            "corporate_research": {
                "market_analysis": "Market research indicates {percentage}% growth in {sector} sector, with {company_type} companies leading adoption of {technology} solutions",
                "technology_trends": "Industry analysis shows {technology} implementation increased by {percentage}% in {timeframe}, driving {impact_metric} improvements",
                "customer_analytics": "Customer behavior analysis reveals {percentage}% preference for {solution_type}, indicating {market_trend} market shift",
                "business_intelligence": "Business intelligence data shows {metric_value}% increase in {performance_area} through {strategy_implementation}",
                "digital_transformation": "Digital transformation initiatives report {percentage}% efficiency gains in {business_area} operations over {timeframe}",
            },
            "academic_research": {
                "climate_science": "Climate research demonstrates {measurement_value}% change in {environmental_factor} over {study_period}, indicating {trend_direction} environmental trend",
                "biotechnology": "Biotechnology studies show {effectiveness_percentage}% improvement in {treatment_area} using {technology_approach} methodologies",
                "computer_science": "Computer science research achieves {performance_metric}% enhancement in {technical_area} through {algorithm_type} optimization",
                "materials_engineering": "Materials engineering analysis reveals {strength_value}% increased durability in {material_type} composite structures",
                "artificial_intelligence": "AI research demonstrates {accuracy_percentage}% accuracy improvement in {application_domain} prediction models",
            },
            "journalism_media": {
                "breaking_news": "Breaking investigation reveals {impact_percentage}% increase in {issue_area}, affecting {affected_population} across {geographic_scope}",
                "technology_features": "Technology analysis uncovers {adoption_rate}% user adoption of {tech_innovation}, transforming {industry_sector} practices",
                "business_analysis": "Business investigation shows {financial_metric}% revenue impact from {business_strategy} implementation in {market_segment}",
                "industry_coverage": "Industry report identifies {growth_percentage}% market expansion in {sector}, driven by {key_factor} developments",
                "expert_interviews": "Expert analysis confirms {trend_percentage}% shift toward {emerging_trend} in {professional_domain} sector",
            },
            "marketing_communications": {
                "brand_analysis": "Brand research identifies {engagement_percentage}% increase in consumer engagement through {strategy_approach} campaigns",
                "market_research": "Market analysis shows {preference_percentage}% consumer preference for {product_category} with {feature_emphasis} features",
                "digital_trends": "Digital marketing data reveals {conversion_rate}% improvement in {channel_type} performance using {tactic_type} strategies",
                "consumer_behavior": "Consumer behavior study indicates {behavior_percentage}% shift toward {consumption_pattern} in {demographic_group}",
                "campaign_effectiveness": "Campaign analysis demonstrates {roi_percentage}% return on investment for {campaign_type} initiatives in {target_market}",
            },
            "healthcare_research": {
                "clinical_trials": "Clinical study results show {efficacy_percentage}% treatment effectiveness in {patient_population} using {therapy_type} intervention",
                "medical_devices": "Medical device research demonstrates {improvement_percentage}% enhanced patient outcomes through {device_technology} implementation",
                "patient_outcomes": "Patient outcome analysis reveals {recovery_percentage}% faster recovery rates with {treatment_protocol} methodology",
                "health_informatics": "Health informatics data shows {efficiency_percentage}% operational improvement in {healthcare_process} through {system_type} integration",
                "preventive_care": "Preventive care research indicates {reduction_percentage}% decrease in {health_condition} through {intervention_strategy} programs",
            },
            "financial_services": {
                "investment_research": "Investment analysis reveals {return_percentage}% portfolio performance improvement through {strategy_type} allocation in {market_sector}",
                "market_analysis": "Market research identifies {growth_percentage}% potential in {investment_category} driven by {economic_factor} trends",
                "risk_assessment": "Risk analysis shows {risk_reduction}% decreased portfolio volatility using {risk_strategy} management approaches",
                "economic_trends": "Economic analysis predicts {forecast_metric}% growth in {economic_sector} based on {indicator_type} market indicators",
                "financial_planning": "Financial planning research demonstrates {optimization_percentage}% improved outcomes through {planning_approach} strategies",
            },
        }

        # Select appropriate template and generate values
        domain_templates = fact_templates.get(domain, fact_templates["corporate_research"])
        template = domain_templates.get(research_area, list(domain_templates.values())[0])

        # Generate realistic values based on domain
        values = self._generate_contextual_values(domain, temp_random)

        try:
            key_finding = template.format(**values)
        except KeyError:
            # Fallback if template formatting fails
            key_finding = f"Research shows significant progress in {research_area.replace('_', ' ')} with measurable improvements"

        # Generate supporting content
        content = {
            "key_finding": key_finding,
            "research_overview": self._generate_research_overview(domain_context, scenario, temp_random),
            "detailed_analysis": self._generate_detailed_analysis(domain_context, values, temp_random),
            "methodology": self._generate_methodology_description(domain, temp_random),
            "implications": self._generate_implications(domain_context, values, temp_random),
            "context_description": self._generate_context_description(domain_context, scenario),
            "supporting_data": values,
        }

        return content

    def _generate_contextual_values(self, domain: str, temp_random: random.Random) -> Dict[str, Any]:
        """Generate realistic values based on domain context."""
        base_values = {
            "percentage": temp_random.randint(15, 85),
            "metric_value": temp_random.randint(20, 95),
            "timeframe": temp_random.choice(["6 months", "12 months", "18 months", "2 years"]),
            "study_period": temp_random.choice(["3 years", "5 years", "10 years"]),
            "geographic_scope": temp_random.choice(["regional markets", "national industry", "global sector"]),
        }

        domain_specific = {
            "corporate_research": {
                "sector": temp_random.choice(["technology", "healthcare", "finance", "manufacturing", "retail"]),
                "technology": temp_random.choice(["AI solutions", "cloud platforms", "automation tools", "analytics systems"]),
                "company_type": temp_random.choice(["enterprise", "mid-market", "startup", "Fortune 500"]),
                "business_area": temp_random.choice(["operations", "customer service", "supply chain", "marketing"]),
                "strategy_implementation": temp_random.choice(["process optimization", "technology adoption", "digital transformation"]),
            },
            "academic_research": {
                "environmental_factor": temp_random.choice(["temperature patterns", "precipitation levels", "carbon emissions"]),
                "treatment_area": temp_random.choice(["disease treatment", "diagnostic accuracy", "patient recovery"]),
                "technical_area": temp_random.choice(["processing speed", "algorithm efficiency", "data accuracy"]),
                "algorithm_type": temp_random.choice(["machine learning", "deep learning", "neural network"]),
                "application_domain": temp_random.choice(["medical diagnosis", "financial forecasting", "image recognition"]),
            },
            "journalism_media": {
                "issue_area": temp_random.choice(["technology adoption", "market activity", "industry trends"]),
                "affected_population": temp_random.choice(["businesses", "consumers", "professionals", "organizations"]),
                "tech_innovation": temp_random.choice(["mobile platforms", "cloud services", "AI applications"]),
                "industry_sector": temp_random.choice(["healthcare", "finance", "education", "manufacturing"]),
                "emerging_trend": temp_random.choice(["automation", "sustainability", "digitalization"]),
            },
        }

        # Merge base values with domain-specific values
        result = {**base_values, **domain_specific.get(domain, {})}
        return result

    def _generate_research_overview(self, domain_context: Dict[str, Any], scenario: Dict[str, Any], temp_random: random.Random) -> str:
        """Generate comprehensive research overview."""
        company = domain_context["company"]
        research_area = domain_context["research_area"].replace("_", " ")
        project_type = scenario["project_type"]

        overviews = [
            f"This comprehensive {project_type} conducted by {company} examines current trends and developments in {research_area}. The research encompasses multiple data sources and analytical approaches to provide actionable insights for strategic decision-making.",
            f"Our research team at {company} has undertaken an extensive {project_type} focusing on {research_area}. This analysis combines quantitative data analysis with qualitative insights to deliver a thorough understanding of market dynamics and emerging opportunities.",
            f"The {project_type} represents a significant research initiative by {company} to analyze {research_area} trends and their implications. Through systematic data collection and analysis, this study provides evidence-based recommendations for stakeholders.",
        ]

        return temp_random.choice(overviews)

    def _generate_detailed_analysis(self, domain_context: Dict[str, Any], values: Dict[str, Any], temp_random: random.Random) -> str:
        """Generate detailed analysis content."""
        domain = domain_context["domain"]
        research_area = domain_context["research_area"].replace("_", " ")

        analyses = {
            "corporate_research": [
                f"The analysis reveals significant market dynamics in {research_area}, with performance metrics indicating sustained growth patterns. Key performance indicators demonstrate measurable improvements across multiple business dimensions, suggesting strong market fundamentals and positive outlook for continued development.",
                f"Market assessment shows robust activity in {research_area} sector, with stakeholder engagement reaching new levels. The data suggests strategic opportunities for organizations willing to invest in innovative approaches and technologies that address current market needs.",
                f"Our comprehensive evaluation of {research_area} trends indicates substantial market potential, supported by strong fundamentals and growing stakeholder interest. The research identifies key success factors that differentiate high-performing organizations in this competitive landscape.",
            ],
            "academic_research": [
                f"The scientific investigation into {research_area} reveals important findings that contribute to the existing body of knowledge. Methodology validation and peer review processes confirm the reliability of results and their significance for future research directions.",
                f"Research findings in {research_area} demonstrate measurable outcomes that advance theoretical understanding and practical applications. The study's rigorous methodology and comprehensive data analysis provide a solid foundation for continued investigation.",
                f"This scholarly examination of {research_area} contributes valuable insights to the academic community, with implications for both theoretical frameworks and practical implementations. The research methodology ensures reproducibility and scientific validity.",
            ],
        }

        domain_analyses = analyses.get(domain, analyses["corporate_research"])
        return temp_random.choice(domain_analyses)

    def _generate_methodology_description(self, domain: str, temp_random: random.Random) -> str:
        """Generate methodology description based on domain."""
        methodologies = {
            "corporate_research": [
                "The research methodology combines quantitative market analysis with qualitative stakeholder interviews. Data collection included industry surveys, financial performance analysis, and competitive benchmarking to ensure comprehensive market coverage.",
                "Our analytical approach integrates multiple data sources including market research databases, industry reports, and primary research interviews. Statistical analysis and trend modeling provide robust insights into market dynamics.",
                "The study employs a mixed-methods approach combining statistical analysis of market data with structured interviews of industry experts. This methodology ensures both depth and breadth in research findings.",
            ],
            "academic_research": [
                "The research follows established scientific protocols with controlled experimental design and peer-reviewed methodology. Data collection procedures adhere to institutional standards for research integrity and reproducibility.",
                "Our methodology incorporates systematic literature review, experimental design, and statistical analysis following established academic standards. Quality assurance processes ensure research validity and reliability.",
                "The study design follows rigorous academic standards with appropriate controls, statistical analysis, and peer review processes. Methodology validation ensures scientific integrity and reproducible results.",
            ],
        }

        domain_methods = methodologies.get(domain, methodologies["corporate_research"])
        return temp_random.choice(domain_methods)

    def _generate_implications(self, domain_context: Dict[str, Any], values: Dict[str, Any], temp_random: random.Random) -> str:
        """Generate implications and recommendations."""
        domain = domain_context["domain"]
        research_area = domain_context["research_area"].replace("_", " ")

        implications = {
            "corporate_research": [
                f"The research findings have significant implications for strategic planning in {research_area}. Organizations should consider these insights when developing long-term strategies and operational plans to maintain competitive advantage.",
                f"These results suggest important opportunities for innovation and growth in {research_area}. Strategic implementation of these findings could drive substantial business value and market differentiation.",
                f"The analysis indicates critical success factors for {research_area} initiatives. Organizations that align their strategies with these insights are likely to achieve superior performance outcomes.",
            ],
            "academic_research": [
                f"The research contributes important knowledge to {research_area} and opens new avenues for future investigation. These findings have implications for both theoretical understanding and practical applications.",
                f"Results provide valuable insights for {research_area} research community and suggest directions for continued scholarly investigation. The findings contribute to the advancement of knowledge in this field.",
                f"This research advances understanding of {research_area} and provides a foundation for future studies. The implications extend to both academic research and practical applications.",
            ],
        }

        domain_implications = implications.get(domain, implications["corporate_research"])
        return temp_random.choice(domain_implications)

    def _generate_context_description(self, domain_context: Dict[str, Any], scenario: Dict[str, Any]) -> str:
        """Generate context description for task instructions."""
        domain_name = domain_context["domain"].replace("_", " ")
        business_context = domain_context["business_context"]
        document_type = domain_context["document_type"].replace("_", " ")

        return f"{domain_name} {document_type} for {business_context}"

    def generate_image_scenario(self, domain_context: Dict[str, Any], seed_offset: int = 0) -> Dict[str, Any]:
        """Generate realistic image processing scenario based on domain context."""
        temp_random = random.Random()
        temp_random.seed(self.random.getstate()[1][0] + seed_offset)

        # Select appropriate image scenario based on domain
        scenario_type = temp_random.choice(list(self.image_scenarios.keys()))
        scenario_info = self.image_scenarios[scenario_type]

        selected_purpose = temp_random.choice(scenario_info["purposes"])
        selected_dimensions = temp_random.choice(scenario_info["dimensions"])
        selected_context = temp_random.choice(scenario_info["contexts"])

        # Generate domain-appropriate image file names
        domain = domain_context["domain"]
        image_prefixes = {
            "corporate_research": ["business", "corporate", "market", "strategy", "analysis"],
            "academic_research": ["research", "scientific", "academic", "study", "data"],
            "journalism_media": ["news", "article", "story", "feature", "editorial"],
            "marketing_communications": ["brand", "campaign", "marketing", "creative", "promotional"],
            "healthcare_research": ["medical", "health", "clinical", "research", "patient"],
            "financial_services": ["financial", "investment", "market", "economic", "portfolio"],
        }

        prefix = temp_random.choice(image_prefixes.get(domain, image_prefixes["corporate_research"]))
        suffix = temp_random.randint(100, 999)

        return {
            "scenario_type": scenario_type,
            "purpose": selected_purpose,
            "dimensions": selected_dimensions,
            "width": selected_dimensions[0],
            "height": selected_dimensions[1],
            "context": selected_context,
            "source_filename": f"{prefix}_source_{suffix}.png",
            "processed_filename": f"{prefix}_processed_{suffix}.png",
            "processing_description": f"resize for {selected_purpose} ({selected_context})",
        }

    def generate_professional_html_content(self, domain_context: Dict[str, Any], research_content: Dict[str, Any], seed_offset: int = 0) -> str:
        """Generate professional HTML content using domain-appropriate templates."""
        temp_random = random.Random()
        temp_random.seed(self.random.getstate()[1][0] + seed_offset)

        domain = domain_context["domain"]

        # Select appropriate template based on domain
        template_mapping = {
            "corporate_research": "corporate_portal",
            "academic_research": "academic_journal",
            "journalism_media": "news_article",
            "marketing_communications": "marketing_brief",
            "healthcare_research": "academic_journal",  # Use academic template for healthcare
            "financial_services": "corporate_portal",  # Use corporate template for financial
        }

        template_key = template_mapping.get(domain, "corporate_portal")
        template = self.html_templates[template_key]

        # Generate template variables
        template_vars = self._generate_template_variables(domain_context, research_content, temp_random)

        try:
            return template.format(**template_vars)
        except KeyError:
            # Fallback template if formatting fails
            return self._generate_fallback_html(domain_context, research_content)

    def _generate_template_variables(
        self, domain_context: Dict[str, Any], research_content: Dict[str, Any], temp_random: random.Random
    ) -> Dict[str, Any]:
        """Generate all template variables for HTML content."""
        domain = domain_context["domain"]
        company = domain_context["company"]
        research_area = domain_context["research_area"].replace("_", " ").title()

        # Current date formatting
        current_date = datetime.now()
        formatted_date = current_date.strftime("%B %d, %Y")

        # Base variables common to all templates
        base_vars = {
            "company_name": company,
            "institution_name": company,
            "publication_name": company,
            "agency_name": company,
            "research_title": f"{research_area} Analysis Report",
            "page_title": f"{research_area} Research",
            "publish_date": formatted_date,
            "creation_date": formatted_date,
            "publication_date": formatted_date,
            "primary_color": domain_context["primary_color"],
            "secondary_color": domain_context["secondary_color"],
            "accent_color": domain_context["accent_color"],
            "key_finding": research_content["key_finding"],
            "research_overview": research_content["research_overview"],
            "detailed_analysis": research_content["detailed_analysis"],
            "methodology_description": research_content["methodology"],
            "implications": research_content["implications"],
        }

        # Domain-specific variables
        domain_vars = {
            "corporate_research": {
                "research_category": "Business Intelligence",
                "critical_insight": research_content["implications"],
                "campaign_title": f"{research_area} Strategic Initiative",
                "campaign_description": f"Comprehensive analysis of {research_area} market opportunities",
            },
            "academic_research": {
                "research_abstract": research_content["research_overview"],
                "research_keywords": f"{research_area}, research, analysis, methodology, findings",
                "introduction_text": research_content["detailed_analysis"],
                "findings_detail": research_content["implications"],
                "discussion_content": research_content["methodology"],
                "conclusion_text": research_content["implications"],
                "citation_format": f"Author. ({current_date.year}). {research_area} Analysis. {company}.",
            },
            "journalism_media": {
                "article_title": f"Breaking: New Developments in {research_area}",
                "author_name": temp_random.choice(["Sarah Johnson", "Michael Chen", "Dr. Emily Rodriguez", "James Thompson"]),
                "category": research_area,
                "article_lead": research_content["research_overview"][:200] + "...",
                "opening_paragraph": research_content["research_overview"],
                "context_paragraph": research_content["detailed_analysis"],
                "expert_quote": f"This represents a significant development in {research_area}",
                "analysis_paragraph": research_content["methodology"],
                "implications_paragraph": research_content["implications"],
                "conclusion_paragraph": f"The implications of this {research_area} development continue to unfold.",
            },
            "marketing_communications": {
                "campaign_title": f"{research_area} Marketing Initiative",
                "campaign_description": f"Strategic marketing approach for {research_area} market segment",
                "market_research": research_content["research_overview"],
                "audience_analysis": research_content["detailed_analysis"],
                "performance_metric": research_content["key_finding"],
                "primary_strategy": research_content["implications"],
                "strategy_details": research_content["methodology"],
                "implementation_steps": f"Execute {research_area} marketing strategy",
                "timeline_details": "Implementation planned over next quarter",
            },
        }

        # Merge base variables with domain-specific variables
        result = {**base_vars, **domain_vars.get(domain, {})}
        return result

    def _generate_fallback_html(self, domain_context: Dict[str, Any], research_content: Dict[str, Any]) -> str:
        """Generate fallback HTML if template formatting fails."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report - {domain_context["company"]}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: {domain_context["primary_color"]}; color: white; padding: 20px; margin-bottom: 30px; }}
        .finding {{ background: #f0f9ff; border-left: 4px solid {domain_context["primary_color"]}; padding: 15px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{domain_context["research_area"].replace("_", " ").title()} Research</h1>
        <p>{domain_context["company"]}</p>
    </div>
    
    <div class="finding">
        <h3>Key Finding</h3>
        <p>{research_content["key_finding"]}</p>
    </div>
    
    <h2>Overview</h2>
    <p>{research_content["research_overview"]}</p>
    
    <h2>Analysis</h2>
    <p>{research_content["detailed_analysis"]}</p>
    
    <h2>Methodology</h2>
    <p>{research_content["methodology"]}</p>
    
    <h2>Implications</h2>
    <p>{research_content["implications"]}</p>
</body>
</html>"""

    def generate_document_requirements(self, domain_context: Dict[str, Any], scenario: Dict[str, Any], seed_offset: int = 0) -> Dict[str, Any]:
        """Generate realistic document creation requirements."""
        temp_random = random.Random()
        temp_random.seed(self.random.getstate()[1][0] + seed_offset)

        domain = domain_context["domain"]

        # Domain-specific document requirements
        requirements = {
            "corporate_research": {
                "structure": ["Executive Summary", "Market Analysis", "Key Findings", "Strategic Recommendations"],
                "formatting": ["professional fonts", "company branding", "consistent spacing", "clear headings"],
                "content_elements": ["research data", "visual elements", "executive summary", "action items"],
                "file_naming": f"{domain_context['research_area']}_report_{temp_random.randint(100, 999)}.docx",
            },
            "academic_research": {
                "structure": ["Abstract", "Introduction", "Methodology", "Results", "Discussion", "Conclusion"],
                "formatting": ["academic style", "proper citations", "structured layout", "scientific formatting"],
                "content_elements": ["research findings", "methodology description", "data analysis", "references"],
                "file_naming": f"{domain_context['research_area']}_paper_{temp_random.randint(100, 999)}.docx",
            },
            "journalism_media": {
                "structure": ["Headline", "Lead Paragraph", "Body Content", "Supporting Details", "Conclusion"],
                "formatting": ["journalistic style", "engaging layout", "clear structure", "readable fonts"],
                "content_elements": ["news content", "quotes", "factual information", "story narrative"],
                "file_naming": f"{domain_context['research_area']}_article_{temp_random.randint(100, 999)}.docx",
            },
        }

        domain_req = requirements.get(domain, requirements["corporate_research"])

        return {
            "document_structure": domain_req["structure"],
            "formatting_requirements": domain_req["formatting"],
            "required_elements": domain_req["content_elements"],
            "output_filename": domain_req["file_naming"],
            "professional_tone": True,
            "include_research_content": True,
            "include_processed_image": True,
        }

    def get_content_variation_count(self) -> int:
        """Calculate total possible content variations."""
        domain_count = len(self.domains)
        avg_companies = sum(len(domain["companies"]) for domain in self.domains.values()) // domain_count
        avg_research_areas = sum(len(domain["research_areas"]) for domain in self.domains.values()) // domain_count
        avg_document_types = sum(len(domain["document_types"]) for domain in self.domains.values()) // domain_count
        image_scenario_count = sum(len(scenario["purposes"]) * len(scenario["dimensions"]) for scenario in self.image_scenarios.values())

        base_combinations = domain_count * avg_companies * avg_research_areas * avg_document_types * image_scenario_count

        # Content generation creates additional variations through dynamic facts and context
        content_multiplier = 10  # Each base combination can generate ~10 different content variants

        return base_combinations * content_multiplier
