"""
Enhanced Content Generator for Information Synthesis & Presentation tasks.
Provides domain-based content generation with professional HTML templates.
"""

import random
from typing import Dict, Any
from datetime import datetime, timedelta


class SimplifiedContentGenerator:
    """Enhanced content generator with domain-based variability and professional HTML templates."""

    def __init__(self):
        self.random = random.Random()

        # Domain data with realistic business information
        self.domains = {
            "technology": {
                "companies": [
                    "TechFlow",
                    "DataCorp",
                    "InnovateAI",
                    "CodeCraft",
                    "ByteStream",
                    "CloudTech",
                    "DevTools Pro",
                    "SmartCode",
                    "TechVision",
                    "DigitalEdge",
                    "CyberSoft",
                    "NetLogic",
                    "SystemCore",
                    "DataMind",
                    "TechForge",
                    "CodeWorks",
                    "DigitalPro",
                    "InfoTech",
                    "SoftLine",
                    "TechBase",
                    "DataFlow",
                    "CloudCore",
                    "WebTech",
                    "CodeLab",
                    "TechSuite",
                ],
                "price_range": (99, 1999),
                "founded_range": (1995, 2015),
                "products": [
                    "Software Platform",
                    "Mobile App",
                    "AI Tool",
                    "Cloud Service",
                    "Development Framework",
                    "Data Analytics Suite",
                    "Security Software",
                    "Project Management Tool",
                    "API Gateway",
                    "Machine Learning Platform",
                ],
                "colors": ["#2563eb", "#1e40af", "#3730a3"],  # Blue tones
            },
            "healthcare": {
                "companies": [
                    "MedCare",
                    "HealthTech",
                    "WellnessPlus",
                    "CareLink",
                    "MedFlow",
                    "HealthCore",
                    "MedSoft",
                    "CareSystem",
                    "HealthPro",
                    "MedNet",
                    "WellCare",
                    "HealthLink",
                    "MedBase",
                    "CareFlow",
                    "HealthSuite",
                    "MedLab",
                    "CarePoint",
                    "HealthEdge",
                    "MedSync",
                    "CareTrack",
                    "HealthVision",
                    "MedCore",
                    "CareLogic",
                    "HealthForge",
                    "MedTech Pro",
                ],
                "price_range": (29, 299),
                "founded_range": (1985, 2010),
                "products": [
                    "Health Monitor",
                    "Medical Device",
                    "Wellness App",
                    "Patient Portal",
                    "Telemedicine Platform",
                    "Health Tracker",
                    "Medical Software",
                    "Diagnostic Tool",
                    "Health Analytics",
                    "Patient Management System",
                ],
                "colors": ["#059669", "#047857", "#065f46"],  # Green tones
            },
            "finance": {
                "companies": [
                    "FinTech Pro",
                    "MoneyFlow",
                    "CreditCore",
                    "PayLink",
                    "FinBase",
                    "BankTech",
                    "PayFlow",
                    "FinSoft",
                    "MoneyTech",
                    "CreditPro",
                    "FinCore",
                    "PaySystem",
                    "MoneyCore",
                    "FinLogic",
                    "BankFlow",
                    "PayTech",
                    "FinEdge",
                    "MoneyLink",
                    "CreditFlow",
                    "PayCore",
                    "FinSuite",
                    "BankPro",
                    "MoneyNet",
                    "CreditTech",
                    "PayLogic",
                ],
                "price_range": (199, 2999),
                "founded_range": (1990, 2010),
                "products": [
                    "Trading Platform",
                    "Payment Gateway",
                    "Financial Analytics",
                    "Investment Tool",
                    "Banking Software",
                    "Credit Management",
                    "Risk Assessment Tool",
                    "Portfolio Tracker",
                    "Accounting Software",
                    "Financial Dashboard",
                ],
                "colors": ["#dc2626", "#b91c1c", "#991b1b"],  # Red tones
            },
            "education": {
                "companies": [
                    "EduTech",
                    "LearnFlow",
                    "StudyPro",
                    "EduCore",
                    "LearnLink",
                    "StudyTech",
                    "EduSoft",
                    "LearnPro",
                    "StudyFlow",
                    "EduNet",
                    "LearnCore",
                    "StudyLink",
                    "EduLogic",
                    "LearnTech",
                    "StudyBase",
                    "EduFlow",
                    "LearnSoft",
                    "StudyCore",
                    "EduEdge",
                    "LearnBase",
                    "StudyNet",
                    "EduPro",
                    "LearnLogic",
                    "StudyEdge",
                    "EduSuite",
                ],
                "price_range": (49, 499),
                "founded_range": (1988, 2012),
                "products": [
                    "Learning Platform",
                    "Study App",
                    "Course Management System",
                    "Student Portal",
                    "Educational Software",
                    "Virtual Classroom",
                    "Assessment Tool",
                    "Learning Analytics",
                    "Study Tracker",
                    "Educational Content Platform",
                ],
                "colors": ["#7c3aed", "#6d28d9", "#5b21b6"],  # Purple tones
            },
            "retail": {
                "companies": [
                    "ShopTech",
                    "RetailPro",
                    "StoreFlow",
                    "SaleTech",
                    "ShopCore",
                    "RetailFlow",
                    "StorePro",
                    "SaleCore",
                    "ShopLink",
                    "RetailNet",
                    "StoreCore",
                    "SaleFlow",
                    "ShopLogic",
                    "RetailBase",
                    "StoreTech",
                    "SaleLink",
                    "ShopEdge",
                    "RetailLogic",
                    "StoreLink",
                    "SalePro",
                    "ShopNet",
                    "RetailCore",
                    "StoreSoft",
                    "SaleEdge",
                    "ShopSuite",
                ],
                "price_range": (19, 199),
                "founded_range": (1992, 2015),
                "products": [
                    "E-commerce Platform",
                    "POS System",
                    "Inventory Management",
                    "Customer Portal",
                    "Sales Analytics",
                    "Product Catalog",
                    "Order Management",
                    "Retail Software",
                    "Store Management",
                    "Customer Relationship Tool",
                ],
                "colors": ["#ea580c", "#dc2626", "#b91c1c"],  # Orange/Red tones
            },
            "manufacturing": {
                "companies": [
                    "ManuTech",
                    "ProducePro",
                    "FactoryFlow",
                    "MakeCore",
                    "BuildTech",
                    "ManuFlow",
                    "ProduceTech",
                    "FactoryPro",
                    "MakeFlow",
                    "BuildCore",
                    "ManuCore",
                    "ProduceFlow",
                    "FactoryCore",
                    "MakeTech",
                    "BuildFlow",
                    "ManuPro",
                    "ProduceCore",
                    "FactoryTech",
                    "MakeLink",
                    "BuildPro",
                    "ManuLink",
                    "ProduceLink",
                    "FactoryLink",
                    "MakeNet",
                    "BuildNet",
                ],
                "price_range": (499, 4999),
                "founded_range": (1980, 2005),
                "products": [
                    "Production Software",
                    "Quality Control System",
                    "Supply Chain Tool",
                    "Manufacturing Platform",
                    "Process Management",
                    "Factory Analytics",
                    "Equipment Monitor",
                    "Production Tracker",
                    "Quality Assurance Tool",
                    "Manufacturing Dashboard",
                ],
                "colors": ["#6b7280", "#4b5563", "#374151"],  # Gray tones
            },
            "food_beverage": {
                "companies": [
                    "FoodTech",
                    "BeveragePro",
                    "TasteFlow",
                    "FreshCore",
                    "FlavorTech",
                    "FoodFlow",
                    "BeverageFlow",
                    "TastePro",
                    "FreshFlow",
                    "FlavorCore",
                    "FoodCore",
                    "BeverageCore",
                    "TasteCore",
                    "FreshTech",
                    "FlavorFlow",
                    "FoodPro",
                    "BeverageTech",
                    "TasteLink",
                    "FreshPro",
                    "FlavorPro",
                    "FoodLink",
                    "BeverageLink",
                    "TasteTech",
                    "FreshLink",
                    "FlavorLink",
                ],
                "price_range": (9, 89),
                "founded_range": (1995, 2018),
                "products": [
                    "Recipe Management",
                    "Nutrition Tracker",
                    "Food Safety System",
                    "Menu Planning Tool",
                    "Inventory System",
                    "Quality Control",
                    "Restaurant POS",
                    "Food Analytics",
                    "Delivery Platform",
                    "Nutrition Software",
                ],
                "colors": ["#16a34a", "#15803d", "#166534"],  # Green tones
            },
            "professional_services": {
                "companies": [
                    "ServicePro",
                    "ConsultTech",
                    "ProFlow",
                    "ServiceCore",
                    "ConsultPro",
                    "ProTech",
                    "ServiceFlow",
                    "ConsultCore",
                    "ProCore",
                    "ServiceTech",
                    "ConsultFlow",
                    "ProLink",
                    "ServiceLink",
                    "ConsultLink",
                    "ProNet",
                    "ServiceNet",
                    "ConsultNet",
                    "ProBase",
                    "ServiceBase",
                    "ConsultBase",
                    "ProLogic",
                    "ServiceLogic",
                    "ConsultLogic",
                    "ProEdge",
                    "ServiceEdge",
                ],
                "price_range": (99, 999),
                "founded_range": (1985, 2012),
                "products": [
                    "Consulting Platform",
                    "Project Management",
                    "Client Portal",
                    "Service Management",
                    "Professional Tools",
                    "Business Analytics",
                    "Client Relationship System",
                    "Service Tracking",
                    "Professional Dashboard",
                    "Business Intelligence Tool",
                ],
                "colors": ["#0891b2", "#0e7490", "#155e75"],  # Cyan tones
            },
        }

    def set_seed(self, seed: int):
        """Set random seed for reproducible generation."""
        self.random.seed(seed)

    def get_base_css(self, domain: str) -> str:
        """Get professional CSS styling for the domain."""
        primary_color = self.random.choice(self.domains[domain]["colors"])

        return f"""
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, {primary_color} 0%, {primary_color}dd 100%);
                color: white;
                padding: 2rem;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }}
            
            .header .subtitle {{
                font-size: 1.1rem;
                opacity: 0.9;
                font-weight: 300;
            }}
            
            .content {{
                padding: 2rem;
            }}
            
            .card {{
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 1.5rem;
                margin: 1rem 0;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }}
            
            .card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            
            .highlight {{
                background: linear-gradient(120deg, {primary_color}22 0%, {primary_color}44 100%);
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                font-weight: 600;
                color: {primary_color};
                border-left: 3px solid {primary_color};
                padding-left: 0.8rem;
                display: inline-block;
                margin: 0.5rem 0;
            }}
            
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }}
            
            .info-item {{
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 1.5rem;
                text-align: center;
                transition: all 0.3s ease;
            }}
            
            .info-item:hover {{
                border-color: {primary_color};
                box-shadow: 0 0 0 2px {primary_color}22;
            }}
            
            .info-label {{
                font-size: 0.9rem;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 0.5rem;
            }}
            
            .info-value {{
                font-size: 1.5rem;
                font-weight: 700;
                color: {primary_color};
            }}
            
            .footer {{
                background: #f8f9fa;
                padding: 2rem;
                text-align: center;
                border-top: 1px solid #e9ecef;
                color: #6c757d;
            }}
            
            .btn {{
                display: inline-block;
                background: {primary_color};
                color: white;
                padding: 0.75rem 1.5rem;
                text-decoration: none;
                border-radius: 6px;
                font-weight: 600;
                transition: all 0.3s ease;
                border: none;
                cursor: pointer;
            }}
            
            .btn:hover {{
                background: {primary_color}dd;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px {primary_color}44;
            }}
            
            .navigation {{
                background: white;
                padding: 1rem 2rem;
                border-bottom: 1px solid #e9ecef;
            }}
            
            .nav-links {{
                display: flex;
                gap: 2rem;
                list-style: none;
            }}
            
            .nav-links a {{
                color: #6c757d;
                text-decoration: none;
                font-weight: 500;
                padding: 0.5rem 0;
                border-bottom: 2px solid transparent;
                transition: all 0.3s ease;
            }}
            
            .nav-links a:hover {{
                color: {primary_color};
                border-bottom-color: {primary_color};
            }}
        </style>
        """

    def generate_company_data(self, domain: str, seed: int) -> Dict[str, Any]:
        """Generate realistic company data for the domain."""
        self.set_seed(seed)
        domain_data = self.domains[domain]

        # Select company name and add variation
        company_name = self.random.choice(domain_data["companies"])
        if self.random.random() < 0.3:  # 30% chance of suffix
            company_name += f" {self.random.choice(['Inc', 'LLC', 'Corp', 'Ltd', 'Solutions'])}"

        # Generate founded year
        founded_year = self.random.randint(*domain_data["founded_range"])

        # Generate other company details
        employee_count = self.random.choice(["25-50", "50-100", "100-250", "250-500", "500-1000", "1000+"])

        location = self.random.choice(
            [
                "San Francisco, CA",
                "New York, NY",
                "Austin, TX",
                "Seattle, WA",
                "Boston, MA",
                "Denver, CO",
                "Atlanta, GA",
                "Chicago, IL",
                "Los Angeles, CA",
                "Miami, FL",
            ]
        )

        return {
            "name": company_name,
            "founded": founded_year,
            "domain": domain,
            "employees": employee_count,
            "location": location,
            "website": f"www.{company_name.lower().replace(' ', '').replace(',', '')}.com",
        }

    def generate_product_data(self, domain: str, seed: int) -> Dict[str, Any]:
        """Generate realistic product data for the domain."""
        self.set_seed(seed + 100)  # Offset seed for variety
        domain_data = self.domains[domain]

        product_name = self.random.choice(domain_data["products"])
        price = self.random.randint(*domain_data["price_range"])

        # Format price realistically
        if price < 100:
            price_str = f"${price}.99"
        else:
            price_str = f"${price:,}.00"

        return {
            "name": product_name,
            "price": price_str,
            "domain": domain,
            "description": f"Professional {product_name.lower()} designed for modern {domain} needs.",
            "features": self.random.randint(3, 8),
        }

    def generate_event_data(self, domain: str, seed: int) -> Dict[str, Any]:
        """Generate realistic event data for the domain."""
        self.set_seed(seed + 200)  # Offset seed for variety

        # Generate future date
        base_date = datetime.now()
        days_ahead = self.random.randint(30, 365)
        event_date = base_date + timedelta(days=days_ahead)

        event_types = {
            "technology": ["Tech Conference", "Developer Summit", "AI Workshop", "Startup Showcase"],
            "healthcare": ["Medical Conference", "Health Summit", "Wellness Workshop", "Care Innovation"],
            "finance": ["Financial Summit", "Investment Conference", "FinTech Expo", "Banking Workshop"],
            "education": ["Education Conference", "Learning Summit", "Teaching Workshop", "Academic Expo"],
            "retail": ["Retail Expo", "Commerce Summit", "Sales Conference", "Customer Workshop"],
            "manufacturing": ["Manufacturing Expo", "Industry Summit", "Production Conference", "Quality Workshop"],
            "food_beverage": ["Food Expo", "Culinary Summit", "Nutrition Conference", "Taste Workshop"],
            "professional_services": ["Professional Summit", "Service Expo", "Business Conference", "Client Workshop"],
        }

        event_name = self.random.choice(event_types.get(domain, ["Industry Conference"]))

        return {
            "name": event_name,
            "date": event_date.strftime("%B %d, %Y"),
            "time": f"{self.random.randint(9, 15)}:00 AM",
            "location": self.random.choice(["Convention Center", "Hotel Ballroom", "Conference Hall", "Event Center"]),
            "domain": domain,
        }

    def generate_contact_data(self, domain: str, seed: int) -> Dict[str, Any]:
        """Generate realistic contact data."""
        self.set_seed(seed + 300)

        # Generate phone number
        area_codes = ["555", "415", "212", "713", "206", "617", "303", "404", "312", "310"]
        area_code = self.random.choice(area_codes)
        phone = f"({area_code}) {self.random.randint(100, 999)}-{self.random.randint(1000, 9999)}"

        # Generate email
        email_prefixes = ["info", "contact", "support", "hello", "sales", "service"]
        email_prefix = self.random.choice(email_prefixes)
        email_domain = self.random.choice(["company.com", "business.com", "corp.com", "enterprise.com", "group.com"])
        email = f"{email_prefix}@{email_domain}"

        return {"phone": phone, "email": email, "domain": domain}

    def generate_level1_content(self, domain: str, content_type: str, seed: int) -> Dict[str, Any]:
        """Generate Level 1 content with enhanced HTML template."""
        self.set_seed(seed)

        if content_type == "company_info":
            company_data = self.generate_company_data(domain, seed)
            target_text = f"Founded in {company_data['founded']}"

            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{company_data["name"]} - Company Information</title>
    {self.get_base_css(domain)}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{company_data["name"]}</h1>
            <div class="subtitle">Leading {domain.title()} Solutions</div>
        </div>
        
        <nav class="navigation">
            <ul class="nav-links">
                <li><a href="#about">About</a></li>
                <li><a href="#services">Services</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </nav>
        
        <div class="content">
            <div class="card">
                <h2>About Our Company</h2>
                <p>Welcome to {company_data["name"]}, your trusted partner in {domain} solutions. We have built our reputation on delivering exceptional results and innovative approaches.</p>
                
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Established</div>
                        <div class="highlight">{target_text}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Location</div>
                        <div class="info-value">{company_data["location"]}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Team Size</div>
                        <div class="info-value">{company_data["employees"]} employees</div>
                    </div>
                </div>
                
                <p>Since our founding, we have been committed to excellence and have served thousands of satisfied customers across the {domain} industry.</p>
            </div>
        </div>
        
        <div class="footer">
            <p>Visit us at {company_data["website"]} | © 2024 {company_data["name"]}. All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""

        elif content_type == "product_info":
            product_data = self.generate_product_data(domain, seed)
            target_text = f"Price: {product_data['price']}"

            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{product_data["name"]} - Product Details</title>
    {self.get_base_css(domain)}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{product_data["name"]}</h1>
            <div class="subtitle">Professional {domain.title()} Solution</div>
        </div>
        
        <nav class="navigation">
            <ul class="nav-links">
                <li><a href="#features">Features</a></li>
                <li><a href="#pricing">Pricing</a></li>
                <li><a href="#support">Support</a></li>
            </ul>
        </nav>
        
        <div class="content">
            <div class="card">
                <h2>Product Overview</h2>
                <p>{product_data["description"]}</p>
                
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Features</div>
                        <div class="info-value">{product_data["features"]}+ powerful features</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Category</div>
                        <div class="info-value">{domain.title()}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Investment</div>
                        <div class="highlight">{target_text}</div>
                    </div>
                </div>
                
                <p>Experience the power of modern {domain} technology with our comprehensive solution. Designed for professionals who demand excellence and reliability.</p>
                
                <div style="text-align: center; margin-top: 2rem;">
                    <a href="#purchase" class="btn">Get Started Today</a>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Professional {domain} solutions you can trust | 24/7 support available</p>
        </div>
    </div>
</body>
</html>"""

        elif content_type == "event_info":
            event_data = self.generate_event_data(domain, seed)
            target_text = f"Date: {event_data['date']}"

            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{event_data["name"]} - Event Information</title>
    {self.get_base_css(domain)}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{event_data["name"]}</h1>
            <div class="subtitle">Premier {domain.title()} Industry Event</div>
        </div>
        
        <nav class="navigation">
            <ul class="nav-links">
                <li><a href="#agenda">Agenda</a></li>
                <li><a href="#speakers">Speakers</a></li>
                <li><a href="#register">Register</a></li>
            </ul>
        </nav>
        
        <div class="content">
            <div class="card">
                <h2>Join Us for an Exceptional Event</h2>
                <p>Don't miss this opportunity to connect with industry leaders, learn about the latest trends, and expand your professional network in the {domain} sector.</p>
                
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Event Date</div>
                        <div class="highlight">{target_text}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Time</div>
                        <div class="info-value">{event_data["time"]}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Venue</div>
                        <div class="info-value">{event_data["location"]}</div>
                    </div>
                </div>
                
                <p>This premier {domain} event brings together professionals, thought leaders, and innovators to share insights and drive the future of our industry.</p>
                
                <div style="text-align: center; margin-top: 2rem;">
                    <a href="#register" class="btn">Register Now</a>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Secure your spot today - Limited seats available | Early bird pricing ends soon</p>
        </div>
    </div>
</body>
</html>"""

        else:
            raise ValueError(f"Unknown content type: {content_type}")

        return {"html_content": html_content, "target_text": target_text, "template_type": content_type, "domain": domain}

    def generate_level2_content(self, domain: str, content_type: str, seed: int) -> Dict[str, Any]:
        """Generate Level 2 content with enhanced HTML template."""
        self.set_seed(seed)

        if content_type == "product_catalog":
            product_data = self.generate_product_data(domain, seed)
            target_texts = [f"Product: {product_data['name']}", f"Price: {product_data['price']}"]

            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Product Catalog - {domain.title()} Solutions</title>
    {self.get_base_css(domain)}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Featured Products</h1>
            <div class="subtitle">Professional {domain.title()} Solutions</div>
        </div>
        
        <nav class="navigation">
            <ul class="nav-links">
                <li><a href="#products">Products</a></li>
                <li><a href="#compare">Compare</a></li>
                <li><a href="#support">Support</a></li>
            </ul>
        </nav>
        
        <div class="content">
            <div class="card">
                <h2>Premium Product Offering</h2>
                <p>Discover our flagship {domain} solution designed for modern businesses and professionals.</p>
                
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Featured Solution</div>
                        <div class="highlight">{target_texts[0]}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Investment</div>
                        <div class="highlight">{target_texts[1]}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Features</div>
                        <div class="info-value">{product_data["features"]}+ capabilities</div>
                    </div>
                </div>
                
                <p>{product_data["description"]} Our solution combines cutting-edge technology with user-friendly design to deliver exceptional results for {domain} professionals.</p>
                
                <div style="text-align: center; margin-top: 2rem;">
                    <a href="#details" class="btn">View Full Details</a>
                    <a href="#purchase" class="btn" style="margin-left: 1rem;">Purchase Now</a>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Browse our complete catalog of {domain} solutions | 30-day money-back guarantee</p>
        </div>
    </div>
</body>
</html>"""

        elif content_type == "contact_info":
            contact_data = self.generate_contact_data(domain, seed)
            target_texts = [f"Phone: {contact_data['phone']}", f"Email: {contact_data['email']}"]

            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact Information - {domain.title()} Support</title>
    {self.get_base_css(domain)}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Get In Touch</h1>
            <div class="subtitle">Professional {domain.title()} Support</div>
        </div>
        
        <nav class="navigation">
            <ul class="nav-links">
                <li><a href="#contact">Contact</a></li>
                <li><a href="#support">Support</a></li>
                <li><a href="#hours">Hours</a></li>
            </ul>
        </nav>
        
        <div class="content">
            <div class="card">
                <h2>We're Here to Help</h2>
                <p>Our expert {domain} team is ready to assist you with any questions or support needs you may have.</p>
                
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Call Us</div>
                        <div class="highlight">{target_texts[0]}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Email Us</div>
                        <div class="highlight">{target_texts[1]}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Response Time</div>
                        <div class="info-value">Within 24 hours</div>
                    </div>
                </div>
                
                <p>Our dedicated support team has extensive experience in {domain} solutions and is committed to providing you with prompt, professional assistance.</p>
                
                <div style="text-align: center; margin-top: 2rem;">
                    <a href="tel:{contact_data["phone"]}" class="btn">Call Now</a>
                    <a href="mailto:{contact_data["email"]}" class="btn" style="margin-left: 1rem;">Send Email</a>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Available Monday-Friday 9AM-6PM PST | Emergency support available 24/7</p>
        </div>
    </div>
</body>
</html>"""

        else:
            raise ValueError(f"Unknown content type: {content_type}")

        return {"html_content": html_content, "target_texts": target_texts, "template_type": content_type, "domain": domain}

    def generate_level3_content(self, domain: str, seed: int) -> Dict[str, Any]:
        """Generate Level 3 content with enhanced HTML template and downloadable files."""
        self.set_seed(seed)

        # Generate document content based on domain (max 2 lines, 20 words)
        doc_types = {
            "technology": [
                "API Documentation\nComplete technical specifications.",
                "Product Roadmap\nQuarterly development milestones.",
                "System Requirements\nMinimum specifications needed.",
            ],
            "healthcare": [
                "Clinical Guidelines\nPatient care standards.",
                "Health Report\nQuarterly outcome analysis.",
                "Device Manual\nOperation instructions included.",
            ],
            "finance": [
                "Investment Analysis\nMarket performance review.",
                "Compliance Report\nRegulatory audit results.",
                "Risk Guidelines\nAssessment framework outlined.",
            ],
            "education": [
                "Curriculum Guidelines\nLearning objectives defined.",
                "Performance Report\nStudent achievement metrics.",
                "Teaching Manual\nInstructional resources provided.",
            ],
            "retail": [
                "Sales Report\nQuarterly revenue analysis.",
                "Inventory Guide\nStock control procedures.",
                "Service Manual\nCustomer support standards.",
            ],
            "manufacturing": [
                "Quality Manual\nProduction testing standards.",
                "Efficiency Report\nOperational performance analysis.",
                "Safety Manual\nWorkplace protection protocols.",
            ],
            "food_beverage": [
                "Safety Guidelines\nHACCP quality standards.",
                "Nutrition Report\nHealth content analysis.",
                "Recipe Manual\nCulinary preparation standards.",
            ],
            "professional_services": [
                "Consulting Guide\nProject delivery framework.",
                "Engagement Report\nClient satisfaction analysis.",
                "Development Guide\nSkill enhancement opportunities.",
            ],
        }

        # Select document content
        domain_docs = doc_types.get(domain, doc_types["professional_services"])
        file_content = self.random.choice(domain_docs)

        # Generate filename
        doc_number = self.random.randint(1000, 9999)
        domain_short = domain.replace("_", "")[:4].upper()
        download_filename = f"{domain_short}_{doc_number}_document.txt"
        file_description = f"{domain.replace('_', ' ').title()} Document #{doc_number}"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Repository - {domain.title()} Resources</title>
    {self.get_base_css(domain)}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Document Center</h1>
            <div class="subtitle">Professional {domain.title()} Resources</div>
        </div>
        
        <nav class="navigation">
            <ul class="nav-links">
                <li><a href="#documents">Documents</a></li>
                <li><a href="#resources">Resources</a></li>
                <li><a href="#support">Support</a></li>
            </ul>
        </nav>
        
        <div class="content">
            <div class="card">
                <h2>Professional Documentation</h2>
                <p>Access our comprehensive library of {domain.replace("_", " ")} documentation, guidelines, and professional resources.</p>
                
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Available Document</div>
                        <div class="info-value">{file_description}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Category</div>
                        <div class="info-value">{domain.title()}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Format</div>
                        <div class="info-value">Text Document</div>
                    </div>
                </div>
                
                <div style="text-align: center; margin: 2rem 0;">
                    <a href="http://localhost:8080/files/{download_filename}" download class="btn">
                        Download {file_description}
                    </a>
                </div>
                
                <p>This document contains essential information for {domain.replace("_", " ")} professionals and provides detailed guidance on current industry practices and standards.</p>
                
                <div class="card" style="background: #f8f9fa; margin-top: 2rem;">
                    <h3>Additional Resources</h3>
                    <ul style="margin: 1rem 0; padding-left: 2rem;">
                        <li>Professional guidelines and best practices</li>
                        <li>Industry standards and compliance information</li>
                        <li>Technical documentation and specifications</li>
                        <li>Training materials and educational resources</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>All documents are provided in accessible formats | Updated regularly with latest industry standards</p>
        </div>
    </div>
</body>
</html>"""

        return {
            "html_content": html_content,
            "file_content": file_content,
            "download_filename": download_filename,
            "file_description": file_description,
            "template_type": "document_repository",
            "domain": domain,
        }
