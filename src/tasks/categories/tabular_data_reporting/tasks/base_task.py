"""
Base generator class for data analysis tasks with enhanced variability.
"""

import random
from abc import abstractmethod
from typing import Dict, Any, List, Optional
from ...base import BaseTask


class TabularDataReportingBaseTask(BaseTask):
    """Base class for tabular data reporting task implementations with enhanced variability."""

    # Domain-specific data scenarios for enhanced variability
    DOMAIN_DATA_SCENARIOS = {
        "business_analytics": {
            "data_types": ["sales_records", "customer_data", "revenue_figures", "performance_metrics", "quarterly_reports"],
            "operations": ["analyze", "summarize", "calculate", "aggregate", "evaluate"],
            "contexts": ["monthly sales", "customer analytics", "profit margins", "growth metrics", "business KPIs"],
            "file_prefixes": ["sales", "revenue", "customers", "metrics", "reports"],
            "value_ranges": {"low": (100, 500), "medium": (500, 2000), "high": (2000, 10000)},
        },
        "scientific_research": {
            "data_types": ["measurements", "observations", "test_results", "sample_data", "experiment_records"],
            "operations": ["measure", "observe", "calculate", "analyze", "validate"],
            "contexts": ["lab experiments", "field studies", "research data", "clinical trials", "sensor readings"],
            "file_prefixes": ["experiment", "lab", "research", "study", "measurements"],
            "value_ranges": {"low": (0.1, 10), "medium": (10, 100), "high": (100, 1000)},
        },
        "education_management": {
            "data_types": ["student_scores", "grade_records", "attendance_data", "course_evaluations", "academic_metrics"],
            "operations": ["grade", "evaluate", "calculate", "track", "assess"],
            "contexts": ["student performance", "course grades", "class averages", "academic progress", "test scores"],
            "file_prefixes": ["grades", "students", "scores", "class", "academic"],
            "value_ranges": {"low": (0, 50), "medium": (50, 85), "high": (85, 100)},
        },
        "inventory_logistics": {
            "data_types": ["stock_levels", "inventory_counts", "shipment_records", "warehouse_data", "supply_metrics"],
            "operations": ["count", "track", "manage", "calculate", "monitor"],
            "contexts": ["warehouse inventory", "stock management", "supply levels", "shipment tracking", "logistics data"],
            "file_prefixes": ["inventory", "stock", "warehouse", "shipment", "supplies"],
            "value_ranges": {"low": (10, 100), "medium": (100, 1000), "high": (1000, 5000)},
        },
        "financial_planning": {
            "data_types": ["budget_items", "expense_records", "financial_data", "cost_analysis", "investment_figures"],
            "operations": ["budget", "calculate", "plan", "analyze", "forecast"],
            "contexts": ["budget planning", "expense tracking", "financial analysis", "cost management", "investment planning"],
            "file_prefixes": ["budget", "expenses", "financial", "costs", "investments"],
            "value_ranges": {"low": (50, 500), "medium": (500, 5000), "high": (5000, 50000)},
        },
    }

    # Data format templates for structural diversity
    DATA_FORMAT_TEMPLATES = {
        "simple_values": {"description": "Plain numeric values, one per line", "complexity": "minimal"},
        "labeled_entries": {"description": "Values with descriptive labels", "complexity": "moderate"},
        "structured_records": {"description": "Multi-field data records", "complexity": "intermediate"},
        "formatted_reports": {"description": "Report-style formatted data", "complexity": "advanced"},
    }

    def __init__(self, task_type: str, level: int):
        super().__init__()
        self.task_type = task_type
        self.level = level
        self.random = random.Random()
        self.current_domain = None
        self.current_template = None

    def set_seed(self, seed: int):
        """Set random seed for reproducible generation."""
        self.random.seed(seed)

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

    def generate_directory_name(self, prefix: str = "data") -> str:
        """Generate random directory name."""
        suffix = self.random.choice(["_folder", "_set", "_collection", "_files"])
        number = self.random.randint(1, 50)
        return f"{prefix}{suffix}_{number:02d}"

    def generate_filename(self, prefix: str = "data", extension: str = ".txt") -> str:
        """Generate random filename."""
        number = self.random.randint(1, 999)
        return f"{prefix}_{number:03d}{extension}"

    def generate_cell_reference(self, columns: List[str] = None, max_row: int = 10) -> str:
        """Generate random cell reference."""
        if columns is None:
            columns = ["A", "B", "C", "D", "E", "F"]
        column = self.random.choice(columns)
        row = self.random.randint(1, max_row)
        return f"{column}{row}"

    def generate_directory_path(self, base_path: str = "/home/user") -> str:
        """Generate random directory path."""
        directory_name = self.generate_directory_name()
        return f"{base_path}/{directory_name}"

    def generate_enhanced_task_structure(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate enhanced task structure with domain awareness."""
        if seed is not None:
            self.set_seed(seed)

        # Select domain and format template for this task instance
        domain = self.random.choice(list(self.DOMAIN_DATA_SCENARIOS.keys()))
        format_template = self.random.choice(list(self.DATA_FORMAT_TEMPLATES.keys()))

        self.current_domain = domain
        self.current_template = format_template

        domain_info = self.DOMAIN_DATA_SCENARIOS[domain]

        # Generate domain-specific context
        domain_context = {
            "scenario": domain,
            "data_type": self.random.choice(domain_info["data_types"]),
            "operation": self.random.choice(domain_info["operations"]),
            "context_description": self.random.choice(domain_info["contexts"]),
            "file_prefix": self.random.choice(domain_info["file_prefixes"]),
            "value_range": domain_info["value_ranges"],
        }

        return {
            "task_type": self.task_type,
            "level": self.level,
            "seed": seed or self.random.randint(1, 10000),
            "domain": domain,
            "format_template": format_template,
            "domain_context": domain_context,
            "evaluation_mode": "compare_table",
            "evaluation_method": "compare_table",
            "file_name": self.generate_domain_filename(domain_context),
            "example_id": f"L{self.level}_{self.task_type}_{domain}_{self.random.randint(1, 1000)}",
        }

    def generate_basic_task_structure(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate basic task structure (backwards compatibility)."""
        return self.generate_enhanced_task_structure(seed)

    def format_number_content(self, number: int) -> str:
        """Format number content with random patterns."""
        patterns = [
            str(number),
            f"Value: {number}",
            f"Result = {number}",
            f"Data: {number}",
            f"Number: {number}",
        ]
        return self.random.choice(patterns)

    def format_numbers_list(self, numbers: List[int]) -> str:
        """Format list of numbers with random patterns."""
        patterns = [
            "\n".join(str(n) for n in numbers),
            "\n".join(f"{n}" for n in numbers),
            "\n".join(f"Item: {n}" for n in numbers),
            "\n".join(f"Value {i + 1}: {n}" for i, n in enumerate(numbers)),
        ]
        return self.random.choice(patterns)

    def generate_spreadsheet_data(self, rows: int, cols: int = 2) -> Dict[str, List[int]]:
        """Generate data for spreadsheet with specified rows and columns."""
        data = {}
        column_names = ["A", "B", "C", "D", "E", "F"][:cols]

        for i, col_name in enumerate(column_names):
            data[f"column_{col_name.lower()}"] = [self.generate_random_number(1, 20) for _ in range(rows)]

        return data

    def generate_domain_filename(self, domain_context: Dict[str, Any]) -> str:
        """Generate domain-appropriate filename."""
        prefix = domain_context["file_prefix"]
        suffix_options = ["data", "records", "list", "info", "report"]
        suffix = self.random.choice(suffix_options)
        number = self.random.randint(100, 999)
        extension = self.random.choice([".txt", ".csv", ".dat"])
        return f"{prefix}_{suffix}_{number}{extension}"

    def generate_contextual_numbers(self, count: int, domain_context: Dict[str, Any], complexity: str = "medium") -> List[int]:
        """Generate numbers appropriate for the domain context."""
        value_ranges = domain_context["value_range"]
        range_info = value_ranges.get(complexity, value_ranges["medium"])

        # Handle scientific research with decimals
        if domain_context["scenario"] == "scientific_research" and complexity == "low":
            return [round(self.random.uniform(range_info[0], range_info[1]), 1) for _ in range(count)]

        return [self.random.randint(int(range_info[0]), int(range_info[1])) for _ in range(count)]

    def format_data_content(self, data: List[float], domain_context: Dict[str, Any], format_template: str) -> str:
        """Format data content based on domain and template."""
        data_type = domain_context["data_type"]
        context_desc = domain_context["context_description"]

        # For Level 1 tasks (single value), use simpler formats to avoid ambiguity
        if len(data) == 1:
            target_value = data[0]

            if format_template == "simple_values":
                return str(target_value)

            elif format_template == "labeled_entries":
                # Use clear, unambiguous labeling
                data_label = data_type.replace("_", " ").title()
                return f"Target {data_label}: {target_value}"

            elif format_template == "structured_records":
                return f"Data Record | {data_type}: {target_value} | Source: Verified"

            else:  # formatted_reports - simplified for single values
                header = f"=== {context_desc.title()} Data ==="
                return f"{header}\n\nValue: {target_value}\n\n[End of Report]"

        # For multiple values (Level 2+), use the original formatting
        else:
            if format_template == "simple_values":
                return "\n".join(str(value) for value in data)

            elif format_template == "labeled_entries":
                labels = [f"{data_type.replace('_', ' ').title()} {i + 1}" for i in range(len(data))]
                return "\n".join(f"{label}: {value}" for label, value in zip(labels, data))

            elif format_template == "structured_records":
                return "\n".join(f"Record {i + 1} | {data_type}: {value} | Status: Active" for i, value in enumerate(data))

            else:  # formatted_reports
                header = f"=== {context_desc.title()} Report ==="
                lines = [header, ""]
                lines.extend(
                    f"Entry {i + 1:02d}: {value:,}" if isinstance(value, int) else f"Entry {i + 1:02d}: {value}" for i, value in enumerate(data)
                )
                lines.append(f"\nTotal entries: {len(data)}")
                lines.append("Note: Use only the values after 'Entry XX:' for calculations")
                return "\n".join(lines)

    def generate_contextual_instruction(self, domain_context: Dict[str, Any], task_specifics: Dict[str, Any]) -> str:
        """Generate instruction text that reflects the domain context."""
        data_type = domain_context["data_type"]
        context_desc = domain_context["context_description"]

        # Use domain-appropriate language
        action_verbs = {
            "business_analytics": "analyze and process",
            "scientific_research": "examine and calculate",
            "education_management": "evaluate and compute",
            "inventory_logistics": "count and summarize",
            "financial_planning": "review and calculate",
        }

        domain = domain_context["scenario"]
        action = action_verbs.get(domain, "process and calculate")

        return f"Please {action} the {data_type} from the {context_desc} data file."

    def generate_contextual_directory_path(self, domain_context: Dict[str, Any], base_path: str = "/home/user") -> str:
        """Generate domain-appropriate directory path."""
        scenario = domain_context["scenario"]

        directory_names = {
            "business_analytics": ["business_data", "analytics_files", "reports", "sales_data"],
            "scientific_research": ["research_data", "lab_files", "experiments", "study_data"],
            "education_management": ["student_data", "grades", "academic_files", "class_records"],
            "inventory_logistics": ["inventory_data", "warehouse_files", "stock_records", "logistics"],
            "financial_planning": ["financial_data", "budget_files", "expense_reports", "planning"],
        }

        dir_options = directory_names.get(scenario, ["data_files"])
        dir_name = self.random.choice(dir_options)
        number = self.random.randint(1, 20)

        return f"{base_path}/{dir_name}_{number:02d}"
