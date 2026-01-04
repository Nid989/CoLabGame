"""
Base generator class for dynamic task generation.
"""

import random
import string
from abc import abstractmethod
from typing import Dict, Any, Optional
from ...base import BaseTask


class BaseTaskGenerator(BaseTask):
    """Base class for dynamic task implementations."""

    # Domain-specific variable pools for enhanced variability
    DOMAIN_VARIABLE_POOLS = {
        "tabular_data_reporting": {
            "variables": ["records", "entries", "dataset", "batch", "rows", "items", "documents"],
            "operations": ["process", "filter", "sort", "validate", "clean", "transform", "analyze"],
            "outputs": ["results", "summary", "report", "output", "processed_data", "final_data"],
            "prefixes": ["data", "file", "record", "entry", "batch"],
        },
        "web_services": {
            "variables": ["requests", "responses", "users", "sessions", "endpoints", "clients"],
            "operations": ["handle", "route", "authenticate", "validate", "cache", "serve"],
            "outputs": ["response", "status", "result", "data", "payload", "json_data"],
            "prefixes": ["web", "api", "service", "request", "user"],
        },
        "file_management": {
            "variables": ["files", "documents", "folders", "paths", "archives", "backups"],
            "operations": ["backup", "organize", "compress", "sync", "restore", "copy"],
            "outputs": ["backup_log", "file_list", "status_report", "summary", "manifest"],
            "prefixes": ["file", "backup", "folder", "path", "archive"],
        },
        "calculations": {
            "variables": ["values", "numbers", "measurements", "scores", "totals", "amounts"],
            "operations": ["calculate", "compute", "evaluate", "analyze", "sum", "average"],
            "outputs": ["result", "total", "average", "final_score", "calculation", "answer"],
            "prefixes": ["calc", "math", "num", "value", "score"],
        },
        "system_admin": {
            "variables": ["configs", "settings", "parameters", "options", "properties", "values"],
            "operations": ["configure", "setup", "initialize", "update", "deploy", "manage"],
            "outputs": ["config_file", "system_log", "status", "deployment_report", "log_entry"],
            "prefixes": ["config", "system", "admin", "setup", "deploy"],
        },
    }

    # Template variants for structural diversity
    TEMPLATE_VARIANTS = {
        "simple_script": {"description": "Linear execution with basic variable operations", "complexity": "minimal"},
        "function_based": {"description": "Modular approach with function definitions", "complexity": "moderate"},
        "tabular_data_reporting": {"description": "List/loop processing patterns", "complexity": "intermediate"},
        "interactive": {"description": "Input/output interaction flow", "complexity": "moderate"},
    }

    def __init__(self, task_type: str, level: int):
        super().__init__()
        self.task_type = task_type
        self.level = level
        self.random = random.Random()
        self.current_domain = None
        self.current_template_variant = None

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

    def generate_random_string(self, length: int = 8) -> str:
        """Generate random string for variable names."""
        return "".join(self.random.choices(string.ascii_lowercase, k=length))

    def generate_random_number(self, min_val: int = 1, max_val: int = 100) -> int:
        """Generate random number."""
        return self.random.randint(min_val, max_val)

    def generate_variable_name(self, prefix: str = "var", domain: str = None) -> str:
        """Generate domain-aware variable name."""
        # Use domain-specific naming if domain is provided or already set
        if domain or self.current_domain:
            domain = domain or self.current_domain
            if domain in self.DOMAIN_VARIABLE_POOLS:
                pool = self.DOMAIN_VARIABLE_POOLS[domain]
                # Generate contextual variable names
                patterns = [
                    lambda: self.random.choice(pool["variables"]),
                    lambda: f"{self.random.choice(pool['prefixes'])}_{self.random.choice(pool['variables'])}",
                    lambda: f"{prefix}_{self.random.choice(pool['variables'])}",
                    lambda: f"{self.random.choice(pool['variables'])}_{self.random.randint(1, 99)}",
                ]
                return self.random.choice(patterns)()

        # Fallback to original method for backward compatibility
        suffix = self.random.choice(["_data", "_value", "_result", "_item", "_info", "_code"])
        number = self.random.randint(1, 9)
        return f"{prefix}{suffix}_{number}"

    def generate_filename(self, prefix: str = "task", extension: str = ".py", domain: str = None) -> str:
        """Generate domain-aware filename."""
        # Use domain-specific naming if domain is provided or already set
        if domain or self.current_domain:
            domain = domain or self.current_domain
            if domain in self.DOMAIN_VARIABLE_POOLS:
                pool = self.DOMAIN_VARIABLE_POOLS[domain]
                # Use domain-specific operations as filename suffixes
                suffix = f"_{self.random.choice(pool['operations'])}"
                number = self.random.randint(100, 999)
                domain_prefix = self.random.choice(pool["prefixes"])
                return f"{domain_prefix}{suffix}_{number}{extension}"

        # Fallback to original method for backward compatibility
        suffix = self.random.choice(["_process", "_handle", "_manage", "_display", "_calc"])
        number = self.random.randint(1, 20)
        return f"{prefix}{suffix}_{number:02d}{extension}"

    def generate_string_literal(self) -> str:
        """Generate random string literal content."""
        options = [
            "Hello World",
            "Processing Data",
            "Task Complete",
            "Starting Program",
            "Loading File",
            "Saving Results",
            "Update Value",
            "Main Input",
            "Calculate File",
            "Next Input",
            "Result Found",
            "Operation Done",
        ]
        return self.random.choice(options)

    def generate_print_message(self) -> str:
        """Generate random print message."""
        templates = [
            "Result: {value}",
            "Output: {value}",
            "Value: {value}",
            "Final: {value}",
            "Answer: {value}",
            "Data: {value}",
        ]
        return self.random.choice(templates)

    def generate_function_name(self) -> str:
        """Generate random function name."""
        prefixes = ["process", "handle", "format", "calculate", "convert", "validate"]
        suffixes = ["data", "value", "result", "info", "item", "content"]
        return f"{self.random.choice(prefixes)}_{self.random.choice(suffixes)}"

    def generate_expected_output(self, value: Any) -> str:
        """Generate expected output string."""
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return str(value)

    def select_domain(self, preferred_domain: str = None) -> str:
        """Select a domain for the task, either specified or random."""
        if preferred_domain and preferred_domain in self.DOMAIN_VARIABLE_POOLS:
            return preferred_domain
        return self.random.choice(list(self.DOMAIN_VARIABLE_POOLS.keys()))

    def select_template_variant(self, preferred_variant: str = None) -> str:
        """Select a template variant for the task."""
        if preferred_variant and preferred_variant in self.TEMPLATE_VARIANTS:
            return preferred_variant
        return self.random.choice(list(self.TEMPLATE_VARIANTS.keys()))

    def get_domain_context(self, domain: str) -> Dict[str, Any]:
        """Get contextual information for a domain."""
        if domain in self.DOMAIN_VARIABLE_POOLS:
            pool = self.DOMAIN_VARIABLE_POOLS[domain]
            return {
                "domain": domain,
                "main_variable": self.random.choice(pool["variables"]),
                "operation_name": self.random.choice(pool["operations"]),
                "output_name": self.random.choice(pool["outputs"]),
                "prefix": self.random.choice(pool["prefixes"]),
            }
        return {"domain": "generic", "main_variable": "data", "operation_name": "process", "output_name": "result", "prefix": "task"}

    def generate_basic_task_structure(self, seed: Optional[int] = None, domain: str = None, template_variant: str = None) -> Dict[str, Any]:
        """Generate enhanced task structure with domain and template variant support."""
        if seed is not None:
            self.set_seed(seed)

        # Select domain and template variant
        self.current_domain = self.select_domain(domain)
        self.current_template_variant = self.select_template_variant(template_variant)

        # Get domain context
        domain_context = self.get_domain_context(self.current_domain)

        return {
            "task_type": self.task_type,
            "level": self.level,
            "seed": seed or self.random.randint(1, 10000),
            "file_name": self.generate_filename(domain=self.current_domain),
            "evaluation_method": "compare_text_file" if self.level == 1 else "compare_answer",
            "example_id": f"L{self.level}_{self.task_type}_{self.random.randint(1, 1000)}",
            "domain": self.current_domain,
            "template_variant": self.current_template_variant,
            "domain_context": domain_context,
        }
