"""
Level 1 task implementations for dynamic task creation.
Basic Python Syntax Fix & Execution tasks.
"""

from typing import Dict, Any, Optional
import ast
from .base_task import BaseTaskGenerator


class BasicPythonSyntaxFixGenerator(BaseTaskGenerator):
    """Generate dynamic basic Python syntax fix tasks."""

    def __init__(self):
        super().__init__("basic_python_syntax_fix", 1)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic Python syntax fix task with enhanced variability."""
        task_data = self.generate_basic_task_structure(seed)

        # Use domain context from enhanced base task
        domain_context = task_data.get("domain_context", {})

        # Generate domain-aware elements
        script_name = task_data.get("file_name", f"processor_{self.generate_random_number(100, 999)}.py")
        variable_name = self.generate_variable_name(domain_context.get("main_variable", "items"))
        count_value = self.generate_random_number(10, 100)

        # Use domain-appropriate process types
        domain = task_data.get("domain", "data_processing")
        if domain == "data_processing":
            process_types = ["files", "records", "entries", "documents", "datasets", "batches"]
        elif domain == "web_services":
            process_types = ["requests", "users", "sessions", "endpoints", "responses", "clients"]
        elif domain == "file_management":
            process_types = ["files", "documents", "folders", "backups", "archives", "paths"]
        elif domain == "calculations":
            process_types = ["values", "numbers", "measurements", "scores", "calculations", "results"]
        else:  # system_admin
            process_types = ["configs", "settings", "parameters", "properties", "deployments", "logs"]

        process_type = self.random.choice(process_types)

        # Error type selection is done dynamically based on content

        # Generate content based on template variant and error type
        operation_name = domain_context.get("operation_name", "process")
        template_variant = task_data.get("template_variant", "simple_script")

        # Generate base content structure based on template variant
        if template_variant == "function_based":
            # Function-based structure
            base_structure = self._generate_function_based_content(variable_name, count_value, process_type, operation_name)
        elif template_variant == "data_processing":
            # Data processing with lists/loops
            base_structure = self._generate_data_processing_content(variable_name, count_value, process_type, operation_name)
        elif template_variant == "interactive":
            # Interactive input/output
            base_structure = self._generate_interactive_content(variable_name, count_value, process_type, operation_name)
        else:
            # Simple script (default)
            base_structure = self._generate_simple_script_content(variable_name, count_value, process_type, operation_name)

        # Get correct content and expected output from base structure
        correct_content = base_structure["correct"]
        expected_output = base_structure["expected_output"]

        # Apply exactly one robust error with validation
        broken_content = self._apply_error_with_validation(correct_content)

        task_data.update(
            {
                "file_name": script_name,
                "broken_file_content": broken_content,
                "correct_file_content": correct_content,
                "instructions": f"Ensure that the Python script located at /home/user/coding_tasks/{script_name} runs successfully and produces the expected output. This requires locating and resolving any syntax errors in the script before execution.",
                "expected_output": expected_output,
                "evaluation_data": {
                    "correct_file": correct_content,
                    "ignore_whitespace": True,
                },
            }
        )

        return task_data

    def _apply_error_with_validation(self, correct_content: str) -> str:
        """Apply exactly one error type with equal probability.
        Choose error type first, then validate it can be applied to this content.
        Never inject new lines; only mutate existing content.
        """
        # All possible error types with equal probability
        all_error_types = ["missing_quote", "wrong_capitalization", "missing_colon", "wrong_indentation", "mismatched_brackets"]

        # Shuffle for randomness
        self.random.shuffle(all_error_types)

        # Try each error type until one succeeds
        for error_type in all_error_types:
            if self._can_apply_error_type(correct_content, error_type):
                candidate = self._apply_error_to_content(correct_content, error_type, None, None)
                if candidate and candidate != correct_content:
                    # For syntax errors, validate they actually break parsing
                    if error_type == "wrong_capitalization":
                        return candidate  # Capitalization errors are always valid
                    try:
                        ast.parse(candidate)
                        # If it still parses, the mutation wasn't effective enough
                        continue
                    except SyntaxError:
                        return candidate  # Successfully created syntax error

        # Fallback: if no error type worked, return original content
        return correct_content

    def _can_apply_error_type(self, content: str, error_type: str) -> bool:
        """Check if a specific error type can be applied to this content."""
        import re

        if error_type == "missing_quote":
            # Check for any string literal
            return bool(re.search(r'("[^"]*"|\'[^\']*\')', content))

        elif error_type == "wrong_capitalization":
            # Check for any builtin function call
            builtins = [
                "print",
                "len",
                "range",
                "list",
                "dict",
                "str",
                "int",
                "float",
                "bool",
                "sum",
                "max",
                "min",
                "abs",
                "round",
                "sorted",
                "reversed",
                "enumerate",
                "zip",
                "map",
                "filter",
                "any",
                "all",
                "type",
                "isinstance",
                "hasattr",
                "getattr",
                "setattr",
                "open",
                "input",
            ]
            return any(re.search(rf"\b{b}\s*\(", content) for b in builtins)

        elif error_type == "missing_colon":
            # Check for any block header (def, if, for, while, else, elif)
            return bool(re.search(r"(?m)^\s*(def|if|for|while|else|elif)\b[^\n]*:", content))

        elif error_type == "wrong_indentation":
            # Check for any indented non-empty line
            return any(line.startswith("    ") and line.strip() for line in content.splitlines())

        elif error_type == "mismatched_brackets":
            # Check for any function call
            return bool(re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\)", content))

        return False

    def _get_applicable_error_types(self, content: str) -> list:
        """Determine applicable error types for content without injecting text."""
        import re

        applicable: list = []

        # missing_quote: any string literal present
        if re.search(r'("[^"]*"|\'[^\']*\')', content):
            applicable.append("missing_quote")

        # wrong_capitalization: any builtin call present
        builtins = [
            "print",
            "len",
            "range",
            "list",
            "dict",
            "str",
            "int",
            "float",
            "bool",
            "sum",
            "max",
            "min",
            "abs",
            "round",
            "sorted",
            "reversed",
            "enumerate",
            "zip",
            "map",
            "filter",
            "any",
            "all",
            "type",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
            "open",
            "input",
        ]
        if any(re.search(rf"\b{b}\s*\(", content) for b in builtins):
            applicable.append("wrong_capitalization")

        # missing_colon: any block header
        if re.search(r"(?m)^\s*(def|if|for|while|else|elif)\b[^\n]*:", content):
            applicable.append("missing_colon")

        # wrong_indentation: any indented non-empty line
        if any(line.startswith("    ") and line.strip() for line in content.splitlines()):
            applicable.append("wrong_indentation")

        # mismatched_brackets: any function call
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\)", content):
            applicable.append("mismatched_brackets")

        return applicable

    def _generate_simple_script_content(self, variable_name: str, count_value: int, process_type: str, operation_name: str) -> Dict[str, str]:
        """Generate simple script structure with some indentation and control structures."""
        # Generate script that supports all error types
        correct = f"""{variable_name}_count = {count_value}
if {variable_name}_count > 0:
    print(f"Processing {process_type}...")
    print(f"Complete: {{{variable_name}_count}} {process_type} processed")"""
        return {"correct": correct, "expected_output": f"Processing {process_type}...\nComplete: {count_value} {process_type} processed"}

    def _generate_function_based_content(self, variable_name: str, count_value: int, process_type: str, operation_name: str) -> Dict[str, str]:
        """Generate function-based script structure."""
        correct = f"""def {operation_name}_{process_type}():
    {variable_name}_count = {count_value}
    print(f"Processing {process_type}...")
    print(f"Complete: {{{variable_name}_count}} {process_type} processed")

{operation_name}_{process_type}()"""
        return {"correct": correct, "expected_output": f"Processing {process_type}...\nComplete: {count_value} {process_type} processed"}

    def _generate_data_processing_content(self, variable_name: str, count_value: int, process_type: str, operation_name: str) -> Dict[str, str]:
        """Generate data processing with list/loop structure."""
        correct = f"""{variable_name}_list = list(range({count_value}))
{variable_name}_count = len({variable_name}_list)
print(f"Processing {process_type}...")
for item in {variable_name}_list:
    pass
print(f"Complete: {{{variable_name}_count}} {process_type} processed")"""
        return {"correct": correct, "expected_output": f"Processing {process_type}...\nComplete: {count_value} {process_type} processed"}

    def _generate_interactive_content(self, variable_name: str, count_value: int, process_type: str, operation_name: str) -> Dict[str, str]:
        """Generate interactive input/output structure."""
        correct = f"""print("Starting {operation_name} for {process_type}")
{variable_name}_count = {count_value}
if {variable_name}_count > 0:
    print(f"Processing {process_type}...")
    print(f"Complete: {{{variable_name}_count}} {process_type} processed")
else:
    print("No {process_type} to process")"""
        return {
            "correct": correct,
            "expected_output": f"Starting {operation_name} for {process_type}\nProcessing {process_type}...\nComplete: {count_value} {process_type} processed",
        }

    def _apply_error_to_content(self, correct_content: str, error_type: str, variable_name: str, count_value: int) -> str:
        """Apply specific error type to correct content dynamically."""
        if error_type == "missing_quote":
            return self._apply_missing_quote_error(correct_content)
        elif error_type == "wrong_capitalization":
            return self._apply_wrong_capitalization_error(correct_content)
        elif error_type == "missing_colon":
            return self._apply_missing_colon_error(correct_content)
        elif error_type == "wrong_indentation":
            return self._apply_wrong_indentation_error(correct_content)
        elif error_type == "mismatched_brackets":
            return self._apply_mismatched_brackets_error(correct_content)

        # This should not happen with the current error types
        return correct_content

    def _apply_missing_quote_error(self, content: str) -> str:
        """Dynamically find and break a quote in string literals."""
        import re

        # Find all string literals with quotes
        patterns = [
            r'"[^"]*"',  # Double quoted strings
            r"'[^']*'",  # Single quoted strings
        ]

        all_matches = []
        for pattern in patterns:
            all_matches.extend(list(re.finditer(pattern, content)))

        if all_matches:
            # Randomly select a string to break
            match = self.random.choice(all_matches)
            original_string = match.group()
            # Remove the closing quote
            broken_string = original_string[:-1]
            return content[: match.start()] + broken_string + content[match.end() :]
        # No injection fallback
        return content

    def _apply_wrong_capitalization_error(self, content: str) -> str:
        """Dynamically find and capitalize a Python built-in function call."""
        import re

        # Python built-ins that should be lowercase
        python_builtins = [
            "print",
            "len",
            "range",
            "list",
            "dict",
            "str",
            "int",
            "float",
            "bool",
            "sum",
            "max",
            "min",
            "abs",
            "round",
            "sorted",
            "reversed",
            "enumerate",
            "zip",
            "map",
            "filter",
            "any",
            "all",
            "type",
            "isinstance",
            "hasattr",
            "getattr",
            "setattr",
            "open",
            "input",
        ]

        # Find function calls that are Python built-ins
        builtin_matches = []
        for builtin in python_builtins:
            pattern = rf"\b({builtin})\s*\("
            matches = list(re.finditer(pattern, content))
            builtin_matches.extend(matches)

        if builtin_matches:
            # Randomly select a built-in function to capitalize
            match = self.random.choice(builtin_matches)
            function_name = match.group(1)
            capitalized_name = function_name.upper()
            return content[: match.start(1)] + capitalized_name + content[match.end(1) :]
        # No injection fallback
        return content

    def _apply_missing_colon_error(self, content: str) -> str:
        """Dynamically find and remove colon from control structures or create a syntax error."""
        import re

        # Find patterns that should end with colon
        patterns = [
            r"(def\s+\w+\s*\([^)]*\))\s*:",  # Function definitions
            r"(if\s+[^:]+)\s*:",  # If statements
            r"(for\s+[^:]+)\s*:",  # For loops
            r"(while\s+[^:]+)\s*:",  # While loops
            r"(else)\s*:",  # Else statements
            r"(elif\s+[^:]+)\s*:",  # Elif statements
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                # Remove the colon
                before_colon = match.group(1)
                return content[: match.start()] + before_colon + content[match.end() :]
        # No injection fallback
        return content

    def _apply_wrong_indentation_error(self, content: str) -> str:
        """Dynamically find and break indentation."""
        lines = content.split("\n")

        # Find all indented lines that are not empty
        indented_lines = []
        for i, line in enumerate(lines):
            if line.startswith("    ") and line.strip():
                indented_lines.append(i)

        if indented_lines:
            # Randomly select an indented line to break
            selected_line_idx = self.random.choice(indented_lines)
            # Remove all leading spaces
            lines[selected_line_idx] = lines[selected_line_idx].lstrip()

        return "\n".join(lines)

    def _apply_mismatched_brackets_error(self, content: str) -> str:
        """Dynamically find and break bracket/parentheses matching."""
        import re

        # Find function calls with parentheses
        pattern = r"([a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*)\)"
        matches = list(re.finditer(pattern, content))

        if matches:
            # Randomly select a function call and remove its closing parenthesis
            match = self.random.choice(matches)
            before_closing_paren = match.group(1)
            return content[: match.start()] + before_closing_paren + content[match.end() :]
        # No injection fallback
        return content


# Level 1 tasks are registered in __init__.py to avoid circular imports
