"""
Level 1 task implementations for data analysis: Simple Data Transfer.
"""

from typing import Dict, Any, Optional
from .base_task import TabularDataReportingBaseTask


class SimpleDataTransferGenerator(TabularDataReportingBaseTask):
    """Generate dynamic simple_data_transfer tasks."""

    def __init__(self):
        super().__init__("simple_data_transfer", 1)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic simple data transfer task with enhanced variability."""
        task_data = self.generate_enhanced_task_structure(seed)

        domain_context = task_data["domain_context"]
        format_template = task_data["format_template"]

        # Generate contextual target number based on domain
        complexity = "low"  # Level 1 uses simpler values
        target_numbers = self.generate_contextual_numbers(1, domain_context, complexity)
        target_number = target_numbers[0]

        # Format content based on domain and template
        data_content = self.format_data_content([target_number], domain_context, format_template)

        # Generate domain-appropriate paths and cells
        data_filename = task_data["file_name"]
        target_cell = self.generate_cell_reference(["A", "B", "C", "D"], 8)  # Expanded range

        # Create contextual instruction with specific guidance
        data_type = domain_context["data_type"]

        # Create format-specific instruction to avoid ambiguity
        if format_template == "simple_values":
            value_instruction = "The standalone numeric value from the file should be identified (ignore headers or labels)"
        elif format_template == "labeled_entries":
            data_label = data_type.replace("_", " ").title()
            value_instruction = f"The value labeled 'Target {data_label}' needs to be located"
        elif format_template == "structured_records":
            value_instruction = f"The {data_type} value from the data record should be retrieved"
        else:  # formatted_reports
            value_instruction = "The main 'Value' from the data report needs to be identified (ignore headers and footers)"

        instruction = (
            f"Complete a data extraction task using '{data_filename}' from the Desktop with a text editor. "
            f"{value_instruction} and record this result in cell {target_cell} of a blank spreadsheet. "
            f"Save the completed analysis as 'result.xlsx' on the Desktop."
        )

        task_data.update(
            {
                "file_name": data_filename,
                "data_file_content": data_content,
                "data_filename": data_filename,
                # Keep for compatibility, but Desktop is the actual location used by setup
                "directory_path": "/home/user/Desktop",
                "target_number": target_number,
                "target_cell": target_cell,
                "instructions": instruction,
                "broken_file_content": data_content,
                "correct_file_content": str(target_number),
                "evaluation_data": {
                    "expected_value": target_number,
                    "target_cell": target_cell,
                    "evaluation_type": "single_cell",
                    "domain_context": domain_context,
                },
            }
        )

        return task_data
