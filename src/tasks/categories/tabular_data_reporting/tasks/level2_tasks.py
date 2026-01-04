"""
Level 2 task implementations for data analysis: Basic Data Aggregation.
"""

from typing import Dict, Any, Optional
from .base_task import TabularDataReportingBaseTask


class BasicDataAggregationGenerator(TabularDataReportingBaseTask):
    """Generate dynamic basic_data_aggregation tasks."""

    def __init__(self):
        super().__init__("basic_data_aggregation", 2)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic basic data aggregation task with enhanced variability."""
        task_data = self.generate_enhanced_task_structure(seed)

        domain_context = task_data["domain_context"]
        format_template = task_data["format_template"]

        # Generate contextual list of numbers for aggregation (3-8 numbers)
        count = self.generate_random_number(3, 8)
        complexity = "medium"  # Level 2 uses medium complexity values
        numbers = self.generate_contextual_numbers(count, domain_context, complexity)

        # Format the numbers content based on domain and template
        data_content = self.format_data_content(numbers, domain_context, format_template)

        # Generate domain-appropriate paths and cells
        data_filename = task_data["file_name"]
        result_cell = self.generate_cell_reference(["D", "E", "F", "G"], 8)  # Expanded range

        # Domain-aware operation selection
        scenario = domain_context["scenario"]
        if scenario == "business_analytics":
            operations = [("SUM", "total revenue"), ("COUNT", "number of records"), ("AVERAGE", "average value")]
        elif scenario == "scientific_research":
            operations = [("SUM", "total measurements"), ("COUNT", "sample count"), ("AVERAGE", "mean value")]
        elif scenario == "education_management":
            operations = [("SUM", "total points"), ("COUNT", "number of students"), ("AVERAGE", "class average")]
        elif scenario == "inventory_logistics":
            operations = [("SUM", "total inventory"), ("COUNT", "item count"), ("AVERAGE", "average stock level")]
        else:  # financial_planning
            operations = [("SUM", "total amount"), ("COUNT", "number of items"), ("AVERAGE", "average cost")]

        operation, operation_desc = self.random.choice(operations)

        # Calculate expected result
        if operation == "SUM":
            expected_result = sum(numbers)
            operation_instruction = f"determine the {operation_desc}"
        elif operation == "COUNT":
            expected_result = len(numbers)
            operation_instruction = f"determine the {operation_desc}"
        else:  # AVERAGE
            expected_result = round(sum(numbers) / len(numbers), 2)
            operation_instruction = f"determine the {operation_desc}"

        # Create contextual instruction with specific guidance about which values to extract
        data_type = domain_context["data_type"]

        # Create format-specific instructions to avoid ambiguity about which numbers to use
        if format_template == "simple_values":
            extract_instruction = "All numeric values from the file need to be gathered (one per line)"
        elif format_template == "labeled_entries":
            data_label = data_type.replace("_", " ").title()
            extract_instruction = (
                f"Only the main data values after the colon should be collected (ignore entry numbers) from each '{data_label}' line"
            )
        elif format_template == "structured_records":
            extract_instruction = f"Only the {data_type} values from each data record should be collected (ignore record numbers and status)"
        else:  # formatted_reports
            extract_instruction = "Only the numeric values after 'Entry XX:' need to be gathered (ignore entry numbers, headers, and totals)"

        instruction = (
            f"Complete a data aggregation task using '{data_filename}' from the Desktop with a text editor or file viewer (avoid LibreOffice for proper data parsing). "
            f"{extract_instruction} and compile these values in column A of LibreOffice Calc spreadsheet. "
            f"Then {operation_instruction} in cell {result_cell}. "
            f"Save the completed analysis as 'result.xlsx' on the Desktop."
        )

        task_data.update(
            {
                "file_name": data_filename,
                "data_file_content": data_content,
                "data_filename": data_filename,
                # Keep for compatibility, but Desktop is the actual location used by setup
                "directory_path": "/home/user/Desktop",
                "numbers_list": numbers,
                "operation": operation,
                "expected_result": expected_result,
                "result_cell": result_cell,
                "instructions": instruction,
                "broken_file_content": data_content,
                "correct_file_content": str(expected_result),
                "evaluation_data": {
                    "expected_value": expected_result,
                    "target_cell": result_cell,
                    "evaluation_type": "aggregation_result",
                    "operation": operation,
                    "source_numbers": numbers,
                    "domain_context": domain_context,
                },
            }
        )

        return task_data
