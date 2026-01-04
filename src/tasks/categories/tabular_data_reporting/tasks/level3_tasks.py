"""
Level 3 task implementations for data analysis: Simple Calculation and Output.
"""

from typing import Dict, Any, Optional
from .base_task import TabularDataReportingBaseTask


class SimpleCalculationOutputGenerator(TabularDataReportingBaseTask):
    """Generate dynamic simple_calculation_output tasks."""

    def __init__(self):
        super().__init__("simple_calculation_output", 3)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic simple calculation and output task with enhanced variability."""
        task_data = self.generate_enhanced_task_structure(seed)

        domain_context = task_data["domain_context"]

        # Generate contextual spreadsheet data (3-5 rows, 2 columns)
        row_count = self.generate_random_number(3, 5)
        complexity = "high"  # Level 3 uses more complex values
        column_a_values = self.generate_contextual_numbers(row_count, domain_context, complexity)
        column_b_values = self.generate_contextual_numbers(row_count, domain_context, complexity)

        # Domain-aware calculation types with contextual descriptions that match the actual data structure
        scenario = domain_context["scenario"]
        if scenario == "business_analytics":
            calc_types = [
                ("sum_columns", "calculate total business performance (revenue + costs)", "combined business total"),
                ("multiply_columns", "calculate revenue efficiency (revenue × cost ratio)", "efficiency score"),
                ("average_column_a", "calculate average revenue", "mean revenue"),
            ]
        elif scenario == "scientific_research":
            calc_types = [
                ("sum_columns", "calculate total sensor readings (Sensor_A + Sensor_B)", "combined sensor total"),
                ("multiply_columns", "calculate interaction coefficient (Sensor_A × Sensor_B)", "interaction result"),
                ("average_column_a", "calculate average Sensor_A reading", "mean sensor value"),
            ]
        elif scenario == "education_management":
            calc_types = [
                ("sum_columns", "calculate total test scores (Test_1 + Test_2)", "overall score total"),
                ("multiply_columns", "calculate weighted performance (Test_1 × Test_2)", "performance index"),
                ("average_column_a", "calculate average Test_1 score", "mean test score"),
            ]
        elif scenario == "inventory_logistics":
            calc_types = [
                ("sum_columns", "calculate total stock levels (Stock_A + Stock_B)", "combined inventory"),
                ("multiply_columns", "calculate stock turnover (Stock_A × Stock_B)", "turnover rate"),
                ("average_column_a", "calculate average Stock_A level", "mean stock level"),
            ]
        else:  # financial_planning
            calc_types = [
                ("sum_columns", "calculate total financial flow (Income + Expenses)", "combined total"),
                ("multiply_columns", "calculate financial impact (Income × Expenses)", "impact value"),
                ("average_column_a", "calculate average income", "mean income"),
            ]

        calc_type, calc_description, result_description = self.random.choice(calc_types)

        # Calculate expected result with domain-aware descriptions
        if calc_type == "sum_columns":
            expected_result = sum(column_a_values) + sum(column_b_values)
            calculation_instruction = f"{calc_description}"
            formula_hint = f"=SUM(A1:A{row_count})+SUM(B1:B{row_count})"
        elif calc_type == "multiply_columns":
            expected_result = sum(a * b for a, b in zip(column_a_values, column_b_values))
            calculation_instruction = f"{calc_description}"
            formula_hint = f"=SUMPRODUCT(A1:A{row_count},B1:B{row_count})"
        else:  # average_column_a
            expected_result = round(sum(column_a_values) / len(column_a_values), 2)
            calculation_instruction = f"{calc_description}"
            formula_hint = f"=AVERAGE(A1:A{row_count})"

        # Generate contextual cell and file details
        result_cell = self.generate_cell_reference(["C", "D", "E"], row_count + 5)  # Expanded range
        output_filename = f"{domain_context['file_prefix']}_output.txt"
        output_path = "/home/user/Desktop"

        # Generate contextual Excel filename BEFORE using it
        spreadsheet_filename = f"{domain_context['file_prefix']}_analysis_data.xlsx"

        # Create contextual instruction with clear spreadsheet guidance
        context_desc = domain_context["context_description"]

        instruction = (
            f"Complete a data analysis task using the pre-populated spreadsheet '{spreadsheet_filename}' from the Desktop. "
            f"Using the values in the spreadsheet, {calculation_instruction}. "
            f"Record the {result_description} in cell {result_cell} and "
            f"save the updated spreadsheet as 'result.xlsx'. Additionally, save just the calculated result "
            f"to 'output.txt'."
        )

        # Create domain-aware content representation
        broken_spreadsheet_content = f"Task: {calc_description} for {context_desc} analysis"
        correct_output = str(expected_result)

        task_data.update(
            {
                "file_name": spreadsheet_filename,
                "spreadsheet_data": {
                    "column_a": column_a_values,
                    "column_b": column_b_values,
                    "row_count": row_count,
                },
                "calculation_type": calc_type,
                "calculation_instruction": calculation_instruction,
                "formula_hint": formula_hint,
                "expected_result": expected_result,
                "result_cell": result_cell,
                "output_filename": output_filename,
                "output_path": output_path,
                "instructions": instruction,
                "broken_file_content": broken_spreadsheet_content,
                "correct_file_content": correct_output,
                "evaluation_data": {
                    "spreadsheet_expected_value": expected_result,
                    "spreadsheet_target_cell": result_cell,
                    "output_file_path": f"{output_path}/{output_filename}",
                    "output_expected_content": str(expected_result),
                    "evaluation_type": "calculation_and_output",
                    "calculation_type": calc_type,
                    "domain_context": domain_context,
                },
            }
        )

        return task_data
