"""
Level 2 task implementations for dynamic task creation.
Simple Logic Completion & File Output tasks.
"""

from typing import Dict, Any, Optional
from .base_task import BaseTaskGenerator


class SimpleLogicCompletionGenerator(BaseTaskGenerator):
    """Generate dynamic simple logic completion tasks with file output."""

    def __init__(self):
        super().__init__("simple_logic_completion", 2)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic logic completion task."""
        task_data = self.generate_basic_task_structure(seed)

        # Generate random elements
        script_name = f"calculator_{self.generate_random_number(100, 999)}.py"
        output_filename = f"results_{self.generate_random_number(100, 999)}.txt"

        # Random arithmetic operation and values
        operations = ["addition", "multiplication", "division"]
        operation = self.random.choice(operations)

        value1 = self.generate_random_number(10, 50)
        value2 = self.generate_random_number(2, 20)

        if operation == "addition":
            calculation_code = "result = num1 + num2"
            expected_result = value1 + value2
            operation_desc = "add"
        elif operation == "multiplication":
            calculation_code = "result = num1 * num2"
            expected_result = value1 * value2
            operation_desc = "multiply"
        else:  # division
            calculation_code = "result = num1 / num2"
            expected_result = value1 / value2
            operation_desc = "divide"

        # Task content with TODO
        broken_content = f"""# Simple calculator program
num1 = {value1}
num2 = {value2}

# TODO: Complete the calculation to {operation_desc} num1 and num2
# Store the result in a variable called 'result'

# Write result to file
with open("/home/user/coding_tasks/{output_filename}", "w") as f:
    f.write(f"Calculation result: {{result}}")

print(f"Calculation completed. Result written to {output_filename}")"""

        # Completed content
        correct_content = f"""# Simple calculator program
num1 = {value1}
num2 = {value2}

# TODO: Complete the calculation to {operation_desc} num1 and num2
# Store the result in a variable called 'result'
{calculation_code}

# Write result to file
with open("/home/user/coding_tasks/{output_filename}", "w") as f:
    f.write(f"Calculation result: {{result}}")

print(f"Calculation completed. Result written to {output_filename}")"""

        # Expected file content
        expected_file_content = f"Calculation result: {expected_result}"

        task_data.update(
            {
                "file_name": script_name,
                "broken_file_content": broken_content,
                "correct_file_content": correct_content,
                "instructions": f"Complete the Python script at /home/user/coding_tasks/{script_name} by implementing the missing logic in the TODO section to perform the required calculation. Ensure the script executes successfully and creates the output file '{output_filename}' with the correct calculated value.",
                "expected_output": f"Calculation completed. Result written to {output_filename}",
                "output_filename": output_filename,
                "expected_file_content": expected_file_content,
                "evaluation_mode": "multi_evaluator",  # Use multi-evaluator for file existence + content
                "evaluation_data": {
                    "ignore_whitespace": True,
                    "output_filename": output_filename,
                    "expected_file_content": expected_file_content,
                },
            }
        )

        return task_data


# Level 2 tasks are registered in __init__.py to avoid circular imports
