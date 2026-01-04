"""
Level 3 task implementations for dynamic task creation.
Multi-File Configuration Update & Verification tasks.
"""

from typing import Dict, Any, Optional
from .base_task import BaseTaskGenerator


class MultiFileConfigUpdateGenerator(BaseTaskGenerator):
    """Generate dynamic multi-file configuration update tasks."""

    def __init__(self):
        super().__init__("multi_file_config_update", 3)

    def generate_task_data(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate dynamic multi-file config update task."""
        task_data = self.generate_basic_task_structure(seed)

        # Generate random elements
        script_name = f"config_loader_{self.generate_random_number(100, 999)}.py"
        log_filename = f"system_log_{self.generate_random_number(100, 999)}.txt"

        # Random configuration values
        old_timeout = self.generate_random_number(30, 50)
        new_timeout = self.generate_random_number(60, 90)
        max_connections = self.generate_random_number(50, 100)
        app_version = self.random.choice(["2.1", "2.2", "2.3", "3.0"])
        log_level = self.random.choice(["INFO", "DEBUG", "WARNING"])

        # Config file types
        config_types = ["json", "ini"]
        config_type = self.random.choice(config_types)

        # Generate config filename based on the chosen format
        config_filename = f"app_config_{self.generate_random_number(100, 999)}.{config_type}"

        if config_type == "json":
            config_content = f"""{{
    "timeout": {old_timeout},
    "max_connections": {max_connections},
    "version": "{app_version}",
    "log_level": "{log_level}"
}}"""
            expected_config_content = f"""{{
    "timeout": {new_timeout},
    "max_connections": {max_connections},
    "version": "{app_version}",
    "log_level": "{log_level}"
}}"""

            script_content = f"""import json

# Load configuration
with open("/home/user/coding_tasks/{config_filename}", "r") as f:
    config = json.load(f)

# Use configuration
timeout = config["timeout"]
print(f"Configuration loaded successfully: timeout={{timeout}}")

# Create log file with configuration info
with open("/home/user/coding_tasks/{log_filename}", "w") as f:
    f.write(f"System initialized with timeout: {{timeout}} seconds\\n")
    f.write(f"Max connections: {{config['max_connections']}}\\n")
    f.write(f"Version: {{config['version']}}\\n")

print(f"Log file created: {log_filename}")"""

            instruction = f"Update the configuration file at /home/user/coding_tasks/{config_filename} by changing the 'timeout' value to {new_timeout}. Then ensure the Python script /home/user/coding_tasks/{script_name} executes successfully, reads the updated configuration, and creates a log file with the new timeout value."

        else:  # ini format
            config_content = f"""[system]
timeout = {old_timeout}
max_connections = {max_connections}

[application]
version = {app_version}
log_level = {log_level}"""

            expected_config_content = f"""[system]
timeout = {new_timeout}
max_connections = {max_connections}

[application]
version = {app_version}
log_level = {log_level}"""

            script_content = f"""import configparser

# Load configuration
config = configparser.ConfigParser()
config.read("/home/user/coding_tasks/{config_filename}")

# Use configuration
timeout = int(config['system']['timeout'])
print(f"Configuration loaded successfully: timeout={{timeout}}")

# Create log file with configuration info
with open("/home/user/coding_tasks/{log_filename}", "w") as f:
    f.write(f"System initialized with timeout: {{timeout}} seconds\\n")
    f.write(f"Max connections: {{config['system']['max_connections']}}\\n")
    f.write(f"Version: {{config['application']['version']}}\\n")

print(f"Log file created: {log_filename}")"""

            instruction = f"Update the configuration file at /home/user/coding_tasks/{config_filename} by changing the 'timeout' value to {new_timeout}. Then ensure the Python script /home/user/coding_tasks/{script_name} executes successfully, reads the updated configuration, and creates a log file with the new timeout value."

        # Expected outputs
        expected_output = f"Configuration loaded successfully: timeout={new_timeout}\nLog file created: {log_filename}"
        expected_log_content = f"System initialized with timeout: {new_timeout} seconds\nMax connections: {max_connections}\nVersion: {app_version}"

        # Create additional files
        additional_files = {config_filename: config_content}

        task_data.update(
            {
                "file_name": script_name,
                "broken_file_content": script_content,  # Script doesn't change, config file changes
                "correct_file_content": script_content,
                "instructions": instruction,
                "expected_output": expected_output,
                "config_filename": config_filename,
                "log_filename": log_filename,
                "expected_config_content": expected_config_content,
                "expected_log_content": expected_log_content,
                "new_timeout": new_timeout,
                "additional_files": additional_files,
                "evaluation_mode": "multi_evaluator",  # Use multi-evaluator for config + execution + file verification
                "evaluation_data": {
                    "ignore_whitespace": True,
                    "config_filename": config_filename,
                    "log_filename": log_filename,
                    "expected_config_content": expected_config_content,
                    "expected_log_content": expected_log_content,
                    "new_timeout": new_timeout,
                },
            }
        )

        return task_data


# Level 3 tasks are registered in __init__.py to avoid circular imports
