"""
Utility functions, constants, and mappings for framework task generation.
"""

import uuid
import os
from datetime import datetime
from typing import Dict, List, Any


# Note: File placement and directory creation mappings have been moved to
# category-specific providers to follow the new architecture pattern.
# Use category.get_file_provider() methods instead of these hardcoded mappings.

# Evaluation method mapping to framework functions
EVALUATION_MAPPING = {
    "compare_text_file": {"func": "compare_text_file"},
    "compare_answer": {"func": "compare_answer"},
    "exact_match": {"func": "exact_match"},
    "file_contains": {"func": "file_contains"},
    "check_include_exclude": {"func": "check_include_exclude"},
}

# VS Code launch patterns based on task complexity
VSCODE_LAUNCH_PATTERNS = {
    "single_file": ["code", "{file_path}"],
    "directory": ["code", "{directory_path}"],
    "project": ["code", "{project_path}"],
}


def generate_task_instance_id() -> str:
    """Generate a pure UUID for task instance ID as required by schema."""
    return str(uuid.uuid4())


def generate_timestamp_prefix() -> str:
    """Generate timestamp prefix for S3 structure."""
    return datetime.now().strftime("task_configs_%Y%m%d_%H%M%S")


def get_file_placement_path(task_type: str, filename: str, file_provider=None) -> str:
    """
    Get the VM file placement path for a task type and filename.

    Args:
        task_type: Type of task (e.g., 'fix_missing_quote')
        filename: Name of the file
        file_provider: Optional file provider instance. If None, uses default path.

    Returns:
        Full path where file should be placed in VM
    """
    if file_provider and file_provider.supports_task_type(task_type):
        return file_provider.get_file_placement_path(task_type, filename)

    # Default to desktop if no provider or task type not supported
    return f"/home/user/Desktop/{filename}"


def get_directories_to_create(task_type: str, file_provider=None) -> List[str]:
    """
    Get list of directories that need to be created for a task type.

    Args:
        task_type: Type of task
        file_provider: Optional file provider instance. If None, returns empty list.

    Returns:
        List of directory paths to create
    """
    if file_provider and file_provider.supports_task_type(task_type):
        return file_provider.get_directories_to_create(task_type)

    return []


def get_evaluation_config(evaluation_method: str) -> Dict[str, Any]:
    """
    Get evaluation configuration for a given evaluation method.

    Args:
        evaluation_method: Method name from test examples

    Returns:
        Dictionary with evaluation configuration
    """
    return EVALUATION_MAPPING.get(evaluation_method, {"func": "compare_text_file"})


def generate_s3_key(base_prefix: str, task_id: str, filename: str) -> str:
    """
    Generate S3 key for file upload.

    Args:
        base_prefix: Base S3 prefix (e.g., 'task_configs_20250101_120000')
        task_id: Task instance UUID
        filename: Name of the file

    Returns:
        S3 key path
    """
    return f"{base_prefix}/{task_id}/{filename}"


def add_uuid_suffix_to_filename(filename: str, task_id: str) -> str:
    """
    Add UUID suffix to filename while preserving extension.

    Args:
        filename: Original filename
        task_id: Task UUID

    Returns:
        Filename with UUID suffix
    """
    name, ext = os.path.splitext(filename)
    short_id = task_id.replace("-", "")[:8]  # Use first 8 chars for readability
    return f"{name}_{short_id}{ext}"


def get_vscode_launch_command(task_type: str, file_path: str) -> List[str]:
    """
    Get appropriate VS Code launch command based on task type.

    Args:
        task_type: Type of task
        file_path: Path to file or directory

    Returns:
        Command list for launching VS Code
    """
    # Level 3 tasks get directory launch, others get file launch
    if task_type in [
        "write_simple_function",
        "fix_logic_error",
        "fix_variable_scope",
        "multi_file_config",
    ]:
        directory_path = os.path.dirname(file_path)
        return ["code", directory_path]
    else:
        return ["code", file_path]


def validate_test_example(test_example: Dict[str, Any]) -> bool:
    """
    Validate that a test example has required fields.

    Args:
        test_example: Test example dictionary

    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "task_type",
        "level",
        "seed",
        "instructions",
        "file_name",
        "broken_file_content",
        "correct_file_content",
        "evaluation_method",
    ]

    return all(field in test_example for field in required_fields)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system operations.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    import re

    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    return sanitized.strip()
