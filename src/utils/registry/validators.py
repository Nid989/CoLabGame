import re
import json
import ast
from enum import Enum
from typing import Optional, Tuple, Union, Dict, Any, Callable

from ..constants import ACTION_SPACE
from .base import Registry


class Computer13ValidationErrorTypes(str, Enum):
    MULTIPLE_EXECUTE = "multiple execute blocks"
    MULTIPLE_JSON_BLOCKS = "multiple json blocks"
    INVALID_JSON = "invalid json content"
    MISSING_ACTION_TYPE = "missing action type"
    INVALID_ACTION_TYPE = "invalid action type"
    MISSING_REQUIRED_PARAM = "missing required parameter"
    INVALID_PARAM_TYPE = "invalid parameter type"
    PARAM_OUT_OF_RANGE = "parameter out of range"
    UNEXPECTED_PARAM = "unexpected parameter"


class PyAutoGUIValidationErrorTypes(str, Enum):
    MULTIPLE_EXECUTE = "multiple execute blocks"
    MISSING_CODE_BLOCK = "missing code block"
    MULTIPLE_CODE_BLOCKS = "multiple code blocks"
    EMPTY_CODE_BLOCK = "empty code block"
    INVALID_PYTHON = "invalid python syntax"
    FORBIDDEN_FUNCTION = "forbidden function"


class DoneOrFailValidationErrorTypes(str, Enum):
    INVALID_STATUS = "invalid status keyword"
    MULTIPLE_KEYWORDS = "multiple status keywords"


class QueryValidationErrorTypes(str, Enum):
    MULTIPLE_QUERY = "multiple query tags"
    MULTIPLE_QUERY_BLOCKS = "multiple query blocks"
    EMPTY_QUERY = "empty query content"


class ResponseValidationErrorTypes(str, Enum):
    MULTIPLE_RESPONSE = "multiple response tags"
    MULTIPLE_RESPONSE_BLOCKS = "multiple response blocks"
    EMPTY_RESPONSE = "empty response content"


ValidationErrorType = Union[
    Computer13ValidationErrorTypes,
    PyAutoGUIValidationErrorTypes,
    DoneOrFailValidationErrorTypes,
    QueryValidationErrorTypes,
    ResponseValidationErrorTypes,
]


class ValidationError:
    def __init__(
        self,
        error_type: ValidationErrorType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}

    def get_dict(self) -> Dict[str, Any]:
        return {"type": self.error_type, "message": self.message, **self.details}


def is_computer13_format(utterance: str) -> bool:
    """Check if an utterance is intended for the Computer13 action format.
    This is a lightweight structural check that determines if the basic format matches,
    without validating the content details or raising errors.
    Args:
        utterance (str): The utterance to check
    Returns:
        bool: True if the utterance appears to be intended for Computer13 format, False otherwise
    """
    if "EXECUTE" not in utterance:
        return False

    execute_match = re.search(r"EXECUTE\s*(.*)", utterance, re.DOTALL)
    if not execute_match or not execute_match.group(1).strip():
        return False

    execute_content = execute_match.group(1).strip()

    json_block_pattern = re.search(r"```json\s*\n*(.*?)```", execute_content, re.DOTALL)
    if not json_block_pattern:
        return False

    return True


def is_pyautogui_format(utterance: str) -> bool:
    """Check if an utterance is intended for the PyAutoGUI action format.
    Args:
        utterance (str): The utterance to check
    Returns:
        bool: True if the utterance appears to be intended for Computer13 format, False otherwise
    """
    if "EXECUTE" not in utterance:
        return False

    execute_match = re.search(r"EXECUTE\s*(.*)", utterance, re.DOTALL)
    if not execute_match or not execute_match.group(1).strip():
        return False

    execute_content = execute_match.group(1).strip()
    code_block_pattern = re.search(
        r"```(?:python|py)?\s*\n*(.*?)```", execute_content, re.DOTALL
    )

    return bool(code_block_pattern)


def is_done_or_fail_format(utterance: str) -> bool:
    """Check if an utterance contains a valid status keyword.
    Args:
        utterance (str): The utterance to check
    Returns:
        bool: True if the utterance contains a status keyword
    """
    status_keywords = ["DONE", "FAIL"]
    return any(re.search(rf"\b{keyword}\b", utterance) for keyword in status_keywords)


def is_query_format(utterance: str) -> bool:
    """Check if an utterance follows the QUERY block format.
    Args:
        utterance (str): The utterance to check
    Returns:
        bool: True if the utterance matches QUERY format
    """
    return bool(re.search(r"QUERY\s*```.*?```", utterance, re.DOTALL))


def is_response_format(utterance: str) -> bool:
    """Check if an utterance follows the RESPONSE block format.
    Args:
        utterance (str): The utterance to check
    Returns:
        bool: True if the utterance matches RESPONSE format
    """
    return bool(re.search(r"RESPONSE\s*```.*?```", utterance, re.DOTALL))


validators = Registry[Callable[[str], Tuple[bool, Optional[ValidationError]]]]()


@validators.register("computer13_actions")
def validate_computer13_actions(
    utterance: str,
) -> Tuple[bool, Optional[ValidationError]]:
    """Validates that the utterance strictly follows the Computer13 action format.
    This is a two-stage validation process:
        1. First, we check if the utterance is intended for this format using is_computer13_format()
            If not, we should return early without raising format-specific errors
        2. If the utterance is intended for this format, we perform detailed validation
            including structure and content validation against ACTION_SPACE
    Args:
        utterance (str): The utterance to validate
    Returns:
        Tuple containing:
        - Boolean indicating if validation was successful
        - ValidationError object if validation failed, None otherwise
    """
    if not is_computer13_format(utterance):
        return False, None

    if utterance.count("EXECUTE") > 1:
        return False, ValidationError(
            Computer13ValidationErrorTypes.MULTIPLE_EXECUTE,
            "Utterance must contain exactly one 'EXECUTE' block.",
        )

    execute_match = re.search(r"EXECUTE\s*(.*)", utterance, re.DOTALL)
    execute_content = execute_match.group(1).strip()

    json_blocks = re.findall(r"```json\s*\n*(.*?)```", execute_content, re.DOTALL)
    if len(json_blocks) > 1:
        return False, ValidationError(
            Computer13ValidationErrorTypes.MULTIPLE_JSON_BLOCKS,
            "Only one ```json ...``` block is allowed.",
        )

    json_text = json_blocks[0].strip()
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        return False, ValidationError(
            Computer13ValidationErrorTypes.INVALID_JSON,
            "The content inside the json block is not valid JSON.",
        )

    if not isinstance(obj, dict):
        return False, ValidationError(
            Computer13ValidationErrorTypes.INVALID_JSON,
            "The json block must contain a JSON object.",
        )

    # Validate against ACTION_SPACE
    if "action_type" not in obj:
        return False, ValidationError(
            Computer13ValidationErrorTypes.MISSING_ACTION_TYPE,
            "Missing 'action_type' in action.",
        )

    action_type = obj["action_type"]
    action_spec = next(
        (action for action in ACTION_SPACE if action["action_type"] == action_type),
        None,
    )

    if action_spec is None:
        return False, ValidationError(
            Computer13ValidationErrorTypes.INVALID_ACTION_TYPE,
            f"Invalid 'action_type': {action_type}",
        )

    param_specs = action_spec["parameters"]

    # Validate parameters
    for param, spec in param_specs.items():
        if param not in obj:
            if not spec["optional"]:
                return False, ValidationError(
                    Computer13ValidationErrorTypes.MISSING_REQUIRED_PARAM,
                    f"Missing required parameter '{param}' for action_type '{action_type}'.",
                    {"param": param, "action_type": action_type},
                )
            continue

        value = obj[param]
        expected_type = spec["type"]

        # Handle type validation
        if expected_type is list:
            if not isinstance(value, list):
                return False, ValidationError(
                    Computer13ValidationErrorTypes.INVALID_PARAM_TYPE,
                    f"Parameter '{param}' must be a list, got {type(value).__name__}.",
                    {
                        "param": param,
                        "expected_type": "list",
                        "actual_type": type(value).__name__,
                    },
                )
            if spec["range"] is not None:
                allowed = spec["range"]
                if not all(isinstance(v, str) and v in allowed for v in value):
                    return False, ValidationError(
                        Computer13ValidationErrorTypes.PARAM_OUT_OF_RANGE,
                        f"All elements in '{param}' must be strings in {allowed}, got {value}.",
                        {
                            "param": param,
                            "allowed_values": allowed,
                            "actual_value": value,
                        },
                    )
        else:
            if not isinstance(value, expected_type):
                return False, ValidationError(
                    Computer13ValidationErrorTypes.INVALID_PARAM_TYPE,
                    f"Parameter '{param}' must be of type {expected_type.__name__}, got {type(value).__name__}.",
                    {
                        "param": param,
                        "expected_type": expected_type.__name__,
                        "actual_type": type(value).__name__,
                    },
                )

        # Handle range validation
        if spec["range"] is not None:
            if isinstance(expected_type, (int, float)):
                min_val, max_val = spec["range"]
                if value < min_val or value > max_val:
                    return False, ValidationError(
                        Computer13ValidationErrorTypes.PARAM_OUT_OF_RANGE,
                        f"Parameter '{param}' out of allowed range [{min_val}, {max_val}], got {value}.",
                        {
                            "param": param,
                            "range": [min_val, max_val],
                            "actual_value": value,
                        },
                    )
            elif isinstance(expected_type, str):
                allowed = spec["range"]
                if value not in allowed:
                    return False, ValidationError(
                        Computer13ValidationErrorTypes.PARAM_OUT_OF_RANGE,
                        f"Parameter '{param}' must be one of {allowed}, got '{value}'.",
                        {
                            "param": param,
                            "allowed_values": allowed,
                            "actual_value": value,
                        },
                    )

    # Check for unexpected parameters
    allowed_keys = set(param_specs.keys()) | {"action_type"}
    for key in obj:
        if key not in allowed_keys:
            return False, ValidationError(
                Computer13ValidationErrorTypes.UNEXPECTED_PARAM,
                f"Unexpected parameter '{key}' for action_type '{action_type}'.",
                {"param": key, "action_type": action_type},
            )

    return True, None


@validators.register("pyautogui_actions")
def validate_pyautogui_actions(
    utterance: str,
) -> Tuple[bool, Optional[ValidationError]]:
    """Validates that the utterance follows the PyAutoGUI action format.
    Args:
        utterance (str): The utterance to validate
    Returns:
        Tuple containing:
        - Boolean indicating if validation was successful
        - ValidationError object if validation failed, None otherwise
    """
    if not is_pyautogui_format(utterance):
        return False, None

    if utterance.count("EXECUTE") > 1:
        return False, ValidationError(
            PyAutoGUIValidationErrorTypes.MULTIPLE_EXECUTE,
            "Utterance must contain exactly one 'EXECUTE' block.",
        )

    execute_match = re.search(r"EXECUTE\s*(.*)", utterance, re.DOTALL)
    execute_content = execute_match.group(1).strip()

    code_blocks = re.findall(
        r"```(?:python|py)?\s*\n*(.*?)```", execute_content, re.DOTALL
    )

    if len(code_blocks) > 1:
        return False, ValidationError(
            PyAutoGUIValidationErrorTypes.MULTIPLE_CODE_BLOCKS,
            "Only one code block is allowed.",
        )

    code_text = code_blocks[0].strip()
    if not code_text:
        return False, ValidationError(
            PyAutoGUIValidationErrorTypes.EMPTY_CODE_BLOCK,
            "Code block cannot be empty.",
        )

    # Validate Python syntax
    try:
        ast_tree = ast.parse(code_text)
    except SyntaxError as e:
        return False, ValidationError(
            PyAutoGUIValidationErrorTypes.INVALID_PYTHON,
            f"Invalid Python syntax: {str(e)}",
        )

    # Check for forbidden functions
    forbidden_functions = ["locateCenterOnScreen", "screenshot"]

    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                if (
                    node.func.value.id == "pyautogui"
                    and node.func.attr in forbidden_functions
                ):
                    return False, ValidationError(
                        PyAutoGUIValidationErrorTypes.FORBIDDEN_FUNCTION,
                        f"Forbidden PyAutoGUI function used: {node.func.attr}",
                    )

    return True, None


@validators.register("done_or_fail")
def validate_done_or_fail(utterance: str) -> Tuple[bool, Optional[ValidationError]]:
    """Validates that the utterance contains a valid status keyword.
    Args:
        utterance (str): The utterance to validate
    Returns:
        Tuple containing:
        - Boolean indicating if validation was successful
        - ValidationError object if validation failed, None otherwise
    """
    if not is_done_or_fail_format(utterance):
        return False, None

    status_keywords = ["DONE", "FAIL"]
    # Count occurrences of each keyword
    keyword_counts = {
        k: len(re.findall(rf"\b{k}\b", utterance)) for k in status_keywords
    }

    # Check for multiple different keywords
    found_keywords = [k for k, count in keyword_counts.items() if count > 0]
    if len(found_keywords) > 1:
        return False, ValidationError(
            DoneOrFailValidationErrorTypes.MULTIPLE_KEYWORDS,
            f"Multiple status keywords found ({', '.join(found_keywords)}). Use only one of: DONE, FAIL",
            {"found_keywords": found_keywords},
        )

    # Check for repeating keywords
    for keyword, count in keyword_counts.items():
        if count > 1:
            return False, ValidationError(
                DoneOrFailValidationErrorTypes.MULTIPLE_KEYWORDS,
                f"Status keyword '{keyword}' appears multiple times ({count}). Should appear exactly once.",
                {"keyword": keyword, "count": count},
            )

    return True, None


@validators.register("query")
def validate_query(utterance: str) -> Tuple[bool, Optional[ValidationError]]:
    """Validates that the utterance follows the QUERY block format.
    Args:
        utterance (str): The utterance to validate
    Returns:
        Tuple containing:
        - Boolean indicating if validation was successful
        - ValidationError object if validation failed, None otherwise
    """
    if not is_query_format(utterance):
        return False, None

    # Check for multiple QUERY tags first
    if utterance.count("QUERY") > 1:
        return False, ValidationError(
            QueryValidationErrorTypes.MULTIPLE_QUERY,
            "Utterance must contain exactly one 'QUERY' tag.",
            {"count": utterance.count("QUERY")},
        )

    # Then check for multiple code blocks
    query_blocks = re.findall(r"QUERY\s*```(.*?)```", utterance, re.DOTALL)
    if len(query_blocks) > 1:
        return False, ValidationError(
            QueryValidationErrorTypes.MULTIPLE_QUERY_BLOCKS,
            "Multiple QUERY blocks found. Only one is allowed",
            {"count": len(query_blocks)},
        )

    if not query_blocks[0].strip():
        return False, ValidationError(
            QueryValidationErrorTypes.EMPTY_QUERY, "QUERY block content cannot be empty"
        )

    return True, None


@validators.register("response")
def validate_response(utterance: str) -> Tuple[bool, Optional[ValidationError]]:
    """Validates that the utterance follows the RESPONSE block format.
    Args:
        utterance (str): The utterance to validate
    Returns:
        Tuple containing:
        - Boolean indicating if validation was successful
        - ValidationError object if validation failed, None otherwise
    """
    if not is_response_format(utterance):
        return False, None

    # Check for multiple RESPONSE tags first
    if utterance.count("RESPONSE") > 1:
        return False, ValidationError(
            ResponseValidationErrorTypes.MULTIPLE_RESPONSE,
            "Utterance must contain exactly one 'RESPONSE' tag.",
            {"count": utterance.count("RESPONSE")},
        )

    # Then check for multiple code blocks
    response_blocks = re.findall(r"RESPONSE\s*```(.*?)```", utterance, re.DOTALL)
    if len(response_blocks) > 1:
        return False, ValidationError(
            ResponseValidationErrorTypes.MULTIPLE_RESPONSE_BLOCKS,
            "Multiple RESPONSE blocks found. Only one is allowed",
            {"count": len(response_blocks)},
        )

    if not response_blocks[0].strip():
        return False, ValidationError(
            ResponseValidationErrorTypes.EMPTY_RESPONSE,
            "RESPONSE block content cannot be empty",
        )

    return True, None
