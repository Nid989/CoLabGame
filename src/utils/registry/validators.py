import re
import json
import ast
from enum import Enum
from typing import Optional, Tuple, Union, Dict, Any, List, Callable

from ..constants import COMPUTER13_ACTIONS
from .base import Registry


class Computer13ValidationErrorTypes(str, Enum):
    MULTIPLE_CODE_BLOCKS = "multiple_code_blocks"
    EMPTY_CODE_BLOCK = "empty_code_block"
    INVALID_JSON = "invalid_json"
    MISSING_ACTION_TYPE = "missing_action_type"
    INVALID_ACTION_TYPE = "invalid_action_type"
    MISSING_REQUIRED_PARAM = "missing_required_param"
    INVALID_PARAM_TYPE = "invalid_param_type"
    PARAM_OUT_OF_RANGE = "param_out_of_range"
    UNEXPECTED_PARAM = "unexpected_param"


class PyAutoGUIValidationErrorTypes(str, Enum):
    MULTIPLE_CODE_BLOCKS = "multiple_code_blocks"
    EMPTY_CODE_BLOCK = "empty_code_block"
    INVALID_PYTHON = "invalid_python"
    FORBIDDEN_FUNCTION = "forbidden_function"


class DoneOrFailValidationErrorTypes(str, Enum):
    MULTIPLE_CODE_BLOCKS = "multiple_code_blocks"
    EMPTY_CODE_BLOCK = "empty_code_block"
    INVALID_STATUS = "invalid_status"


class QueryValidationErrorTypes(str, Enum):
    MULTIPLE_CODE_BLOCKS = "multiple_code_blocks"
    EMPTY_CODE_BLOCK = "empty_code_block"


class ResponseValidationErrorTypes(str, Enum):
    MULTIPLE_CODE_BLOCKS = "multiple_code_blocks"
    EMPTY_CODE_BLOCK = "empty_code_block"


class GeneralValidationErrorTypes(str, Enum):
    UNRECOGNIZED_FORMAT = "unrecognized_format"


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
        """Initialize a ValidationError with error type, message, and optional details.

        Args:
            error_type (ValidationErrorType): The type of validation error.
            message (str): A descriptive error message.
            details (Optional[Dict[str, Any]]): Additional error context (default: None).
        """
        self.error_type = error_type
        self.message = message
        self.details = details or {}

    def get_dict(self) -> Dict[str, Any]:
        """Convert the ValidationError to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary with error type, message, and details.
        """
        return {"type": self.error_type, "message": self.message, **self.details}


def is_computer13_format(utterance: str) -> bool:
    """Check if an utterance matches the Computer13 action format.

    This is a lightweight structural check without detailed content validation.

    Args:
        utterance (str): The utterance to check.

    Returns:
        bool: True if the utterance matches the Computer13 format, False otherwise.
    """
    if "EXECUTE" not in utterance:
        return False
    execute_match = re.search(r"EXECUTE\s*(.*)", utterance, re.DOTALL)
    if not execute_match or not execute_match.group(1).strip():
        return False
    execute_content = execute_match.group(1).strip()
    return bool(re.search(r"```json\s*\n*(.*?)```", execute_content, re.DOTALL))


def is_pyautogui_format(utterance: str) -> bool:
    """Check if an utterance matches the PyAutoGUI action format.

    This is a lightweight structural check without detailed content validation.

    Args:
        utterance (str): The utterance to check.

    Returns:
        bool: True if the utterance matches the PyAutoGUI format, False otherwise.
    """
    if "EXECUTE" not in utterance:
        return False
    execute_match = re.search(r"EXECUTE\s*(.*)", utterance, re.DOTALL)
    if not execute_match or not execute_match.group(1).strip():
        return False
    execute_content = execute_match.group(1).strip()
    return bool(
        re.search(r"```(?:python|py)?\s*\n*(.*?)```", execute_content, re.DOTALL)
    )


def is_done_or_fail_format(utterance: str) -> bool:
    """Check if an utterance matches the STATUS format.

    This is a lightweight structural check without detailed content validation.

    Args:
        utterance (str): The utterance to check.

    Returns:
        bool: True if the utterance matches the STATUS format, False otherwise.
    """
    return bool(re.search(r"STATUS\s*\n\s*```[^`]*```", utterance, re.DOTALL))


def is_query_format(utterance: str) -> bool:
    """Check if an utterance matches the QUERY format.

    This is a lightweight structural check without detailed content validation.

    Args:
        utterance (str): The utterance to check.

    Returns:
        bool: True if the utterance matches the QUERY format, False otherwise.
    """
    return bool(re.search(r"QUERY\s*```.*?```", utterance, re.DOTALL))


def is_response_format(utterance: str) -> bool:
    """Check if an utterance matches the RESPONSE format.

    This is a lightweight structural check without detailed content validation.

    Args:
        utterance (str): The utterance to check.

    Returns:
        bool: True if the utterance matches the RESPONSE format, False otherwise.
    """
    return bool(re.search(r"RESPONSE\s*```.*?```", utterance, re.DOTALL))


validators = Registry[Callable[[str], Tuple[bool, Optional[ValidationError]]]]()


@validators.register("computer13_actions")
def validate_computer13_actions(
    utterance: str,
) -> Tuple[bool, Optional[ValidationError]]:
    """Validate an utterance against the Computer13 action format.

    Performs a two-stage validation: checks if the utterance matches the format, then validates
    JSON structure and parameters against the COMPUTER13_ACTIONS specification.

    Args:
        utterance (str): The utterance to validate.

    Returns:
        Tuple[bool, bool, Optional[ValidationError]]: A tuple containing:
            - bool: True if validation succeeds, False otherwise.
            - bool: True if the utterance matches the Computer13 format, False otherwise.
            - Optional[ValidationError]: Error details if validation fails, None otherwise.
    """
    if not is_computer13_format(utterance):
        return False, False, None

    execute_match = re.search(r"EXECUTE\s*(.*)", utterance, re.DOTALL)
    execute_content = execute_match.group(1).strip()

    json_blocks = re.findall(r"```json\s*\n*(.*?)```", execute_content, re.DOTALL)
    if len(json_blocks) > 1:
        return (
            False,
            True,
            ValidationError(
                Computer13ValidationErrorTypes.MULTIPLE_CODE_BLOCKS,
                "Multiple JSON code blocks found. Only one is allowed.",
                {"count": len(json_blocks)},
            ),
        )

    json_text = json_blocks[0].strip()
    if not json_text:
        return (
            False,
            True,
            ValidationError(
                Computer13ValidationErrorTypes.EMPTY_CODE_BLOCK,
                "JSON code block cannot be empty.",
            ),
        )

    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        return (
            False,
            True,
            ValidationError(
                Computer13ValidationErrorTypes.INVALID_JSON,
                "The content inside the JSON block is not valid JSON.",
            ),
        )

    if not isinstance(obj, dict):
        return (
            False,
            True,
            ValidationError(
                Computer13ValidationErrorTypes.INVALID_JSON,
                "The JSON block must contain a JSON object.",
            ),
        )

    if "action_type" not in obj:
        return (
            False,
            True,
            ValidationError(
                Computer13ValidationErrorTypes.MISSING_ACTION_TYPE,
                "Missing 'action_type' in action.",
            ),
        )

    action_type = obj["action_type"]
    action_spec = next(
        (
            action
            for action in COMPUTER13_ACTIONS
            if action["action_type"] == action_type
        ),
        None,
    )
    if action_spec is None:
        return (
            False,
            True,
            ValidationError(
                Computer13ValidationErrorTypes.INVALID_ACTION_TYPE,
                f"Invalid 'action_type': {action_type}.",
            ),
        )

    param_specs = action_spec["parameters"]
    for param, spec in param_specs.items():
        if param not in obj and not spec["optional"]:
            return (
                False,
                True,
                ValidationError(
                    Computer13ValidationErrorTypes.MISSING_REQUIRED_PARAM,
                    f"Missing required parameter '{param}' for '{action_type}'.",
                    {"param": param, "action_type": action_type},
                ),
            )
        if param in obj:
            value = obj[param]
            expected_type = spec["type"]
            if not isinstance(value, expected_type):
                return (
                    False,
                    True,
                    ValidationError(
                        Computer13ValidationErrorTypes.INVALID_PARAM_TYPE,
                        f"Parameter '{param}' must be {expected_type.__name__}, got {type(value).__name__}.",
                        {
                            "param": param,
                            "expected_type": expected_type.__name__,
                            "actual_type": type(value).__name__,
                        },
                    ),
                )
            if spec["range"] is not None:
                if isinstance(expected_type, (int, float)):
                    min_val, max_val = spec["range"]
                    if value < min_val or value > max_val:
                        return (
                            False,
                            True,
                            ValidationError(
                                Computer13ValidationErrorTypes.PARAM_OUT_OF_RANGE,
                                f"Parameter '{param}' must be between {min_val} and {max_val}, got {value}.",
                                {
                                    "param": param,
                                    "range": [min_val, max_val],
                                    "actual_value": value,
                                },
                            ),
                        )
                elif isinstance(expected_type, str):
                    allowed = spec["range"]
                    if value not in allowed:
                        return (
                            False,
                            True,
                            ValidationError(
                                Computer13ValidationErrorTypes.PARAM_OUT_OF_RANGE,
                                f"Parameter '{param}' must be one of {allowed}, got '{value}'.",
                                {
                                    "param": param,
                                    "allowed_values": allowed,
                                    "actual_value": value,
                                },
                            ),
                        )

    allowed_keys = set(param_specs.keys()) | {"action_type"}
    for key in obj:
        if key not in allowed_keys:
            return (
                False,
                True,
                ValidationError(
                    Computer13ValidationErrorTypes.UNEXPECTED_PARAM,
                    f"Unexpected parameter '{key}' for '{action_type}'.",
                    {"param": key, "action_type": action_type},
                ),
            )

    return True, True, None


@validators.register("pyautogui_actions")
def validate_pyautogui_actions(
    utterance: str,
) -> Tuple[bool, Optional[ValidationError]]:
    """Validate an utterance against the PyAutoGUI action format.

    Performs a two-stage validation: checks if the utterance matches the format, then validates
    Python syntax and checks for forbidden functions.

    Args:
        utterance (str): The utterance to validate.

    Returns:
        Tuple[bool, bool, Optional[ValidationError]]: A tuple containing:
            - bool: True if validation succeeds, False otherwise.
            - bool: True if the utterance matches the PyAutoGUI format, False otherwise.
            - Optional[ValidationError]: Error details if validation fails, None otherwise.
    """
    if not is_pyautogui_format(utterance):
        return False, False, None

    execute_match = re.search(r"EXECUTE\s*(.*)", utterance, re.DOTALL)
    execute_content = execute_match.group(1).strip()

    code_blocks = re.findall(
        r"```(?:python|py)?\s*\n*(.*?)```", execute_content, re.DOTALL
    )
    if len(code_blocks) > 1:
        return (
            False,
            True,
            ValidationError(
                PyAutoGUIValidationErrorTypes.MULTIPLE_CODE_BLOCKS,
                "Multiple Python code blocks found. Only one is allowed.",
                {"count": len(code_blocks)},
            ),
        )

    code_text = code_blocks[0].strip()
    if not code_text:
        return (
            False,
            True,
            ValidationError(
                PyAutoGUIValidationErrorTypes.EMPTY_CODE_BLOCK,
                "Python code block cannot be empty.",
            ),
        )

    try:
        ast.parse(code_text)
    except SyntaxError as e:
        return (
            False,
            True,
            ValidationError(
                PyAutoGUIValidationErrorTypes.INVALID_PYTHON,
                f"Invalid Python syntax: {str(e)}.",
            ),
        )

    forbidden_functions = ["locateCenterOnScreen", "screenshot"]
    ast_tree = ast.parse(code_text)
    for node in ast.walk(ast_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            if (
                node.func.value.id == "pyautogui"
                and node.func.attr in forbidden_functions
            ):
                return (
                    False,
                    True,
                    ValidationError(
                        PyAutoGUIValidationErrorTypes.FORBIDDEN_FUNCTION,
                        f"Forbidden function '{node.func.attr}' used.",
                    ),
                )

    return True, True, None


@validators.register("done_or_fail")
def validate_done_or_fail(utterance: str) -> Tuple[bool, Optional[ValidationError]]:
    """Validate an utterance against the STATUS format.

    Performs a two-stage validation: checks if the utterance matches the format, then validates
    the status keyword (DONE or FAIL).

    Args:
        utterance (str): The utterance to validate.

    Returns:
        Tuple[bool, bool, Optional[ValidationError]]: A tuple containing:
            - bool: True if validation succeeds, False otherwise.
            - bool: True if the utterance matches the STATUS format, False otherwise.
            - Optional[ValidationError]: Error details if validation fails, None otherwise.
    """
    if not is_done_or_fail_format(utterance):
        return False, False, None

    status_sections = re.findall(r"STATUS\s*\n\s*```[^`]*```", utterance, re.DOTALL)
    if len(status_sections) > 1:
        return (
            False,
            True,
            ValidationError(
                DoneOrFailValidationErrorTypes.MULTIPLE_CODE_BLOCKS,
                "Multiple STATUS sections found. Only one is allowed.",
                {"count": len(status_sections)},
            ),
        )

    status_section = status_sections[0]
    content_match = re.search(r"```([^`]*)```", status_section, re.DOTALL)
    if not content_match:
        return (
            False,
            True,
            ValidationError(
                DoneOrFailValidationErrorTypes.INVALID_STATUS,
                "Invalid STATUS section format.",
            ),
        )

    block_content = content_match.group(1).strip()
    if not block_content:
        return (
            False,
            True,
            ValidationError(
                DoneOrFailValidationErrorTypes.EMPTY_CODE_BLOCK,
                "Status code block cannot be empty.",
            ),
        )

    if block_content not in ["DONE", "FAIL"]:
        return (
            False,
            True,
            ValidationError(
                DoneOrFailValidationErrorTypes.INVALID_STATUS,
                "Invalid status: must be DONE or FAIL.",
                {"block_content": block_content},
            ),
        )

    return True, True, None


@validators.register("query")
def validate_query(utterance: str) -> Tuple[bool, Optional[ValidationError]]:
    """Validate an utterance against the QUERY format.

    Performs a two-stage validation: checks if the utterance matches the format, then validates
    the presence of non-empty content.

    Args:
        utterance (str): The utterance to validate.

    Returns:
        Tuple[bool, bool, Optional[ValidationError]]: A tuple containing:
            - bool: True if validation succeeds, False otherwise.
            - bool: True if the utterance matches the QUERY format, False otherwise.
            - Optional[ValidationError]: Error details if validation fails, None otherwise.
    """
    if not is_query_format(utterance):
        return False, False, None

    query_blocks = re.findall(r"QUERY\s*```(.*?)```", utterance, re.DOTALL)
    if len(query_blocks) > 1:
        return (
            False,
            True,
            ValidationError(
                QueryValidationErrorTypes.MULTIPLE_CODE_BLOCKS,
                "Multiple QUERY sections found. Only one is allowed.",
                {"count": len(query_blocks)},
            ),
        )

    if not query_blocks[0].strip():
        return (
            False,
            True,
            ValidationError(
                QueryValidationErrorTypes.EMPTY_CODE_BLOCK,
                "QUERY code block cannot be empty.",
            ),
        )

    return True, True, None


@validators.register("response")
def validate_response(utterance: str) -> Tuple[bool, Optional[ValidationError]]:
    """Validate an utterance against the RESPONSE format.

    Performs a two-stage validation: checks if the utterance matches the format, then validates
    the presence of non-empty content.

    Args:
        utterance (str): The utterance to validate.

    Returns:
        Tuple[bool, bool, Optional[ValidationError]]: A tuple containing:
            - bool: True if validation succeeds, False otherwise.
            - bool: True if the utterance matches the RESPONSE format, False otherwise.
            - Optional[ValidationError]: Error details if validation fails, None otherwise.
    """
    if not is_response_format(utterance):
        return False, False, None

    response_blocks = re.findall(r"RESPONSE\s*```(.*?)```", utterance, re.DOTALL)
    if len(response_blocks) > 1:
        return (
            False,
            True,
            ValidationError(
                ResponseValidationErrorTypes.MULTIPLE_CODE_BLOCKS,
                "Multiple RESPONSE sections found. Only one is allowed.",
                {"count": len(response_blocks)},
            ),
        )

    if not response_blocks[0].strip():
        return (
            False,
            True,
            ValidationError(
                ResponseValidationErrorTypes.EMPTY_CODE_BLOCK,
                "RESPONSE code block cannot be empty.",
            ),
        )

    return True, True, None


def raise_unrecognized_format_error(allowed_components: List[str]) -> ValidationError:
    """Create a ValidationError for unrecognized format with actual format examples.

    Args:
        allowed_components: List of allowed format components for the player

    Returns:
        ValidationError: Error with message showing allowed format structures
    """
    format_structures = {
        "computer13": """EXECUTE
```json
{
    "action_type": "<action_type>",
    ...parameters
}
```""",
        "pyautogui": """EXECUTE
```python
pyautogui.<action>(...parameters)
```""",
        "done_or_fail": """STATUS
```
DONE
```
or
STATUS
```
FAIL
```""",
        "query": """QUERY
```
<query_text>
```""",
        "response": """RESPONSE
```
<response_text>
```""",
    }

    allowed_formats = [
        format_structures[component]
        for component in allowed_components
        if component in format_structures
    ]
    format_list = "\n\n".join(allowed_formats)
    message = f"Response format is invalid. Your response must follow one of these formats: \n\n{format_list}"
    return ValidationError(
        error_type=GeneralValidationErrorTypes.UNRECOGNIZED_FORMAT,
        message=message,
        details={"allowed_components": allowed_components},
    )
