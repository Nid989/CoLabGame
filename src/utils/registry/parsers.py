import ast
from typing import Callable, Optional, Tuple, Dict, Any, List

from .base import Registry
from ..constants import COMPUTER13_ACTIONS
from clemcore.clemgame import ParseError, RuleViolationError

# Registry for parsers, keyed by (message_type, environment_type, action_space)
# None for action_space means the parser applies to all action spaces for that environment/message type
parsers = Registry[Callable[[Any, str, Optional[str]], Tuple[bool, Optional[Any]]]]()


# Function to store metadata about parsers
def parser_config(target_field: Optional[str], description: Optional[str] = None):
    """
    Configure a parser with field mapping and description
    Args:
        target_field: Field to update with parsed content (e.g., 'actions', 'request', etc.)
        description: Human-readable description of the parser
    """

    def decorator(func):
        func.target_field = target_field
        func.description = description
        return func

    return decorator


def get_parser_metadata(parser_id: Tuple[str, str, Optional[str]]) -> Dict[str, Any]:
    """
    Get metadata for a parser
    Args:
        parser_id: Tuple of (message_type, environment_type, action_space)
    Returns:
        Dict containing parser metadata (target_field, description)
    """
    if parser_id not in parsers:
        return {}

    parser_func = parsers[parser_id]
    metadata = {}
    if hasattr(parser_func, "target_field"):
        metadata["target_field"] = parser_func.target_field
    if hasattr(parser_func, "description"):
        metadata["description"] = parser_func.description
    return metadata


def process_content(
    content: Any,
    message_type: str,
    environment_type: str,
    action_space: Optional[str] = None,
) -> Tuple[bool, Optional[Any] | Exception]:
    """
    Process content based on message type, environment, and action space
    Args:
        content: The content to process (str for pyautogui/REQUEST/RESPONSE/STATUS/TASK, List[Dict] for computer13)
        message_type: The message type (e.g., 'EXECUTE', 'REQUEST', 'RESPONSE', 'STATUS', 'TASK')
        environment_type: The environment type (e.g., 'osworld')
        action_space: The action space (e.g., 'pyautogui', 'computer13') or None
    Returns:
        Tuple of (success: bool, parsed_content: Optional[Any] | Exception)
    """
    parser_key = (message_type, environment_type, action_space)

    try:
        parser = parsers[parser_key]
    except KeyError:
        parser_key = (message_type, environment_type, None)
        try:
            parser = parsers[parser_key]
        except KeyError:
            return False, ParseError(
                reason=f"No parser registered for message_type={message_type}, environment_type={environment_type}, action_space={action_space}"
            )

    return parser(content, environment_type, action_space)


# Parser functions
@parsers.register(("EXECUTE", "osworld", "pyautogui"))
@parser_config(
    target_field="actions",
    description="Parse and validate pyautogui actions for osworld environment",
)
def parse_osworld_pyautogui(
    content: str, environment_type: str, action_space: Optional[str]
) -> Tuple[bool, Optional[List[str]] | Exception]:
    """
    Parse and validate pyautogui code actions
    Args:
        content: The content string containing pyautogui code
        environment_type: The environment type (expected to be 'osworld')
        action_space: The action space (expected to be 'pyautogui')
    Returns:
        Tuple of (success: bool, actions: Optional[List[str]] | Exception)
    """
    if environment_type != "osworld" or action_space != "pyautogui":
        return False, ParseError(
            reason=f"Invalid context: expected environment_type='osworld', action_space='pyautogui', got {environment_type}, {action_space}"
        )

    if not isinstance(content, str) or not content.strip():
        return False, ParseError(
            reason="PyAutoGUI content must be a non-empty string", response=content
        )

    try:
        ast.parse(content)
    except SyntaxError as e:
        return False, ParseError(
            reason=f"Invalid Python syntax: {str(e)}", response=content
        )

    forbidden_functions = ["locateCenterOnScreen", "screenshot"]
    ast_tree = ast.parse(content)
    for node in ast.walk(ast_tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pyautogui"
            and node.func.attr in forbidden_functions
        ):
            # Changed from GameError to RuleViolationError
            return False, RuleViolationError(
                reason=f"Forbidden function '{node.func.attr}' used", response=content
            )

    return True, [content.strip()]


@parsers.register(("EXECUTE", "osworld", "computer13"))
@parser_config(
    target_field="actions",
    description="Parse and validate computer13 actions for osworld environment",
)
def parse_osworld_computer13(
    content: List[Dict[str, Any]], environment_type: str, action_space: Optional[str]
) -> Tuple[bool, Optional[List[Dict]] | Exception]:
    """
    Parse and validate computer13 actions as a list of dictionaries
    Args:
        content: The content as a list of action dictionaries
        environment_type: The environment type (expected to be 'osworld')
        action_space: The action space (expected to be 'computer13')
    Returns:
        Tuple of (success: bool, actions: Optional[List[Dict]] | Exception)
    """
    if environment_type != "osworld" or action_space != "computer13":
        return False, ParseError(
            reason=f"Invalid context: expected environment_type='osworld', action_space='computer13', got {environment_type}, {action_space}"
        )

    if (
        not isinstance(content, list)
        or not content
        or not all(isinstance(item, dict) for item in content)
    ):
        return False, ParseError(
            reason="Computer13 content must be a non-empty list of dictionaries",
            response=str(content),
        )

    for obj in content:
        if "action_type" not in obj:
            # Changed from GameError to RuleViolationError
            return False, RuleViolationError(
                reason="Missing 'action_type' in action", response=str(obj)
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
            # Changed from GameError to RuleViolationError
            return False, RuleViolationError(
                reason=f"Invalid 'action_type': {action_type}", response=str(obj)
            )

        param_specs = action_spec["parameters"]
        for param, spec in param_specs.items():
            if param not in obj and not spec["optional"]:
                # Changed from GameError to RuleViolationError
                return False, RuleViolationError(
                    reason=f"Missing required parameter '{param}' for '{action_type}'",
                    response=str(obj),
                )
            if param in obj:
                value = obj[param]
                expected_type = spec["type"]
                if not isinstance(value, expected_type):
                    # Changed from GameError to RuleViolationError
                    return False, RuleViolationError(
                        reason=f"Parameter '{param}' must be {expected_type.__name__}, got {type(value).__name__}",
                        response=str(obj),
                    )
                if spec["range"] is not None:
                    if isinstance(expected_type, (int, float)):
                        min_val, max_val = spec["range"]
                        if value < min_val or value > max_val:
                            # Changed from GameError to RuleViolationError
                            return False, RuleViolationError(
                                reason=f"Parameter '{param}' must be between {min_val} and {max_val}, got {value}",
                                response=str(obj),
                            )
                    elif isinstance(expected_type, str):
                        allowed = spec["range"]
                        if value not in allowed:
                            # Changed from GameError to RuleViolationError
                            return False, RuleViolationError(
                                reason=f"Parameter '{param}' must be one of {allowed}, got '{value}'",
                                response=str(obj),
                            )

        allowed_keys = set(param_specs.keys()) | {"action_type"}
        for key in obj:
            if key not in allowed_keys:
                # Changed from GameError to RuleViolationError
                return False, RuleViolationError(
                    reason=f"Unexpected parameter '{key}' for '{action_type}'",
                    response=str(obj),
                )

    return True, content


@parsers.register(("STATUS", "osworld", None))
@parser_config(
    target_field="actions",
    description="Parse and validate STATUS (DONE or FAIL) for osworld environment",
)
def parse_osworld_status(
    content: str, environment_type: str, action_space: Optional[str]
) -> Tuple[bool, Optional[List[str]] | Exception]:
    """
    Parse and validate STATUS content (DONE or FAIL)
    Args:
        content: The content string (should be 'DONE' or 'FAIL')
        environment_type: The environment type (expected to be 'osworld')
        action_space: The action space (ignored for STATUS)
    Returns:
        Tuple of (success: bool, status: Optional[List[str]] | Exception)
    """
    if environment_type != "osworld":
        return False, ParseError(
            reason=f"Invalid context: expected environment_type='osworld', got {environment_type}"
        )

    if not isinstance(content, str) or not content.strip():
        return False, ParseError(
            reason="STATUS content must be a non-empty string", response=content
        )

    status = content.strip()
    if status not in ["DONE", "FAIL"]:
        return False, ParseError(
            reason="Invalid status: must be DONE or FAIL", response=status
        )

    return True, [status]


@parsers.register(("REQUEST", "osworld", None))
@parser_config(
    target_field="request",
    description="Parse and validate REQUEST content for osworld environment",
)
def parse_osworld_request(
    content: str, environment_type: str, action_space: Optional[str]
) -> Tuple[bool, Optional[str] | Exception]:
    """
    Parse and validate REQUEST content
    Args:
        content: The content string
        environment_type: The environment type (expected to be 'osworld')
        action_space: The action space (ignored for REQUEST)
    Returns:
        Tuple of (success: bool, request: Optional[str] | Exception)
    """
    if environment_type != "osworld":
        return False, ParseError(
            reason=f"Invalid context: expected environment_type='osworld', got {environment_type}"
        )

    if not isinstance(content, str) or not content.strip():
        return False, ParseError(
            reason="REQUEST content must be a non-empty string", response=content
        )

    return True, content.strip()


@parsers.register(("RESPONSE", "osworld", None))
@parser_config(
    target_field="response",
    description="Parse and validate RESPONSE content for osworld environment",
)
def parse_osworld_response(
    content: str, environment_type: str, action_space: Optional[str]
) -> Tuple[bool, Optional[str] | Exception]:
    """
    Parse and validate RESPONSE content
    Args:
        content: The content string
        environment_type: The environment type (expected to be 'osworld')
        action_space: The action space (ignored for RESPONSE)
    Returns:
        Tuple of (success: bool, response: Optional[str] | Exception)
    """
    if environment_type != "osworld":
        return False, ParseError(
            reason=f"Invalid context: expected environment_type='osworld', got {environment_type}"
        )

    if not isinstance(content, str) or not content.strip():
        return False, ParseError(
            reason="RESPONSE content must be a non-empty string", response=content
        )

    return True, content.strip()
