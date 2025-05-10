import json
import re
from typing import Callable, Optional, Tuple, Dict, Any, List

from .base import Registry


# Function to store metadata about parsers
def parser_config(target_field: None, description=None):
    """Configure a parser with field mapping and description
    Args:
        target_field: field to update with parsed content (e.g. master.MessageState.__dataclass_fields__ like request, response, etc.)
        description: Human-readable description of the parser
    """

    def decorator(func):
        func.target_field = target_field
        func.description = description
        return func

    return decorator


def get_parser_metadata(parser_id: str) -> Dict[str, Any]:
    """Get metadata for a parser
    Args:
        parser_id: The identifier of the parser
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


parsers = Registry[Callable[[str], Tuple[bool, Optional[str]]]]()


# EXECUTE```python\n<content>```
@parsers.register("pyautogui_actions")
@parser_config(target_field="actions")
def parse_pyautogui_actions(utterance: str) -> Tuple[bool, Optional[List[str]]]:
    """Parse pyautogui code-actions from player utterances.
    Args:
        utterance: The text content to parse
    Returns:
        Tuple containing:
        - Boolean indicating if parsing was successful (code blocks or commands found)
        - List of extracted code actions, or None if no valid content was found
    """
    execute_pattern = r"EXECUTE\s*(.*?)(?=EXECUTE|\Z)"
    execute_matches = re.findall(execute_pattern, utterance, re.DOTALL)

    if not execute_matches:
        return False, None

    utterance = execute_matches[0]
    normalized_utterance = "\n".join(
        [line.strip() for line in utterance.split(";") if line.strip()]
    )

    code_block_pattern = r"```(?:\w+\s+)?(.*?)```"
    code_matches = re.findall(code_block_pattern, normalized_utterance, re.DOTALL)

    if not code_matches:
        return False, None

    parsed_content = [code_block.strip() for code_block in code_matches]

    return bool(parsed_content), parsed_content if parsed_content else None


# EXECUTE```json\n<content>```
@parsers.register("computer13_actions")
@parser_config(target_field="actions")
def parse_computer13_actions(utterance: str) -> Tuple[bool, Optional[List[dict]]]:
    """Parse computer13 json-actions from player utterances.
    Args:
        utterance: The text content to parse
    Returns:
        Tuple containing:
        - Boolean indicating if parsing was successful
        - List of parsed action dictionaries, or None if no valid content was found
    """
    execute_pattern = r"EXECUTE\s*(.*?)(?=EXECUTE|\Z)"
    execute_matches = re.findall(execute_pattern, utterance, re.DOTALL)

    if not execute_matches:
        return False, None

    utterance = execute_matches[0]
    json_blocks = re.findall(r"```(?:json\s+)?(.*?)```", utterance, re.DOTALL)

    if json_blocks:
        parsed_actions = []
        for json_text in json_blocks:
            try:
                action_dict = json.loads(json_text)
                parsed_actions.append(action_dict)
            except json.JSONDecodeError:
                continue

        if parsed_actions:
            return True, parsed_actions

    try:
        action_dict = json.loads(utterance)
        return True, [action_dict]
    except json.JSONDecodeError:
        return False, None


# TODO: revisit this later; it doesn't align with the current parsing approach.
@parsers.register("som_pyautogui_actions")
def parse_som_pyautogui_actions(utterance: str) -> Tuple[bool, Optional[str]]:
    """Parse pyautogui code-actions with screen object model (SOM) tags from player utterances.
    Usage:
        Should be used when 'action_space' is set to 'pyautogui'
        \ & 'observation_space' is set to 'SOM'
    Args:
        utterance: The text content to parse
    Returns:
        Tuple containing:
        - Boolean indicating if parsing was successful
        - Extracted code with SOM tags, or None if no valid content was found
    """
    # TODO: GameMaster should be provided but it somehow raise circular dependency error.
    gm = None  #
    masks = getattr(gm, "masks", [])
    if not masks:
        return False, None

    tag_vars = "\n".join(
        f"tag_{i + 1}=({int(x + w // 2)}, {int(y + h // 2)})"
        for i, (x, y, w, h) in enumerate(masks)
    )

    is_valid, extracted_code = parse_pyautogui_actions(utterance)

    if not is_valid:
        return False, None

    result = f"{tag_vars}\n{extracted_code}"
    return True, result


@parsers.register("done_or_fail")
@parser_config(target_field="actions")
def parse_done_or_fail(utterance: str) -> Tuple[bool, Optional[str]]:
    """Parse player utterances for status keywords (DONE, FAIL)
    Args:
        utterance: The text content to parse
    Returns:
        Tuple containing:
        - Boolean indicating if any status keyword was found
        - The matched keyword, or None if no keyword was found
    """
    status_pattern = r"STATUS\s*```(.*?)```"
    status_matches = re.findall(status_pattern, utterance, re.DOTALL)

    if not status_matches:
        return False, None

    block_content = status_matches[0].strip()
    status_keywords = ["DONE", "FAIL"]
    if block_content in status_keywords:
        return True, [block_content]

    return False, None


# REQUEST```<content>```
@parsers.register("request")
@parser_config(target_field="request")
def parse_request(utterance: str) -> Tuple[bool, Optional[str]]:
    """Parse player utterances for REQUEST blocks.
    Args:
        utterance: The text content to parse
    Returns:
        Tuple containing:
        - Boolean indicating if a REQUEST block was found
        - The content inside the REQUEST block, or None if no valid content was found
    """
    request_pattern = r"REQUEST\s*```(.*?)```"
    request_matches = re.findall(request_pattern, utterance, re.DOTALL)

    if not request_matches:
        return False, None

    request_content = request_matches[0].strip()
    return bool(request_content), request_content if request_content else None


# RESPONSE```<content>```
@parsers.register("response")
@parser_config(target_field="response")
def parse_response(utterance: str) -> Tuple[bool, Optional[str]]:
    """Parse player utterances for RESPONSE blocks.
    Args:
        utterance: The text content to parse
    Returns:
        Tuple containing:
        - Boolean indicating if a RESPONSE block was found
        - The content inside the RESPONSE block, or None if no valid content was found
    """
    response_pattern = r"RESPONSE\s*```(.*?)```"
    response_matches = re.findall(response_pattern, utterance, re.DOTALL)

    if not response_matches:
        return False, None

    response_content = response_matches[0].strip()
    return bool(response_content), response_content if response_content else None


# TASK```<content>```
@parsers.register("task")
@parser_config(target_field="task")
def parse_task(utterance: str) -> Tuple[bool, Optional[str]]:
    """Parse player utterances for TASK blocks.
    Args:
        utterance: The text content to parse
    Returns:
        Tuple containing:
        - Boolean indicating if a TASK block was found
        - The content inside the TASK block, or None if no valid content was found
    """
    task_pattern = r"TASK\s*```(.*?)```"
    task_matches = re.findall(task_pattern, utterance, re.DOTALL)

    if not task_matches:
        return False, None

    task_content = task_matches[0].strip()
    return bool(task_content), task_content if task_content else None
