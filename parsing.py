import inspect
import json
import re
from typing import Callable, Dict, Generic, Optional, Tuple, TypeVar

from clemcore.clemgame import Player, DialogicNetworkGameMaster

T = TypeVar("T")


class FunctionRegistry(Generic[T]):
    """Registry for storing and retrieving functions by ID.
    Allows registering functions with unique IDs and retrieving them later.
    Can be used as a decorator or called directly.
    """

    def __init__(self):
        self._registry: Dict[str, T] = {}

    def register(self, func_id: str = None) -> Callable[[T], T]:
        """Register a function with the given ID.
        Can be used as a decorator:
            @registry.register("my_func")
            def my_func(): ...
        Or called directly:
            registry.register("my_func")(my_func)
        Args:
            func_id: Unique identifier for the function. If None, uses function name.
        Returns:
            Decorator function that registers the decorated function.
        """

        def decorator(func: T) -> T:
            nonlocal func_id
            if func_id is None:
                func_id = func.__name__

            if func_id in self._registry:
                raise ValueError(f"Function with ID '{func_id}' already registered")

            self._registry[func_id] = func
            return func

        return decorator

    def __getitem__(self, func_id: str) -> T:
        """Get a function by its ID.
        Args:
            func_id: ID of the function to retrieve.
        Returns:
            The registered function.
        Raises:
            KeyError: If function with given ID is not registered.
        """
        if func_id not in self._registry:
            raise KeyError(f"Function with ID '{func_id}' not registered")
        return self._registry[func_id]

    def __contains__(self, func_id: str) -> bool:
        """Check if a function with the given ID is registered.
        Args:
            func_id: ID to check.
        Returns:
            True if function is registered, False otherwise.
        """
        return func_id in self._registry

    def list_functions(self) -> Dict[str, str]:
        """List all registered functions with their signatures.
        Returns:
            Dictionary mapping function IDs to their signatures.
        """
        return {
            func_id: str(inspect.signature(func))
            for func_id, func in self._registry.items()
        }


# Special registry for parse functions.
ParseFuncType = Callable[
    [Player, str, "DialogicNetworkGameMaster"], Tuple[bool, Optional[str]]
]
parse_function_registry = FunctionRegistry[ParseFuncType]()


# EXECUTE```python\n<content>```
@parse_function_registry.register("parse_pyautogui_actions")
def parse_pyautogui_actions(
    player: Player, utterance: str, gm: "DialogicNetworkGameMaster"
) -> Tuple[bool, Optional[str]]:
    """Parse pyautogui code-actions from player utterances.
    Usage:
        Should be used when 'action_space' is set to 'pyautogui'.
    Args:
        player: The player who produced the utterance
        utterance: The text content to parse
        gm: The game master instance
    Returns:
        Tuple containing:
        - Boolean indicating if parsing was successful (code blocks or commands found)
        - Extracted content as a string, or None if no valid content was found
    """
    execute_pattern = r"EXECUTE\s*(.*?)(?=EXECUTE|\Z)"
    execute_matches = re.findall(execute_pattern, utterance, re.DOTALL)

    if execute_matches:
        utterance = execute_matches[0]
    else:
        return False, None

    normalized_utterance = "\n".join(
        [line.strip() for line in utterance.split(";") if line.strip()]
    )

    code_block_pattern = r"```(?:\w+\s+)?(.*?)```"
    code_matches = re.findall(code_block_pattern, normalized_utterance, re.DOTALL)

    if not code_matches:
        return False, None

    parsed_content = "\n".join(code_block.strip() for code_block in code_matches)

    return bool(parsed_content), parsed_content if parsed_content else None


# EXECUTE```json\n<content>```
@parse_function_registry.register("parse_computer13_actions")
def parse_computer13_actions(
    player: Player, utterance: str, gm: "DialogicNetworkGameMaster"
) -> Tuple[bool, Optional[str]]:
    """Parse computer13 json-actions from player utterances.
    Usage:
        Should be used when 'action_space' is set to 'computer13'
    Args:
        player: The player who produced the utterance
        utterance: The text content to parse
        gm: The game master instance
    Returns:
        Tuple containing:
        - Boolean indicating if parsing was successful
        - Extracted JSON content as a string, or None if no valid content was found
    """
    execute_pattern = r"EXECUTE\s*(.*?)(?=EXECUTE|\Z)"
    execute_matches = re.findall(execute_pattern, utterance, re.DOTALL)

    if execute_matches:
        utterance = execute_matches[0]
    else:
        return False, None

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
            return True, json.dumps(parsed_actions)

    try:
        action_dict = json.loads(utterance)
        return True, json.dumps([action_dict])
    except json.JSONDecodeError:
        return False, None


# TODO: revisit this later; it doesn't align with the current parsing approach.
@parse_function_registry.register("parse_som_pyautogui_actions")
def parse_som_pyautogui_actions(
    player: Player, utterance: str, gm: "DialogicNetworkGameMaster"
) -> Tuple[bool, Optional[str]]:
    """Parse pyautogui code-actions with screen object model (SOM) tags from player utterances.
    Usage:
        Should be used when 'action_space' is set to 'pyautogui'
        \ & 'observation_space' is set to 'SOM'
    Args:
        player: The player who produced the utterance
        utterance: The text content to parse
        gm: The game master instance
    Returns:
        Tuple containing:
        - Boolean indicating if parsing was successful
        - Extracted code with SOM tags, or None if no valid content was found
    """
    masks = getattr(gm, "masks", [])
    if not masks:
        return False, None

    tag_vars = "\n".join(
        f"tag_{i + 1}=({int(x + w // 2)}, {int(y + h // 2)})"
        for i, (x, y, w, h) in enumerate(masks)
    )

    is_valid, extracted_code = parse_pyautogui_actions(player, utterance, gm)

    if not is_valid:
        return False, None

    result = f"{tag_vars}\n{extracted_code}"
    return True, result


@parse_function_registry.register("parse_done_or_fail")
def parse_done_or_fail(
    player: Player, utterance: str, gm: "DialogicNetworkGameMaster"
) -> Tuple[bool, Optional[str]]:
    """Parse player utterances for status keywords (DONE, FAIL)
    Args:
        player: The player who produced the utterance
        utterance: The text content to parse
        gm: The game master instance
    Returns:
        Tuple containing:
        - Boolean indicating if any status keyword was found
        - The matched keyword, or None if no keyword was found
    """
    status_keywords = ["DONE", "FAIL"]

    for keyword in status_keywords:
        if re.search(r"\b" + keyword + r"\b", utterance):
            return True, keyword


# QUERY```<content>```
@parse_function_registry.register("parse_query")
def parse_query(
    player: Player, utterance: str, gm: "DialogicNetworkGameMaster"
) -> Tuple[bool, Optional[str]]:
    """Parse player utterances for QUERY blocks.
    Usage:
        Should be used to extract content from QUERY```<content>``` blocks.
    Args:
        player: The player who produced the utterance
        utterance: The text content to parse
        gm: The game master instance
    Returns:
        Tuple containing:
        - Boolean indicating if a QUERY block was found
        - The content inside the QUERY block, or None if no valid content was found
    """
    query_pattern = r"QUERY\s*```(.*?)```"
    query_matches = re.findall(query_pattern, utterance, re.DOTALL)

    if not query_matches:
        return False, None

    query_content = query_matches[0].strip()
    return bool(query_content), query_content if query_content else None


# RESPONSE```<content>```
@parse_function_registry.register("parse_response")
def parse_response(
    player: Player, utterance: str, gm: "DialogicNetworkGameMaster"
) -> Tuple[bool, Optional[str]]:
    """Parse player utterances for RESPONSE blocks.
    Usage:
        Should be used to extract content from RESPONSE```<content>``` blocks.
    Args:
        player: The player who produced the utterance
        utterance: The text content to parse
        gm: The game master instance
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
