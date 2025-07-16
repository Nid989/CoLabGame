from typing import Callable, Dict, Any

from .base import Registry
from ..osworld import preprocess_observation


processors = Registry[Callable]()


@processors.register("observation")
def process_observation(observation: Dict[str, Any], game_config: Dict) -> Dict[str, Any]:
    """Process raw observation data into a standardized format
    Args:
        observation: Raw observation data
        game_config: Dictionary containing game configuration
    Returns:
        Processed observation data
    """
    # Check if screenshot exists and is a string (local path)
    if "screenshot" in observation and not isinstance(observation["screenshot"], str):
        raise ValueError(f"Expected 'screenshot' to be a string (local path), but got {type(observation['screenshot']).__name__}")

    processed_obs = {}
    observation_type = game_config.get("observation_type", "a11y_tree")
    platform = game_config.get("platform", "ubuntu")
    a11y_tree_max_tokens = game_config.get("a11y_tree_max_tokens", None)

    preprocessed_obs = preprocess_observation(
        observation=observation,
        observation_type=observation_type,
        platform=platform,
        a11y_tree_max_tokens=a11y_tree_max_tokens,
    )
    processed_obs.update(preprocessed_obs)

    return processed_obs


@processors.register("request")
def process_request(request: str, game_config: Dict) -> str:
    """Process a request string
    Args:
        request: Raw request string
        game_config: Dictionary containing game configuration
    Returns:
        Processed request string
    """
    return request


@processors.register("response")
def process_response(response: str, game_config: Dict) -> str:
    """Process a response string
    Args:
        response: Raw response string
        game_config: Dictionary containing game configuration
    Returns:
        Processed response string
    """
    return response


@processors.register("plan")
def process_plan(plan: str, game_config: Dict) -> str:
    """Process a plan string
    Args:
        plan: Raw plan string
        game_config: Dictionary containing game configuration
    Returns:
        Processed plan string
    """
    return plan


@processors.register("task")
def process_task(task: str, game_config: Dict) -> str:
    """Process a task string
    Args:
        task: Raw task string
        game_config: Dictionary containing game configuration
    Returns:
        Processed task string
    """
    return task


@processors.register("additional")
def process_additional(additional: Dict[str, str], game_config: Dict) -> Dict[str, str]:
    """Process additional tagged content
    Args:
        additional: Dictionary containing tag and content pairs
        game_config: Dictionary containing game configuration
    Returns:
        Processed additional content dictionary
    """
    return additional
