from typing import Callable, Dict, Any

from .base import Registry
from ...utils.osworld_utils import preprocess_observation


processors = Registry[Callable]()


@processors.register("observation")
def process_observation(observation: Dict[str, Any], handler) -> Dict[str, Any]:
    """Process raw observation data into a standardized format
    Args:
        observation: Raw observation data
        handler: Reference to the prompt handler instance
    Returns:
        Processed observation data
    """
    processed_obs = {}
    observation_type = getattr(handler, "observation_type", "a11y_tree")
    platform = getattr(handler, "platform", "ubuntu")
    a11y_tree_max_tokens = getattr(handler, "a11y_tree_max_tokens", None)
    temporary_image_manager = getattr(handler, "temporary_image_manager", None)
    preprocessed_obs = preprocess_observation(
        observation=observation,
        observation_type=observation_type,
        platform=platform,
        a11y_tree_max_tokens=a11y_tree_max_tokens,
    )
    processed_obs.update(preprocessed_obs)
    if temporary_image_manager and "screenshot" in preprocessed_obs:
        screenshot = preprocessed_obs["screenshot"]
        if isinstance(screenshot, bytes):
            image_path = temporary_image_manager.save_image(screenshot)
            processed_obs["screenshot"] = image_path
        else:
            raise ValueError(
                "Expected 'screenshot' to be bytes, but got {}".format(
                    type(screenshot).__name__
                )
            )
    return processed_obs


@processors.register("query")
def process_query(query: str, handler) -> str:
    """Process a query string
    Args:
        query: Raw query string
        handler: Reference to the prompt handler instance
    Returns:
        Processed query string
    """
    return query


@processors.register("response")
def process_response(response: str, handler) -> str:
    """Process a response string
    Args:
        response: Raw response string
        handler: Reference to the prompt handler instance
    Returns:
        Processed response string
    """
    return response


@processors.register("plan")
def process_plan(plan: str, handler) -> str:
    """Process a plan string
    Args:
        plan: Raw plan string
        handler: Reference to the prompt handler instance
    Returns:
        Processed plan string
    """
    return plan


@processors.register("task")
def process_task(task: str, handler) -> str:
    """Process a task string
    Args:
        task: Raw task string
        handler: Reference to the prompt handler instance
    Returns:
        Processed task string
    """
    return task


@processors.register("additional")
def process_additional(additional: Dict[str, str], handler) -> Dict[str, str]:
    """Process additional tagged content
    Args:
        additional: Dictionary containing tag and content pairs
        handler: Reference to the prompt handler instance
    Returns:
        Processed additional content dictionary
    """
    return additional
