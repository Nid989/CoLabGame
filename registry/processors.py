from typing import Callable, Dict, Any
from .base import Registry

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
    # In a real implementation, you might:
    # - Extract key information from DOM
    # - Process screenshots
    # - Convert formats
    # - Summarize large data
    # observation_type = getattr(handler, 'observation_type', None)
    # For now, just return the original data
    return observation


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
