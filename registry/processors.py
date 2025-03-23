from typing import Callable, Dict, Any
from .base import Registry

processors = Registry[Callable]()


@processors.register("observation")
def process_observation(observation: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Process raw observation data into a standardized format

    Args:
        observation: Raw observation data
        role: Role of the agent

    Returns:
        Processed observation data
    """
    # In a real implementation, you might:
    # - Extract key information from DOM
    # - Process screenshots
    # - Convert formats
    # - Summarize large data

    # For now, just return the original data
    return observation


@processors.register("query")
def process_query(query: str, role: str) -> str:
    """Process a query string

    Args:
        query: Raw query string
        role: Role of the agent

    Returns:
        Processed query string
    """
    return query


@processors.register("response")
def process_response(response: str, role: str) -> str:
    """Process a response string

    Args:
        response: Raw response string
        role: Role of the agent

    Returns:
        Processed response string
    """
    return response


@processors.register("plan")
def process_plan(plan: str, role: str) -> str:
    """Process a plan string

    Args:
        plan: Raw plan string
        role: Role of the agent

    Returns:
        Processed plan string
    """
    return plan


@processors.register("task")
def process_task(task: str, role: str) -> str:
    """Process a task string

    Args:
        task: Raw task string
        role: Role of the agent

    Returns:
        Processed task string
    """
    return task
