"""
Task generation package for computergame.

Provides task sampling, generation, and category management functionality.
"""

from .core.task_sampling import TaskSamplingInterface, GenerationSession, TaskPackage

__all__ = ["TaskSamplingInterface", "GenerationSession", "TaskPackage"]
