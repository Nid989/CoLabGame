"""
Framework Task Generator Package

This package converts task instances from test_examples.json format
to framework-compatible configurations for VS Code evaluation.
"""

from .generator import FrameworkTaskGenerator
from .file_manager import FileManager
from .config_builder import ConfigBuilder

__version__ = "1.0.0"
__author__ = "Task Generator"

__all__ = ["FrameworkTaskGenerator", "FileManager", "ConfigBuilder"]
