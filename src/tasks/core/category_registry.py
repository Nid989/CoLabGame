import os
import importlib
from typing import Dict, List, Optional
from src.tasks.categories.base import BaseCategoryInterface


class CategoryRegistry:
    """Registry for managing available categories with automatic discovery."""

    def __init__(self):
        self._categories: Dict[str, BaseCategoryInterface] = {}
        self._discovered = False

    def discover_categories(self):
        """Automatically discover all available categories."""
        if self._discovered:
            return

        # Use an absolute path to find the 'categories' directory, making discovery
        # independent of the current working directory.
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        categories_dir = os.path.join(current_file_dir, "..", "categories")

        if not os.path.isdir(categories_dir):
            return

        # Look for category directories (excluding base)
        for item in os.listdir(categories_dir):
            item_path = os.path.join(categories_dir, item)
            if os.path.isdir(item_path) and not item.startswith("_") and item != "base" and item != "__pycache__":
                self._try_load_category(item)

        self._discovered = True

    def _try_load_category(self, category_name: str):
        """Try to load a category from its directory."""
        try:
            # Try to import the category module
            module = importlib.import_module(f"src.tasks.categories.{category_name}.category")

            # Look for a class that implements BaseCategoryInterface (either directly or through BaseCategory)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseCategoryInterface)
                    and attr != BaseCategoryInterface
                    and not attr.__name__.startswith("Base")
                ):
                    # Instantiate and register the category
                    category_instance = attr()
                    self._categories[category_name] = category_instance
                    print(f"Discovered category: {category_name}")
                    break
        except Exception as e:
            print(f"Failed to load category {category_name}: {e}")

    def register_category(self, category: BaseCategoryInterface):
        """Manually register a category."""
        category_name = category.get_category_name()
        self._categories[category_name] = category

    def get_category(self, category_name: str) -> Optional[BaseCategoryInterface]:
        """Get a category by name."""
        self.discover_categories()
        return self._categories.get(category_name)

    def get_all_categories(self) -> Dict[str, BaseCategoryInterface]:
        """Get all registered categories."""
        self.discover_categories()
        return self._categories.copy()

    def get_category_names(self) -> List[str]:
        """Get names of all available categories."""
        self.discover_categories()
        return list(self._categories.keys())

    def get_task_types_for_category(self, category_name: str, level: Optional[int] = None) -> Dict[str, type]:
        """Get task types for a specific category."""
        category = self.get_category(category_name)
        if category:
            return category.get_task_types(level)
        return {}

    def get_supported_levels_for_category(self, category_name: str) -> List[int]:
        """Get supported levels for a specific category."""
        category = self.get_category(category_name)
        if category:
            return category.get_supported_levels()
        return []


# Global registry instance
category_registry = CategoryRegistry()
