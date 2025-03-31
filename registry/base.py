import inspect
from typing import Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
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

    def __iter__(self):
        """Makes the registry iterable, yielding function IDs.
        Returns:
            Iterator over function IDs in the registry.
        """
        return iter(self._registry)
