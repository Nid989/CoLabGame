from typing import Callable, List, Dict, Any

from registry.base import FunctionRegistry


EditHistoryFuncType = Callable[[List[Dict], Any], List[Dict]]
edit_history_function_registry = FunctionRegistry[EditHistoryFuncType]
