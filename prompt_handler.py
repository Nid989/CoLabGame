from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Any, Union, Protocol, Literal
from PIL import Image

from registry import FunctionRegistry
from constants import HANDLER_TYPE


processor_registry = FunctionRegistry[Callable]()


@dataclass
class MessageEntry:
    """Container for different types of message content"""

    observation: Optional[Dict[str, Union[str, Image.Image, Dict]]] = None
    query: Optional[str] = None
    response: Optional[str] = None
    plan: Optional[str] = None
    task: Optional[str] = None
    # Additional fields can be added upon requirement.


class ComponentHandler(Protocol):
    """Protocol for component handlers"""

    def __call__(self, value: Any, role: str) -> str: ...


class MessageFormatter:
    """Formatter for message entries with component-specific handlers"""

    def __init__(self):
        self.handlers = Dict[str, ComponentHandler] = {}

    def register_handler(self, component_name: str, handler: ComponentHandler) -> None:
        """Register a handler for a specific component type"""
        self.handlers[component_name] = handler

    def format(self, entry: MessageEntry, role: str) -> str:
        """Format a message entry using registered handlers"""
        parts = []

        for field_name, field_value in entry.__dict__.items():
            if field_value is None:
                continue

            if field_name in self.handlers:
                handler = self.handlers[field_name]
                formatted_component = handler(field_value, role)
                parts.append(formatted_component)
            else:
                # Default formatting if no handler exists
                parts.append(f"{field_name.capitalize()}: {str(field_value)}")

        return "\n".join(parts)


class PromptHandler:
    """Prompt-management for `RoleBasedPlayer` instance with
    handler type-specific implementations"""

    @staticmethod
    def create_add_user_message(handler_type: HANDLER_TYPE) -> Callable:
        """Factory method to create the appropriate add_user_message implementation.
        Args:
            handler_type: Type of handler to create implementation for
        Returns:
            Callable: The appropriate add_user_message implementation
        """
        if handler_type == "standard":

            def add_user_message(self, **kwargs) -> None:
                """Handle a standard message (typically ~observation)
                Args:
                    **kwargs: Message components (query, response, etc.)
                """
                entry = MessageEntry(**kwargs)
                processed_entry = self._process_entry(entry)
                formatted_message = self.formatter.format(processed_entry, self.role)
                self.add_message(content=formatted_message, role="user")
                self.raw_entries.append(entry)

            return add_user_message

        elif handler_type == "environment":

            def add_user_message(self, **kwargs) -> None:
                """Handle an environment message (with observations)
                Args:
                    **kwargs: Message components (observation, response, etc.)
                """
                entry = MessageEntry(**kwargs)
                processed_entry = self._process_entry(entry)
                formatted_message = self.formatter.format(processed_entry, self.role)
                self.add_message(content=formatted_message, role="user")
                self.raw_entries.append(entry)

                # For environment handlers, also store raw observations separately
                if hasattr(entry, "observation") and entry.observation:
                    self.observations.append(entry.observation)

            return add_user_message

        raise ValueError(f"Invalid handler type: {handler_type}")

    # Registry mapping handler types to factor method calls
    REGISTRY = {
        "standard": {"add_user_message": create_add_user_message.__func__("standard")},
        "environment": {
            "add_user_message": create_add_user_message.__func__("environment")
        },
    }

    def __init__(
        self,
        handler_type: HANDLER_TYPE = "standard",
        prompt_header: str = None,
        role: str = None,
        **kwargs,
    ):
        """Initialize the prompt handler
        Args:
            handler_type: Type of handler to use, either `standard` or `environment`
            role: Role of the player using this handler
            **kwargs: Additional configuration parameters
        """
        self.handler_type = handler_type
        self.role = role or handler_type
        self.history: List[Dict[str, str]] = []
        self.raw_entries: List[MessageEntry] = []
        if self.handler_type == "environment":
            self.observations = []
            for key, value in kwargs.items():
                setattr(self, key, value)
        if prompt_header:
            self.add_message(content=prompt_header, role="user")

        self.formatter = MessageFormatter()
        self._configure_formatter()
        self._bind_implementations(self.handler_type)

    def add_assistant_message(self, content: str) -> None:
        """Add a message with 'assistant' role to conversation history.
        Args:
            content: The text content of the message to be added
        """
        self.add_message(content=content, role="assistant")

    def add_message(
        self,
        content: str,
        role: Literal["user", "assistant"],
        image: Optional[List[str]] = None,
    ) -> None:
        """Add a message to the conversation history.
        Args:
            content: The text content of the message
            role: The role of the message sender (user or assistant)
            image: Optional list of image data or paths to include with the message.
        """
        message = {"role": role, "content": content}
        if image and len(image) > 0:
            message["image"] = image
        self.messages.append(message)

    def clear_history(self) -> None:
        """Clear all conversation history and observations."""
        self.history = []
        if self.handler_type == "environment":
            self.observations = []

    def get_messages(self) -> List[Dict]:
        """Get the current message history.
        Returns:
            List of message dictionaries
        """
        return self.history

    def _bind_implementations(self, handler_type: HANDLER_TYPE) -> None:
        """Bind the appropriate method implementations to this instance.
        Args:
            handler_type: The type of handler to bind implementations from
        """
        if handler_type not in self.REGISTRY:
            raise ValueError(f"Unknown handler type: {handler_type}")
        for method_name, implementation in self.REGISTRY[handler_type].items():
            # Bind the method to this instance
            bound_method = implementation.__get__(self, self.__class__)
            setattr(self, method_name, bound_method)

    def _configure_formatter(self) -> None:
        """Configure the message formatter with component-specific handlers"""
        self.formatter.register_handler("observation", self._format_observation)
        self.formatter.register_handler("query", self._format_query)
        self.formatter.register_handler("response", self._format_response)
        self.formatter.register_handler("plan", self._format_plan)
        self.formatter.register_handler("task", self._format_task)
        # Other handlers can be registered upon requirement

    def _process_entry(self, entry: MessageEntry) -> MessageEntry:
        """Process each component in the entry using registered processors
        Args:
            entry: Original message entry
        Returns:
            MessageEntry: New entry with processed component
        """
        processed_entry = MessageEntry()
        for field_name, field_value in entry.__dict__.items():
            if field_value is None:
                continue
            processor_id = f"process_{field_name}"
            if processor_id in processor_registry:
                try:
                    processor = processor_registry[processor_id]
                    processed_value = processor(field_value, self.role)
                    setattr(processed_entry, field_name, processed_value)
                except Exception:
                    # NOTE: not sure about this.
                    setattr(processed_entry, field_name, field_value)
            else:
                # NOTE: not sure about this.
                setattr(processed_entry, field_name, field_value)

        return processed_entry

    def _format_observation(self, observation: Dict, role: str) -> str:
        """Format an observation component"""
        if isinstance(observation, dict):
            parts = []
            if "screenshot" in observation:
                parts.append(
                    f"Screenshot: {observation['screenshot']}"
                )  # NOTE: completely wrong.
            if "DOM" in observation:
                dom_str = str(observation["DOM"])
                dom_summary = dom_str[:50] + "..." if len(dom_str) > 50 else dom_str
                parts.append(f"DOM: {dom_summary}")
            return f"Observation: {' | '.join(parts)}"
        return f"Observation: {str(observation)}"

    def _format_query(self, query: str, role: str) -> str:
        """Format a query component"""
        return f"Query: {query}"

    def _format_response(self, response: str, role: str) -> str:
        """Format a response component"""
        return f"Response: {response}"

    def _format_plan(self, plan: str, role: str) -> str:
        """Format a plan component"""
        return f"Plan: {plan}"

    def _format_task(self, task: str, role: str) -> str:
        """Format a task component"""
        return f"Task: {task}"

    @classmethod
    def register_handler_type(
        cls, handler_type: str, implementations: Dict[str, Callable]
    ) -> None:
        """Register a new handler type with implementations.
        Args:
            handler_type: Name of the handler type
            implementations: Dictionary mapping method names to implementation functions
        """
        cls.REGISTRY[handler_type] = implementations
