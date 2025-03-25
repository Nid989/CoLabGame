from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Any, Union, Protocol, Literal
from PIL import Image
import copy

from registry import processors
from constants import HANDLER_TYPE, OBSERVATION_TYPE_values


@dataclass
class MessageEntry:
    """Container for different types of message content"""

    observation: Optional[Dict[str, Union[str, Image.Image, Dict]]] = None
    query: Optional[str] = None
    response: Optional[str] = None
    plan: Optional[str] = None
    task: Optional[str] = None
    # Additional fields can be added upon requirement.

    def __post_init__(self):
        if not any(v is not None for v in self.__dict__.values()):
            raise ValueError("At least one entry must be provided")

    @classmethod
    def for_handler(
        cls,
        handler_type: HANDLER_TYPE = "standard",
        valid_entries: set = None,
        **kwargs,
    ) -> "MessageEntry":
        """Creates a validated MessageEntry instance based on handler type and valid entries.
        Args:
            handler_type: Type of handler ('standard' or 'environment')
            valid_entries: Set of allowed entry types
            **kwargs: Message entry components
        Returns:
            MessageEntry: Validated instance with filtered entries
        Raises:
            ValueError: If invalid entries provided or no valid entries remain after filtering
        """
        handler_rules = {
            "standard": {"query", "response", "plan", "task"},
            "environment": {"observation", "query", "response", "plan", "task"},
        }
        if not valid_entries or not kwargs:
            raise ValueError("Both valid_entries and message components are required")
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        if invalid_fields := valid_entries - valid_fields:
            raise ValueError(
                f"Invalid entries in valid_entries: {invalid_fields}. "
                f"Must be one of: {valid_fields}"
            )
        allowed_entries = handler_rules.get(handler_type, set()) & valid_entries
        valid_components = {
            k: v for k, v in kwargs.items() if k in allowed_entries and v is not None
        }
        if invalid_entries := set(kwargs) - allowed_entries:
            raise ValueError(
                f"Invalid entries for {handler_type} handler: {invalid_entries}"
            )
        if not valid_components:
            raise ValueError(f"No valid entries provided for {handler_type} handler")

        return cls(**valid_components)


class ComponentHandler(Protocol):
    """Protocol for component handlers"""

    def __call__(self, value: Any) -> str: ...


class MessageFormatter:
    """Formatter for message entries with component-specific handlers"""

    def __init__(self):
        self.handlers: Dict[str, ComponentHandler] = {}

    def register_handler(self, component_name: str, handler: ComponentHandler) -> None:
        """Register a handler for a specific component type"""
        self.handlers[component_name] = handler

    def format(self, entry: MessageEntry) -> str:
        """Format a message entry using registered handlers
        Returns:
            Dict with 'content' key containing formatted text and optional 'image' key with image paths"""
        parts = []
        image_paths = []
        for field_name, field_value in entry.__dict__.items():
            if field_value is None:
                continue
            if field_name in self.handlers:
                handler = self.handlers[field_name]
                formatted_component = handler(field_value)
                if isinstance(formatted_component, dict):
                    parts.append(formatted_component.get("content", ""))
                    if "image" in formatted_component and formatted_component["image"]:
                        if isinstance(formatted_component["image"], list):
                            image_paths.extend(formatted_component["image"])
                        else:
                            image_paths.append(formatted_component["image"])
                else:
                    parts.append(formatted_component)
            else:
                parts.append(f"{field_name.capitalize()}: {str(field_value)}")

        return {
            "content": "\n".join(parts),
            "image": image_paths if image_paths else None,
        }


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
                entry = MessageEntry.for_handler(
                    handler_type, set(self.valid_entries), **kwargs
                )
                processed_entry = self._process_entry(entry)
                formatted_message = self.formatter.format(processed_entry)
                content = formatted_message["content"]
                if self.prompt_footer:
                    content += f"\n\n{self.prompt_footer}"
                self.add_message(
                    content=content,
                    role="user",
                    image=formatted_message.get("image"),
                )
                self.raw_entries.append(entry)

            return add_user_message

        elif handler_type == "environment":

            def add_user_message(self, **kwargs) -> None:
                """Handle an environment message (with observations)
                Args:
                    **kwargs: Message components (observation, response, etc.)
                """
                entry = MessageEntry.for_handler(
                    handler_type, set(self.valid_entries), **kwargs
                )
                processed_entry = self._process_entry(entry)
                formatted_message = self.formatter.format(processed_entry)
                content = formatted_message["content"]
                if self.prompt_footer:
                    content += f"\n\n{self.prompt_footer}"
                self.add_message(
                    content=content,
                    role="user",
                    image=formatted_message.get("image"),
                )
                self.raw_entries.append(entry)
                # For environment handlers, also store raw observations separately
                if hasattr(entry, "observation") and entry.observation:
                    self.observations.append(entry.observation)

            return add_user_message

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
        prompt_footer: str = None,
        valid_entries: List[str] = None,
        **kwargs,
    ):
        """Initialize the prompt handler
        Args:
            handler_type: Type of handler to use, either `standard` or `environment`
            prompt_header: Initial prompt to add to history (Usually indicates player-related instructions)
            **kwargs: Additional configuration parameters
        """
        self.handler_type = handler_type
        self.prompt_header = prompt_header
        self.prompt_footer = prompt_footer
        self.valid_entries = valid_entries
        self.history: List[Dict[str, str]] = []
        self.raw_entries: List[MessageEntry] = []
        if self.handler_type == "environment":
            self.observations = []
            for key, value in kwargs.items():
                setattr(self, key, value)
        if self.prompt_header:
            self.add_message(content=self.prompt_header, role="user")

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
        self.history.append(message)

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
        # Additional handlers can be registered upon requirement

    def _process_entry(self, entry: MessageEntry) -> MessageEntry:
        """Process each component using registered processors with cls reference
        Args:
            entry: Original message entry
        Returns:
            MessageEntry: New entry with processed component
        """
        processed_entry = {}
        for field_name, field_value in entry.__dict__.items():
            if field_value is None:
                continue
            if field_name in processors:
                try:
                    processor = processors[field_name]
                    processed_value = processor(field_value, self)
                    if processed_value is not None:
                        processed_entry[field_name] = processed_value
                except Exception as e:
                    raise Exception(f"Failed to process field '{field_name}': {str(e)}")
            else:
                raise ValueError(f"No processor registered for field '{field_name}'")

        return MessageEntry(**processed_entry)

    def _format_observation(self, observation: Dict) -> str:
        """Format an observation component"""
        result = ["## Observation"]
        image_paths = []
        if self.observation_type == "screenshot":
            result.append("### Screenshot")
            if "screenshot" in observation and isinstance(
                observation["screenshot"], str
            ):
                image_paths.append(observation["screenshot"])
        elif self.observation_type == "a11y_tree":
            result.append(
                "### Accessibility Tree\n```\n{}\n```".format(
                    observation.get("accessibility_tree", "")
                )
            )
        elif self.observation_type == "screenshot_a11y_tree":
            result.append("### Screenshot")
            if "screenshot" in observation and isinstance(
                observation["screenshot"], str
            ):
                image_paths.append(observation["screenshot"])
            result.append(
                "### Accessibility Tree\n```\n{}\n```".format(
                    observation.get("accessibility_tree", "")
                )
            )
        elif self.observation_type == "som":
            result.append("### Tagged Screenshot")
            if "screenshot" in observation and isinstance(
                observation["screenshot"], str
            ):
                image_paths.append(observation["screenshot"])
            result.append(
                "### Accessibility Tree\n```\n{}\n```".format(
                    observation.get("accessibility_tree", "")
                )
            )
        else:
            raise ValueError(
                f"Invalid observation_type: {self.observation_type}. Expected one of [{OBSERVATION_TYPE_values}]"
            )

        return {
            "content": "\n\n".join(result),
            "image": image_paths if image_paths else None,
        }

    def _format_query(self, query: str) -> Dict[str, str]:
        """Format a query component"""
        return {"content": f"Query: {query}"}

    def _format_response(self, response: str) -> Dict[str, str]:
        """Format a response component"""
        return {"content": f"Response: {response}"}

    def _format_plan(self, plan: str) -> Dict[str, str]:
        """Format a plan component"""
        return {"content": f"Plan: {plan}"}

    def _format_task(self, task: str) -> Dict[str, str]:
        """Format a task component"""
        return {"content": f"Task: {task}"}

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

    def get_pruned_messages(self, k: int = 3) -> List[Dict]:
        """Get a pruned version of the message history, keeping the last k user messages
        and all assistant messages that occur between them.
        - The prompt header message (position 0) is always preserved.
        - Note: Call this method after self.add_user_message.
        Args:
            k: Number of most recent user messages to keep
        Returns:
            A new pruned copy of the message history
        """
        if not self.history:
            return []
        messages = copy.deepcopy(self.history)
        if len(messages) <= 1:
            return messages
        prompt_header = messages[0]
        user_indices = [
            i for i, msg in enumerate(messages[1:], 1) if msg.get("role") == "user"
        ]
        if len(user_indices) <= k:
            return messages
        last_k_user_indices = user_indices[-k:]
        earliest_index = min(last_k_user_indices)

        return [prompt_header] + messages[earliest_index:]
