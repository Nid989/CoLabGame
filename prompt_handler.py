from typing import Dict, List, Callable, Literal


# Define handler type as a Literal
HandlerType = Literal["standard", "environment"]


class ComputerGamePromptHandler:
    """Prompt-management for RoleBasedPlayer instance"""

    # Factory methods to create appropriate handler implementations.
    @staticmethod
    def create_add_user_message(handler_type: HandlerType) -> Callable:
        """Factory method to create the appropriate add_user_message implementation.
        Args:
            handler_type: Type of handler to create implementation for
        Returns:
            Callable: The appropriate add_user_message implementation
        """
        if handler_type == "standard":

            def add_user_message(self, utterance: str, image: List[str] = None) -> None:
                """Add a message with 'user' role to conversation history.
                Args:
                    utterance: The text content of the message to be added
                    image: Optional list of image paths to include with the message
                """
                message = {"role": "user", "content": utterance}
                if image and len(image) > 0:
                    message["image"] = image
                self.add_message(message)

            return add_user_message

        elif handler_type == "environment":

            def add_user_message(
                self, utterance: str, image: List[str] = None, observation: dict = None
            ) -> None:
                """Add a message with 'user' role to conversation history.
                Args:
                    utterance: The text content of the message to be added
                    image: Optional list of image paths to include with the message
                    observation: Observation data for environment handler
                """
                # Logic for environment handler will be added later
                pass

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
        handler_type: HandlerType = "standard",
        prompt_header: str = None,
        **kwargs,
    ):
        self.handler_type = handler_type
        self.prompt_header = prompt_header
        self.messages = []

        # Set up attributes based on handler type
        if self.handler_type == "environment":
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.observations = []
            self.actions = []

        # Bind the appropriate implementation methods to this instance
        self._bind_implementations(self.handler_type)

        if self.prompt_header:
            self.add_user_message(self.prompt_header)

    def _bind_implementations(self, handler_type: HandlerType) -> None:
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

    def add_assistant_message(self, utterance: str) -> None:
        """Add a message with 'assistant' role to conversation history.
        Args:
            utterance: The text content of the message to be added
        """
        self.add_message({"role": "assistant", "content": utterance})

    def add_message(self, message: dict) -> None:
        """Add a message to the conversation history.
        Args:
            message: Dictionary containing the message data
        """
        self.messages.append(message)

    def clear_history(self) -> None:
        """Clear all conversation history and observations."""
        self.messages = []
        if self.handler_type == "environment":
            self.observations = []
            self.actions = []

    def get_messages(self) -> List[Dict]:
        """Get the current message history.
        Returns:
            List of message dictionaries
        """
        return self.messages

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
