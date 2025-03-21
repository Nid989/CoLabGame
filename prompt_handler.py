from typing import Dict, List, Callable, Literal

from utils import tag_screenshot, linearize_accessibility_tree, trim_accessibility_tree

# Define handler type as a Literal
HandlerType = Literal["standard", "environment"]


class ComputerGamePromptHandler:
    """Prompt-management for RoleBasedPlayer instance.
    Adjusts behavior based on the player's environment access."""

    # Factory methods to create appropriate handler implementations
    @staticmethod
    def create_prepare_messages(handler_type: HandlerType) -> Callable:
        """Factory method to create the appropriate prepare_messages implementation.
        Args:
            handler_type: Type of handler to create implementation for
        Returns:
            Callable: The appropriate prepare_messages implementation
        """
        if handler_type == "standard":

            def prepare_messages(self, instruction: str = None, turn: int = 0) -> list:
                result_messages = []

                # Add prompt header as first user message if available
                if self.prompt_header and turn == 0:
                    header_msg = {"role": "user", "content": self.prompt_header}
                    if instruction:
                        header_msg["content"] += f"\n\nYour task: {instruction}"
                    result_messages.append(header_msg)

                # Add regular conversation history
                result_messages.extend(self.messages)

                return result_messages

            return prepare_messages

        elif handler_type == "environment":

            def prepare_messages(self, instruction: str = None, turn: int = 0) -> list:
                result_messages = []

                # Add prompt header as first user message if available
                if self.prompt_header and turn == 0:
                    header_msg = {"role": "user", "content": self.prompt_header}
                    if instruction:
                        header_msg["content"] += f"\n\nYour task: {instruction}"
                    result_messages.append(header_msg)

                # Add previous conversation history
                result_messages.extend(self.messages)

                # Add the latest observation as a user message if available
                if self.observations:
                    latest_obs = self.observations[-1]

                    if hasattr(self, "observation_type"):
                        if self.observation_type in [
                            "screenshot",
                            "screenshot_a11y_tree",
                            "som",
                        ]:
                            # Process screenshot
                            tmp_path = self.temp_manager.save_image(
                                latest_obs["screenshot"]
                            )

                            if self.observation_type == "screenshot":
                                content = "What's the next step that you will do to help with the task?"
                                user_msg = {
                                    "role": "user",
                                    "content": content,
                                    "image": [tmp_path],
                                }
                            elif self.observation_type in [
                                "screenshot_a11y_tree",
                                "som",
                            ]:
                                content = f"Given the {'tagged ' if self.observation_type == 'som' else ''}screenshot and info from accessibility tree as below:\n{latest_obs['accessibility_tree']}\nWhat's the next step that you will do to help with the task?"
                                user_msg = {
                                    "role": "user",
                                    "content": content,
                                    "image": [tmp_path],
                                }

                        elif self.observation_type == "a11y_tree":
                            content = f"Given the info from accessibility tree as below:\n{latest_obs['accessibility_tree']}\nWhat's the next step that you will do to help with the task?"
                            user_msg = {"role": "user", "content": content}

                        result_messages.append(user_msg)

                return result_messages

            return prepare_messages

        raise ValueError(f"Unknown handler type: {handler_type}")

    @staticmethod
    def create_update(handler_type: HandlerType) -> Callable:
        """Factory method to create the appropriate update implementation.
        Args:
            handler_type: Type of handler to create implementation for
        Returns:
            Callable: The appropriate update implementation
        """
        if handler_type == "standard":

            def update(self, action: str = None, observation: dict = None) -> None:
                # Standard mode ignores actions and observations
                pass

            return update

        elif handler_type == "environment":

            def update(self, action: str = None, observation: dict = None) -> None:
                # Record action if provided
                if action is not None:
                    self.actions.append(action)

                # Process observation if provided
                if observation is not None:
                    processed_obs = {}

                    if hasattr(self, "observation_type"):
                        if self.observation_type in [
                            "screenshot",
                            "screenshot_a11y_tree",
                            "som",
                        ]:
                            if self.observation_type == "som":
                                (
                                    masks,
                                    drew_nodes,
                                    tagged_screenshot,
                                    linearized_a11y_tree,
                                ) = tag_screenshot(
                                    observation["screenshot"],
                                    observation["accessibility_tree"],
                                    self.platform,
                                )
                                screenshot = tagged_screenshot
                            else:
                                screenshot = observation["screenshot"]

                            processed_obs["screenshot"] = screenshot

                        if self.observation_type in [
                            "a11y_tree",
                            "screenshot_a11y_tree",
                            "som",
                        ]:
                            linearized_a11y_tree = linearize_accessibility_tree(
                                accessibility_tree=observation["accessibility_tree"],
                                platform=self.platform,
                            )
                            if linearized_a11y_tree:
                                linearized_a11y_tree = trim_accessibility_tree(
                                    linearized_a11y_tree, self.a11y_tree_max_tokens
                                )
                            processed_obs["accessibility_tree"] = linearized_a11y_tree

                        self.observations.append(processed_obs)

            return update

        raise ValueError(f"Unknown handler type: {handler_type}")

    # Registry mapping handler types to factory method calls
    REGISTRY = {
        "standard": {
            "prepare_messages": create_prepare_messages.__func__("standard"),
            "update": create_update.__func__("standard"),
        },
        "environment": {
            "prepare_messages": create_prepare_messages.__func__("environment"),
            "update": create_update.__func__("environment"),
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
