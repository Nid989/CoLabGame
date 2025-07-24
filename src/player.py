from typing import Dict, Any, List

from clemcore.clemgame import Player


class RoleBasedMeta(type(Player)):
    """Metaclass for creating role-specific class implementations of RoleBasedPlayer

    This metaclass dynamically generates subclasses based on the specified role,
    injecting role-specific method implementations (e.g., `_custom_response`).
    """

    _role_implementations: Dict[str, Dict[str, Any]] = {
        "executor": {
            "_custom_response": lambda self, context: f"Executor response to {context['content']}"  # Placeholder
        },
        "advisor": {
            "_custom_response": lambda self, context: f"Advisor response to {context['content']}"  # Placeholder
        },
    }

    def __call__(cls, model, role: str = "executor", *args, **kwargs):
        """Create an instance of RoleBasedPlayer with role-specific behavior.
        Args:
            cls: The class being instantiated (RoleBasedPlayer).
            model: The model used by the player.
            role: The role of the player (e.g., 'executor', 'advisor', 'executor_1', 'executor_2').
            *args: Positional arguments passed to the class constructor.
            **kwargs: Keyword arguments passed to the class constructor.
        Returns:
            RoleBasedPlayer: An instance of a dynamically generated subclass with
                role-specific behavior.
        Raises:
            ValueError: If the specified role is not supported.
        """
        # Extract base role type for implementation lookup (e.g., 'executor_1' -> 'executor')
        base_role = role.split("_")[0] if "_" in role else role

        if base_role not in cls._role_implementations:
            raise ValueError(f"Invalid base role: {base_role}. Must be one of {list(cls._role_implementations.keys())}")

        # Create a new class dynamically with the role-specific implementation
        role_class = type(
            f"{cls.__name__}_{base_role.capitalize()}",
            (cls,),
            {
                **cls._role_implementations[base_role],
                "__module__": cls.__module__,
                "__qualname__": f"{cls.__name__}_{base_role.capitalize()}",
            },
        )

        # Create instance with the role-specific class
        instance = super(RoleBasedMeta, role_class).__call__(model, *args, **kwargs)
        instance._role = role  # Store the full role (e.g., 'executor_1')
        return instance

    @classmethod
    def register_role(mcs, role: str, implementations: Dict[str, Any]) -> None:
        """Register a new role with its method implementations"""
        mcs._role_implementations[role] = implementations


class RoleBasedPlayer(Player, metaclass=RoleBasedMeta):
    """A role-based interactive player that dynamically changes behavior based on its role.

    Extends the Player class to support role-specific responses, such as 'executor' or
    'advisor', with behavior defined by a metaclass. Supports additional configuration
    for response formatting and component restrictions.
    """

    def __init__(
        self,
        model,
        role: str = "executor",
        footer_prompt: str = "What will be your next step to complete the task?",
        handler_type: str = "standard",
        allowed_components: List[str] = None,
        message_permissions=None,
        memory_config: Dict[str, bool] = None,  # NEW: Memory configuration
        **kwargs,
    ):
        """Initialize a RoleBasedPlayer with role-specific configuration.
        Args (additional):
            footer_prompt: A string append to user-context, typically a question for response.
            handler_type: Types of handler ('standard' or 'environment')
            allowed_components: List of message-components permissible for the player's context.
            message_permissions: MessagePermissions instance for this role, or None for defaults.
            memory_config: Dictionary controlling what content should be forgotten after each turn.
        """
        # Default memory configuration
        default_memory_config = {
            "forget_observations": False,  # Split and forget observation details
            "forget_images": False,  # Forget images
            "forget_goals": False,  # Remember goals
            "forget_requests": False,  # Remember requests
            "forget_responses": False,  # Remember responses
            "forget_plans": False,  # Remember plans
            "forget_tasks": False,  # Remember tasks
            "forget_tagged_content": False,  # Remember tagged content
            "forget_blackboard": False,  # Remember blackboard history
        }

        # Merge with user-provided config
        self.memory_config = {**default_memory_config, **(memory_config or {})}

        # Build forget_extras list based on configuration
        forget_extras = []
        for component, should_forget in self.memory_config.items():
            if should_forget:
                if component == "forget_observations":
                    forget_extras.append("observation_detail")
                elif component == "forget_images":
                    forget_extras.append("image")
                elif component.startswith("forget_"):
                    # Convert forget_requests -> request_detail
                    base_component = component.replace("forget_", "").rstrip("s")  # Remove trailing 's'
                    forget_extras.append(f"{base_component}_detail")

        super().__init__(model, forget_extras=forget_extras, **kwargs)
        self._role = role
        self._footer_prompt = footer_prompt
        self.handler_type = handler_type
        self.allowed_components = allowed_components or []
        self.retries = 0

        # Import here to avoid circular imports
        from src.message import MessagePermissions

        # Set message permissions (use defaults if not provided)
        if message_permissions is None:
            # Extract base role type for permission lookup (e.g., 'executor_1' -> 'executor')
            base_role = role.split("_")[0] if "_" in role else role
            self.message_permissions = MessagePermissions.get_default_for_role(base_role)
        else:
            self.message_permissions = message_permissions

    @property
    def role(self) -> str:
        """Get the current role of the assistant"""
        return self._role

    @property
    def footer_prompt(self) -> str:
        """Get the footer prompt appended to responses."""
        return self._footer_prompt

    @footer_prompt.setter
    def footer_prompt(self, value):
        """Set the footer prompt appended to responses."""
        if not isinstance(value, str):
            raise ValueError("footer_prompt must be a string")
        self._footer_prompt = value

    def can_send(self, message_type) -> bool:
        """Check if this player can send the given message type.

        Args:
            message_type: MessageType enum or string

        Returns:
            bool: True if the player can send this message type
        """
        from src.message import MessageType

        if isinstance(message_type, str):
            message_type = MessageType.from_string(message_type)

        return self.message_permissions.can_send(message_type)

    def can_receive(self, message_type) -> bool:
        """Check if this player can receive the given message type.

        Args:
            message_type: MessageType enum or string

        Returns:
            bool: True if the player can receive this message type
        """
        from src.message import MessageType

        if isinstance(message_type, str):
            message_type = MessageType.from_string(message_type)

        return self.message_permissions.can_receive(message_type)

    def get_allowed_send_types(self) -> List[str]:
        """Get list of message types this player can send as strings."""
        return self.message_permissions.get_send_types_str()

    def get_allowed_receive_types(self) -> List[str]:
        """Get list of message types this player can receive as strings."""
        return self.message_permissions.get_receive_types_str()

    def validate_outgoing_message(self, message_type) -> tuple[bool, str]:
        """Validate if this player can send the given message type.

        Args:
            message_type: MessageType enum or string

        Returns:
            tuple: (is_valid, error_message)
        """
        from src.message import MessageType

        try:
            if isinstance(message_type, str):
                message_type = MessageType.from_string(message_type)

            if not self.can_send(message_type):
                allowed = ", ".join(self.get_allowed_send_types())
                return False, f"Role '{self.role}' cannot send {message_type.name} messages. Allowed: {allowed}"

            return True, ""
        except ValueError as e:
            return False, str(e)

    def validate_incoming_message(self, message_type) -> tuple[bool, str]:
        """Validate if this player can receive the given message type.

        Args:
            message_type: MessageType enum or string

        Returns:
            tuple: (is_valid, error_message)
        """
        from src.message import MessageType

        try:
            if isinstance(message_type, str):
                message_type = MessageType.from_string(message_type)

            if not self.can_receive(message_type):
                allowed = ", ".join(self.get_allowed_receive_types())
                return False, f"Role '{self.role}' cannot receive {message_type.name} messages. Allowed: {allowed}"

            return True, ""
        except ValueError as e:
            return False, str(e)
