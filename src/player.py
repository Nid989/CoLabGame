from typing import Dict, Any, List

from clemcore.clemgame import Player


class RoleBasedMeta(type(Player)):
    """Metaclass for creating role-specific class implementations of RoleBasedPlayer

    This metaclass dynamically generates subclasses based on the specified role,
    injecting role-specific method implementations (e.g., `_custom_response`).
    """

    _role_implementations: Dict[str, Dict[str, Any]] = {
        "executor": {
            "_custom_response": lambda self,
            context: f"Executor response to {context['content']}"  # Placeholder
        },
        "advisor": {
            "_custom_response": lambda self,
            context: f"Advisor response to {context['content']}"  # Placeholder
        },
    }

    def __call__(cls, model, role: str = "executor", *args, **kwargs):
        """Create an instance of RoleBasedPlayer with role-specific behavior.
        Args:
            cls: The class being instantiated (RoleBasedPlayer).
            model: The model used by the player.
            role: The role of the player (currently; 'executor' or 'advisor').
            *args: Positional arguments passed to the class constructor.
            **kwargs: Keyword arguments passed to the class constructor.
        Returns:
            RoleBasedPlayer: An instance of a dynamically generated subclass with
                role-specific behavior.
        Raises:
            ValueError: If the specified role is not supported.
        """
        if role not in cls._role_implementations:
            raise ValueError(
                f"Invalid role: {role}. Must be one of {list(cls._role_implementations.keys())}"
            )

        # Create a new class dynamically with the role-specific implementation
        role_class = type(
            f"{cls.__name__}_{role.capitalize()}",
            (cls,),
            {
                **cls._role_implementations[role],
                "__module__": cls.__module__,
                "__qualname__": f"{cls.__name__}_{role.capitalize()}",
            },
        )

        # Create instance with the role-specific class
        instance = super(RoleBasedMeta, role_class).__call__(model, *args, **kwargs)
        instance._role = role
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
        **kwargs,
    ):
        """Initialize a RoleBasedPlayer with role-specific configuration.
        Args (additional):
            footer_prompt: A string append to user-context, typically a question for response.
            handler_type: Types of handler ('standard' or 'environment')
            allowed_components: List of message-components permissible for the player's context.
        """
        super().__init__(model, **kwargs)
        self._role = role
        self._footer_prompt = footer_prompt
        self.handler_type = handler_type
        self.allowed_components = allowed_components or []

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

    def _custom_response(self, messages, turn_idx) -> str:
        """Response for programmatic Player interaction.
        - Overwrite this method to implement programmatic behavior (model_name: mock, dry_run, programmatic, custom).
        - Base implementation - will be overridden by role-specific implementation
        - This should never be called directly.
        Args:
            messages: A list of dicts that contain the history of the conversation.
            turn_idx: The index of the current turn.
        Returns:
            The programmatic response as text.
        """
        raise NotImplementedError("No role-specific implementation found")
