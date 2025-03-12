from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Literal


class Environment(ABC):
    """Abstract interface for interacting with external environments in a reinforcement learning setting.

    This class defines a standard interface for environment interaction, supporting common
    operations like initialization, action execution, and cleanup across different
    environment implementations.
    """

    @abstractmethod
    def reset(self, task_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reset the environment to an initial state and return the initial observation.

        Args:
            task_config (Dict[str, Any], optional): Configuration for the specific task.

        Returns:
            Dict[str, Any]: Initial observation from the environment.
        """
        pass

    @abstractmethod
    def step(
        self, action: Any, sleep_time: float = 0.0
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute an action and advance the environment by one timestep.

        Args:
            action (Any): The action to execute.
            sleep_time (float, optional): Time to sleep after execution. Defaults to 0.0.

        Returns:
            Tuple containing:
                Dict[str, Any]: Observation from the environment.
                float: Reward signal.
                bool: Whether the episode is done.
                Dict[str, Any]: Additional information.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources used by the environment."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Dict[str, Any]:
        """Get the observation space specification.

        Returns:
            Dict[str, Any]: Specification of the observation space.
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> Dict[str, Any]:
        """Get the action space specification.

        Returns:
            Dict[str, Any]: Specification of the action space.
        """
        pass

    def start_recording(self) -> bool:
        """Start recording interactions with the environment.

        This is an optional method that environments can implement
        if they support recording capabilities.

        Returns:
            bool: True if recording started successfully, False otherwise.
        """
        return False

    def stop_recording(self) -> bool:
        """Stop recording interactions with the environment.

        Returns:
            bool: True if recording stopped successfully, False otherwise.
        """
        return False

    def get_controller(self) -> Optional[Any]:
        """Get access to the underlying environment controller if available.

        This method provides direct access to environment-specific controllers
        for advanced operations not covered by the standard interface.

        Returns:
            Optional[Any]: The environment controller or None if not available.
        """
        return None


class OSWorldDesktopEnvironment(Environment):
    """OSWorld Desktop Environment implementation of the Environment interface.

    Wraps the OSWorld DesktopEnv to provide a standardized interface.
    """

    def __init__(
        self,
        path_to_vm: str,
        action_space: Literal["computer_13", "pyautogui"] = "pyautogui",
        screen_size: Tuple[int, int] = (1920, 1080),
        headless: bool = False,
        os_type: str = "Ubuntu",
        require_a11y_tree: bool = True,
        **kwargs,
    ):
        """Initialize the OSWorld Desktop Environment.

        Args:
            path_to_vm (str): Path to the virtual machine file.
            action_space (str): The action space type. Defaults to "pyautogui".
            screen_size (Tuple[int, int]): Resolution of the screen. Defaults to (1920, 1080).
            headless (bool): Whether to run in headless mode. Defaults to False.
            os_type (str): Type of operating system. Defaults to "Ubuntu".
            require_a11y_tree (bool): Whether to require accessibility tree. Defaults to True.
            **kwargs: Additional arguments to pass to the environment.
        """
        from desktop_env.desktop_env import DesktopEnv

        self._env = DesktopEnv(
            path_to_vm=path_to_vm,
            action_space=action_space,
            screen_size=screen_size,
            headless=headless,
            os_type=os_type,
            require_a11y_tree=require_a11y_tree,
            **kwargs,
        )
        self._action_space_type = action_space

    def reset(self, task_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reset the OSWorld environment with task configuration.

        Args:
            task_config (Dict[str, Any], optional): Task-specific configuration.

        Returns:
            Dict[str, Any]: Initial observation.
        """
        return self._env.reset(task_config or {})

    def step(
        self, action: Any, sleep_time: float = 0.0
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute an action in the OSWorld environment.

        Args:
            action (Any): Action to execute.
            sleep_time (float, optional): Time to sleep after execution. Defaults to 0.0.

        Returns:
            Tuple containing:
                Dict[str, Any]: Observation from the environment.
                float: Reward signal.
                bool: Whether the episode is done.
                Dict[str, Any]: Additional information.
        """
        return self._env.step(action, sleep_time)

    def close(self) -> None:
        """Close the OSWorld environment and clean up resources."""
        return self._env.close()

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Get the observation space of the OSWorld environment.

        Returns:
            Dict[str, Any]: Observation space specification.
        """
        # Access the observation space from the underlying environment
        return getattr(self._env, "observation_space", {})

    @property
    def action_space(self) -> Dict[str, Any]:
        """Get the action space of the OSWorld environment.

        Returns:
            Dict[str, Any]: Action space specification.
        """
        return {
            "type": self._action_space_type,
            # You might want to add more details if available from the environment
        }

    def start_recording(self) -> bool:
        """Start recording in the OSWorld environment.

        Returns:
            bool: True if recording started successfully, False otherwise.
        """
        try:
            self._env.controller.start_recording()
            return True
        except (AttributeError, Exception):
            return False

    def stop_recording(self) -> bool:
        """Stop recording in the OSWorld environment.

        Returns:
            bool: True if recording stopped successfully, False otherwise.
        """
        try:
            self._env.controller.stop_recording()
            return True
        except (AttributeError, Exception):
            return False

    def get_controller(self) -> Optional[Any]:
        """Get the controller of the OSWorld environment.

        Returns:
            Optional[Any]: The environment controller or None if not available.
        """
        return getattr(self._env, "controller", None)


# Factory to create environment instances based on environment type
class EnvironmentFactory:
    """Factory for creating environment instances based on environment type."""

    @staticmethod
    def create_environment(env_type: str, **config) -> Environment:
        """Create an environment instance based on the specified type.

        Args:
            env_type (str): Type of environment to create.
            **config: Configuration for the environment.

        Returns:
            Environment: An instance of the specified environment type.

        Raises:
            ValueError: If the specified environment type is not supported.
        """
        if env_type.lower() == "osworld":
            return OSWorldDesktopEnvironment(**config)
        # Add more environment types as needed
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
