import os
import shutil
import time
from typing import Dict, Literal, Any, List
import tempfile
import atexit

from clemcore.clemgame import Player
from prompt_handler import ComputerGamePromptHandler, HandlerType

# FIXME: (OSWorld) Soon these prompts will be replaced by player specific prompts accessible through instances.json file.
from environment import EnvironmentFactory, Environment


ACTION_SPACE = Literal["computer_13", "pyautogui"]
OBSERVATION_TYPE = Literal["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]


class TemporaryImageManager:
    """
    Manages temporary image files that persist until the program termination.
    Uses tempfile for secure temporary file handling and caches files to avoid duplicates.
    """

    def __init__(self):
        # Create a temporary directory that will be cleaned up at exit
        self.temp_dir = tempfile.mkdtemp()
        # Cache to store mapping of image content to file paths
        self.image_cache = {}
        atexit.register(self.cleanup)

    def save_image(self, image_binary: bytes) -> str:
        """
        Saves a binary image to a temporary file that persists until program exit.
        Returns cached path if the same image was saved before.

        Args:
            image_binary (bytes): Binary image data (PNG format)

        Returns:
            str: Path to saved image file
        """
        # Use image content as cache key
        image_hash = hash(image_binary)

        # Return cached path if image was saved before
        if image_hash in self.image_cache:
            return self.image_cache[image_hash]

        # Create new file if image hasn't been saved before
        tmp_file = tempfile.NamedTemporaryFile(
            suffix=".png", dir=self.temp_dir, delete=False
        )

        with tmp_file as f:
            f.write(image_binary)
            f.flush()

        # Cache the path
        self.image_cache[image_hash] = tmp_file.name
        return tmp_file.name

    def cleanup(self):
        """Removes the temporary directory and all files in it."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.image_cache.clear()


class RoleBasedMeta(type(Player)):
    """Metaclass for creating role-specific class implementations"""

    _role_implementations: Dict[str, Dict[str, Any]] = {
        "executor": {
            "_custom_response": lambda self, messages, turn_idx: (
                # Actor-specific implementation
                None  # Placeholder
            )
        },
        "advisor": {
            "_custom_response": lambda self, messages, turn_idx: (
                # Guide-specific implementation
                None  # Placeholder
            )
        },
    }

    def __call__(cls, model, role: str = "actor", *args, **kwargs):
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
    """
    A role-based interactive assistant that dynamically changes its implementation
    based on the provided role.
    """

    def __init__(
        self,
        model,
        role: str = "executor",
        prompt_header: str = None,
        handler_type: HandlerType = "standard",
        **kwargs,
    ):
        super().__init__(model)
        self._role = role

        handler_kwargs = kwargs.copy()
        if handler_type == "environment":
            handler_kwargs.update(
                {
                    "observation_type": kwargs.get("observation_type", "screenshot"),
                    "action_space": kwargs.get("action_space", "computer_13"),
                    "platform": kwargs.get("platform", "ubuntu"),
                    "a11y_tree_max_tokens": kwargs.get("a11y_tree_max_tokens", 10000),
                }
            )

            use_images = (
                "screenshot" in handler_kwargs.get("observation_type", "")
                or handler_kwargs.get("observation_type", "") == "som"
            )
            if use_images:
                handler_kwargs["temp_manager"] = TemporaryImageManager()

        self.prompt_handler = ComputerGamePromptHandler(
            handler_type=handler_type, prompt_header=prompt_header, **handler_kwargs
        )

    @property
    def role(self) -> str:
        """Get the current role of the assistant"""
        return self._role

    def add_user_message(self, utterance: str, image: List[str] = None) -> None:
        """Delegate to prompt handler to add a user message"""
        self.prompt_handler.add_user_message(utterance, image)

    def add_assistant_message(self, utterance: str) -> None:
        """Delegate to prompt handler to add an assistant message"""
        self.prompt_handler.add_assistant_message(utterance)

    def update_with_observation(self, obs: Dict) -> None:
        """Process observation using prompt handler"""
        self.prompt_handler.update(observation=obs)

    def _custom_response(self, messages, turn_idx) -> str:
        """
        Base implementation - will be overridden by role-specific implementation
        This should never be called directly.
        """
        raise NotImplementedError("No role-specific implementation found")


class ComputerGame:
    _instance = None
    _env = None
    _current_turn = 0  # Add class-level turn counter

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def _create_env(cls, env_config) -> Environment:
        """Create or return the singleton environment instance."""
        if cls._env is None:
            # Use the EnvironmentFactory to create the appropriate environment
            require_a11y_tree = env_config.observation_type in [
                "a11y_tree",
                "screenshot_a11y_tree",
                "som",
            ]

            cls._env = EnvironmentFactory.create_environment(
                "osworld",
                path_to_vm=env_config.path_to_vm,
                action_space=env_config.action_space,
                screen_size=(env_config.screen_width, env_config.screen_height),
                headless=env_config.headless,
                os_type="Ubuntu",
                require_a11y_tree=require_a11y_tree,
            )
        return cls._env

    @property
    def env(self) -> Environment:
        """Get the environment instance, creating it if necessary."""
        if self._env is None:
            self._env = self._create_env(self)
        return self._env

    @property
    def current_turn(self) -> int:
        """Get the current turn number."""
        return self._current_turn

    @current_turn.setter
    def current_turn(self, value: int):
        """Set the current turn number."""
        self._current_turn = value

    def __init__(
        self,
        path_to_vm: str = None,
        headless: bool = False,
        observation_type: OBSERVATION_TYPE = "a11y_tree",
        action_space: ACTION_SPACE = "pyautogui",
        screen_width: int = 1920,
        screen_height: int = 1080,
        sleep_after_execution: float = 0.0,
        max_steps: int = 15,
        max_trajectory_length: int = 3,
        game_instance: dict = None,
    ):
        # Environment Configuration Parameters
        self.path_to_vm = path_to_vm
        self.headless = headless
        self.observation_type = observation_type
        self.action_space = action_space
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sleep_after_execution = sleep_after_execution
        self.max_steps = max_steps
        self.max_trajectory_length = max_trajectory_length

        # Task Configuration
        self.game_instance = game_instance
        if self.game_instance is None:
            raise ValueError(
                "game_instance cannot be None - it should contain task configuration"
            )

        # Component Initialization
        # Initialize virtual environment with configured parameters
        _ = self.env


if __name__ == "__main__":
    game = ComputerGame(
        headless=False,
        observation_type="a11y_tree",
        action_space="pyautogui",
        screen_width=1920,
        screen_height=1080,
        game_instance={"dummy_key": "dummy_value"},
    )

    print("environment initialized successfully!")

    time.sleep(20)

    game.env.close()
    print("environment closed successfully!")
