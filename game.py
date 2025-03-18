import os
import shutil
import time
from typing import Dict, Literal, Any
import tempfile
import atexit


from clemcore.clemgame import Player

# FIXME: (OSWorld) Soon these prompts will be replaced by player specific prompts accessible through instances.json file.
from mm_agents.prompts import (
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE,
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_A11Y_OUT_CODE,
    SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE,
    SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_SOM_OUT_TAG,
)
from utils import linearize_accessibility_tree, trim_accessibility_tree, tag_screenshot
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


class PromptHandler:
    """
    Handles the construction and management of prompts for the agent's interaction with the environment.
    Processes different types of observations (screenshots, accessibility trees, or both) and constructs
    appropriate prompts based on the configured observation type and action space.
    """

    def __init__(
        self,
        platform: str = "ubuntu",
        action_space: ACTION_SPACE = "computer_13",
        observation_type: OBSERVATION_TYPE = "screenshot_a11y_tree",
        max_trajectory_length: int = 3,
        a11y_tree_max_tokens: int = 10000,
    ):
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens

        self.thoughts = []
        self.actions = []
        self.observations = []

        # Initialize the temporary image manager
        self.temp_manager = TemporaryImageManager()

        # Define valid combinations and their corresponding system messages
        prompt_mapping = {
            ("screenshot", "computer_13"): SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
            ("screenshot", "pyautogui"): SYS_PROMPT_IN_SCREENSHOT_OUT_CODE,
            ("a11y_tree", "computer_13"): SYS_PROMPT_IN_A11Y_OUT_ACTION,
            ("a11y_tree", "pyautogui"): SYS_PROMPT_IN_A11Y_OUT_CODE,
            ("screenshot_a11y_tree", "computer_13"): SYS_PROMPT_IN_BOTH_OUT_ACTION,
            ("screenshot_a11y_tree", "pyautogui"): SYS_PROMPT_IN_BOTH_OUT_CODE,
            ("som", "pyautogui"): SYS_PROMPT_IN_SOM_OUT_TAG,
        }

        key = (self.observation_type, self.action_space)
        if key not in prompt_mapping:
            raise ValueError(
                f"Invalid combination of observation_type '{self.observation_type}' and action_space '{self.action_space}'"
            )

        self.system_message = prompt_mapping[key]

    def _get_system_message(self, instruction: str) -> str:
        """
        Returns the system message with the given instruction.

        Args:
            instruction (str): The task instruction or goal that the agent needs to accomplish.

        Returns:
            str: System message content
        """
        return f"{self.system_message}\nYou are asked to complete the following task: {instruction}"

    def _get_turn_context(
        self, obs: Dict, turn: int = None, instruction: str = None
    ) -> Dict:
        """
        Constructs the prompt context for a specific turn.
        If turn is 0, prepends the system message to the current context.

        Args:
            obs (Dict): Current observation dictionary containing either:
                - screenshot (str): Base64 encoded screenshot image
                - accessibility_tree (str): Linearized accessibility tree text
                - Both of the above for combined observation types
            turn (int, optional): Current turn number. If 0, includes system message.
            instruction (str, optional): Task instruction needed for system message.

        Returns:
            Dict: Message containing the turn's observation context
        """
        # Get the base observation context first
        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            # Process accessibility tree if needed
            linearized_accessibility_tree = None
            if self.observation_type == "screenshot_a11y_tree":
                linearized_accessibility_tree = linearize_accessibility_tree(
                    accessibility_tree=obs["accessibility_tree"], platform=self.platform
                )
                if linearized_accessibility_tree:
                    linearized_accessibility_tree = trim_accessibility_tree(
                        linearized_accessibility_tree, self.a11y_tree_max_tokens
                    )

            # Save screenshot to temporary file
            tmp_path = self.temp_manager.save_image(obs["screenshot"])

            base_content = (
                "Given the screenshot as below. What's the next step that you will do to help with the task?"
                if self.observation_type == "screenshot"
                else f"Given the screenshot and info from accessibility tree as below:\n{linearized_accessibility_tree}\nWhat's the next step that you will do to help with the task?"
            )

            message = {"role": "user", "content": base_content, "image": [tmp_path]}

        elif self.observation_type == "a11y_tree":
            linearized_accessibility_tree = linearize_accessibility_tree(
                accessibility_tree=obs["accessibility_tree"], platform=self.platform
            )
            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, self.a11y_tree_max_tokens
                )

            message = {
                "role": "user",
                "content": f"Given the info from accessibility tree as below:\n{linearized_accessibility_tree}\nWhat's the next step that you will do to help with the task?",
            }

        elif self.observation_type == "som":
            masks, drew_nodes, tagged_screenshot, linearized_accessibility_tree = (
                tag_screenshot(
                    obs["screenshot"], obs["accessibility_tree"], self.platform
                )
            )

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, self.a11y_tree_max_tokens
                )

            tmp_path = self.temp_manager.save_image(tagged_screenshot)

            message = {
                "role": "user",
                "content": f"Given the tagged screenshot and info from accessibility tree as below:\n{linearized_accessibility_tree}\nWhat's the next step that you will do to help with the task?",
                "image": [tmp_path],
            }

        else:
            raise ValueError(f"Invalid observation_type: {self.observation_type}")

        # If it's turn 0, prepend the system message to the content
        if turn == 0:
            if instruction is None:
                raise ValueError("Instruction is required for turn 0")
            system_message = self._get_system_message(instruction)
            message["content"] = f"{system_message}\n\n{message['content']}"

        return message

    def update_interaction_history(
        self, thought: str, action: str, obs: Dict = None
    ) -> None:
        """
        Updates the interaction history with new thought, action, and optionally a new observation.
        Ensures no duplicate observations are added and processes observations based on type.

        Args:
            thought (str): The thought from the LLM
            action (str): The action from the LLM
            obs (Dict, optional): The observation dictionary. Defaults to None.
        """
        if obs is not None:
            # Process observation based on type
            processed_obs = {}

            if self.observation_type in ["screenshot", "screenshot_a11y_tree", "som"]:
                if self.observation_type == "som":
                    # Process screenshot with SOM tagging
                    (
                        masks,
                        drew_nodes,
                        tagged_screenshot,
                        linearized_accessibility_tree,
                    ) = tag_screenshot(
                        obs["screenshot"], obs["accessibility_tree"], self.platform
                    )
                    screenshot = tagged_screenshot
                else:
                    screenshot = obs["screenshot"]

                processed_obs["screenshot"] = screenshot

            if self.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"]:
                # Process accessibility tree
                linearized_accessibility_tree = linearize_accessibility_tree(
                    accessibility_tree=obs["accessibility_tree"], platform=self.platform
                )
                if linearized_accessibility_tree:
                    linearized_accessibility_tree = trim_accessibility_tree(
                        linearized_accessibility_tree, self.a11y_tree_max_tokens
                    )
                processed_obs["accessibility_tree"] = linearized_accessibility_tree

            self.observations.append(processed_obs)

        # Always append thoughts and actions
        self.thoughts.append(thought)
        self.actions.append(action)

    def get_last_interaction(self) -> Dict[str, str]:
        """
        Returns the most recent thought and action.

        Returns:
            Dict[str, str]: Dictionary containing the last thought and action
        """
        return {
            "thought": self.thoughts[-1] if self.thoughts else None,
            "action": self.actions[-1] if self.actions else None,
        }


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

    def __init__(self, model):
        super().__init__(model)
        self._role = None  # Will be set by metaclass

    @property
    def role(self) -> str:
        """Get the current role of the assistant"""
        return self._role

    def _custom_response(self, messages, turn_idx) -> str:
        """
        Base implementation - will be overridden by role-specific implementation
        This should never be called directly.
        """
        raise NotImplementedError("No role-specific implementation found")


# class InteractiveAssistant(Player):
#     def __init__(self, model):
#         super().__init__(model)

#     def _custom_response(self, messages, turn_idx) -> str:
#         """TODO: Implement the 'oracle' function, which provides an automated solution for completing a game_instance/ task."""
#         pass


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

        # Initialize prompt orchestrator
        self.prompt_handler = PromptHandler(
            platform="ubuntu",
            action_space=self.action_space,
            observation_type=self.observation_type,
            max_trajectory_length=self.max_trajectory_length,
        )


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
