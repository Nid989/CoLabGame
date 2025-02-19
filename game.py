import os
import shutil
import time 
from typing import List, Dict, Literal
import tempfile
import atexit

# Clembench
from clemcore.clemgame import Player

# OSWorld
from desktop_env.desktop_env import DesktopEnv
from mm_agents.prompts import SYS_PROMPT_IN_SCREENSHOT_OUT_CODE, SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION, \
    SYS_PROMPT_IN_A11Y_OUT_CODE, SYS_PROMPT_IN_A11Y_OUT_ACTION, \
    SYS_PROMPT_IN_BOTH_OUT_CODE, SYS_PROMPT_IN_BOTH_OUT_ACTION, \
    SYS_PROMPT_IN_SOM_OUT_TAG

# Local 
from utils import linearize_accessibility_tree, trim_accessibility_tree, tag_screenshot

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
            suffix='.png',
            dir=self.temp_dir,
            delete=False
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
            platform: str="ubuntu",
            action_space: ACTION_SPACE="computer_13",
            observation_type: OBSERVATION_TYPE="screenshot_a11y_tree",
            max_trajectory_length: int=3,
            a11y_tree_max_tokens: int=10000
    ):
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        
        self.thoughts = []
        self.actions = []
        self.observations = []
        self._last_observation_hash = None  # To track the last observation

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
            raise ValueError(f"Invalid combination of observation_type '{self.observation_type}' and action_space '{self.action_space}'")
        
        self.system_message = prompt_mapping[key]

    def _get_system_message(self, instruction: str) -> Dict:
        """
        Returns the system message with the given instruction.

        Args:
            instruction (str): The task instruction or goal that the agent needs to accomplish.

        Returns:
            Dict: System message in the format {"role": "system", "content": str}
        """
        return {
            "role": "system", 
            "content": f"{self.system_message}\nYou are asked to complete the following task: {instruction}"
        }

    def _get_curr_context(self, obs: Dict) -> Dict:
        """
        Constructs the prompt for the current observation only.

        Args:
            obs (Dict): Current observation dictionary containing either:
                - screenshot (str): Base64 encoded screenshot image
                - accessibility_tree (str): Linearized accessibility tree text
                - Both of the above for combined observation types

        Returns:
            Dict: Message containing the current observation context
        """
        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            # Process accessibility tree if needed
            linearized_accessibility_tree = None
            if self.observation_type == "screenshot_a11y_tree":
                linearized_accessibility_tree = linearize_accessibility_tree(
                    accessibility_tree=obs["accessibility_tree"],
                    platform=self.platform
                )
                if linearized_accessibility_tree:
                    linearized_accessibility_tree = trim_accessibility_tree(
                        linearized_accessibility_tree,
                        self.a11y_tree_max_tokens
                    )

            # Save screenshot to temporary file
            tmp_path = self.temp_manager.save_image(obs["screenshot"])
            
            message = {
                "role": "user",
                "content": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                if self.observation_type == "screenshot"
                else f"Given the screenshot and info from accessibility tree as below:\n{linearized_accessibility_tree}\nWhat's the next step that you will do to help with the task?",
                "image": tmp_path
            }

        elif self.observation_type == "a11y_tree":
            linearized_accessibility_tree = linearize_accessibility_tree(
                accessibility_tree=obs["accessibility_tree"],
                platform=self.platform
            )
            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree,
                    self.a11y_tree_max_tokens
                )

            message = {
                "role": "user",
                "content": f"Given the info from accessibility tree as below:\n{linearized_accessibility_tree}\nWhat's the next step that you will do to help with the task?"
            }

        elif self.observation_type == "som":
            masks, drew_nodes, tagged_screenshot, linearized_accessibility_tree = tag_screenshot(
                obs["screenshot"], 
                obs["accessibility_tree"], 
                self.platform
            )
            
            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree,
                    self.a11y_tree_max_tokens
                )

            tmp_path = self.temp_manager.save_image(tagged_screenshot)

            message = {
                "role": "user",
                "content": f"Given the tagged screenshot and info from accessibility tree as below:\n{linearized_accessibility_tree}\nWhat's the next step that you will do to help with the task?",
                "image": tmp_path
            }

        else:
            raise ValueError(f"Invalid observation_type: {self.observation_type}")

        return message

    def _get_ovr_context(self, instruction: str, obs: Dict) -> List[Dict]:
        """
        Constructs the prompt context for predicting the next action(s) based on current and historical observations.

        Args:
            instructions (str): The task instruction or goal that the agent needs to accomplish.
            obs (Dict): Current observation dictionary containing either:
                - screenshot (str): Base64 encoded screenshot image
                - accessibility_tree (str): Linearized accessibility tree text
                - Both of the above for combined observation types
        """
        messages = [self._get_system_message(instruction)]

        assert len(self.observations) == len(self.actions) == len(self.thoughts), \
            f"Length mismatch: observations={len(self.observations)}, actions={len(self.actions)}, thoughts={len(self.thoughts)}"

        # Use slice notation with max_trajectory_length or empty list if max_trajectory_length is 0
        history_slice = slice(-self.max_trajectory_length if self.max_trajectory_length else 0, None)
        _observations = self.observations[history_slice]
        _thoughts = self.thoughts[history_slice]
        
        # Define observation type handlers
        observation_handlers = {
            "screenshot_a11y_tree": lambda obs, tmp_path: {
                "content": f"Given the screenshot and info from accessibility tree as below:\n{obs['accessibility_tree']}\nWhat's the next step that you will do to help with the task?",
                "image": tmp_path
            },
            "som": lambda obs, tmp_path: {
                "content": "Given the tagged screenshot as below. What's the next step that you will do to help with the task?",
                "image": tmp_path
            },
            "screenshot": lambda obs, tmp_path: {
                "content": "Given the screenshot as below. What's the next step that you will do to help with the task?",
                "image": tmp_path
            },
            "a11y_tree": lambda obs, _: {
                "content": f"Given the info from accessibility tree as below:\n{obs['accessibility_tree']}\nWhat's the next step that you will do to help with the task?",
                "image": None
            }
        }

        if self.observation_type not in observation_handlers:
            raise ValueError(f"Invalid observation_type: {self.observation_type}")

        # Process each observation and thought pair
        for prev_obs, prev_thought in zip(_observations, _thoughts):
            # Handle screenshot if present
            if "screenshot" in prev_obs and self.observation_type != "a11y_tree":
                # Save the screenshot to a temporary file
                tmp_path = self.temp_manager.save_image(prev_obs["screenshot"])
                result = observation_handlers[self.observation_type](prev_obs, tmp_path)
                
                # Construct user message
                user_message = {
                    "role": "user",
                    "content": result["content"]
                }
                
                if result["image"]:
                    user_message["image"] = result["image"]
                    
                messages.append(user_message)
            else:
                # Handle a11y_tree only case
                result = observation_handlers[self.observation_type](prev_obs, None)
                messages.append({
                    "role": "user",
                    "content": result["content"]
                })
            
            # Add assistant's response
            messages.append({
                "role": "assistant",
                "content": prev_thought.strip() if prev_thought else "No valid action"
            })

        # Process current observation
        current_context = self._get_curr_context(obs)
        messages.append(current_context)

        return messages

    def update_interaction_history(self, thought: str, action: str, obs: Dict = None) -> None:
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
                    masks, drew_nodes, tagged_screenshot, linearized_accessibility_tree = tag_screenshot(
                        obs["screenshot"], 
                        obs["accessibility_tree"], 
                        self.platform
                    )
                    screenshot = tagged_screenshot
                else:
                    screenshot = obs["screenshot"]
                
                processed_obs["screenshot"] = screenshot

            if self.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"]:
                # Process accessibility tree
                linearized_accessibility_tree = linearize_accessibility_tree(
                    accessibility_tree=obs["accessibility_tree"],
                    platform=self.platform
                )
                if linearized_accessibility_tree:
                    linearized_accessibility_tree = trim_accessibility_tree(
                        linearized_accessibility_tree,
                        self.a11y_tree_max_tokens
                    )
                processed_obs["accessibility_tree"] = linearized_accessibility_tree

            # Create hash of processed observation
            obs_hash = hash(str(processed_obs))
            
            # Only append if observation is different from last one
            if obs_hash != self._last_observation_hash:
                self.observations.append(processed_obs)
                self._last_observation_hash = obs_hash

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
            'thought': self.thoughts[-1] if self.thoughts else None,
            'action': self.actions[-1] if self.actions else None
        }

class InteractiveAssistant(Player):

    def __init__(self, model_name):
        super().__init__(model_name)

    def __call__(self, 
                 messages,
                 turn_idx: int):
        return super().__call__(messages, turn_idx)
    
    def _custom_response(self, messages, turn_idx):
        """TODO: Implement the 'oracle' function, which provides an automated solution for completing a game_instance/ task."""
        pass

class DesktopGame: 
    _instance = None
    _env = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
        
    @classmethod
    def _create_env(cls, env_config): 
        if cls._env is None:
            cls._env = DesktopEnv(
                path_to_vm=env_config.path_to_vm, 
                action_space=env_config.action_space,
                screen_size=(env_config.screen_width, env_config.screen_height),
                headless=env_config.headless,
                os_type="Ubuntu",
                require_a11y_tree=env_config.observation_type
                in ["a11y_tree", "screenshot_a11y_tree", "som"]
            )   
        return cls._env
    
    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env(self)
        return self._env

    def __init__(
            self,
            path_to_vm: str=None,
            headless: bool=False,
            observation_type: OBSERVATION_TYPE="a11y_tree",
            action_space: ACTION_SPACE="pyautogui",
            screen_width: int=1920,
            screen_height: int=1080, 
            sleep_after_execution: float=0.0,
            max_steps: int=15,
            max_trajectory_length: int=3,
            game_instance: dict=None,
            player_models: List[str]=None
    ):
        # =============================================
        # Environment Configuration Parameters
        # =============================================
        self.path_to_vm = path_to_vm
        self.headless = headless
        self.observation_type = observation_type
        self.action_space = action_space
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sleep_after_execution = sleep_after_execution
        self.max_steps = max_steps
        self.max_trajectory_length = max_trajectory_length

        # =============================================
        # Task Configuration
        # =============================================
        self.game_instance = game_instance
        if self.game_instance is None:
            raise ValueError("game_instance cannot be None - it should contain task configuration")

        # =============================================
        # Component Initialization
        # =============================================
        # Initialize virtual environment with configured parameters
        _ = self.env

        # Initialize prompt orchestrator
        self.prompt_handler = PromptHandler(
            platform="Ubuntu",
            action_space=self.action_space,
            observation_type=self.observation_type,
            max_trajectory_length=self.max_trajectory_length
        )

        # Initialize LLM-based agent for environment interaction/ action prediction
        self.player_models = player_models
        self.assistant = InteractiveAssistant(self.player_models[0]) # limited to single-agent task

if __name__ == "__main__":
    
    game = DesktopGame(
        headless=False,
        observation_type="a11y_tree", 
        action_space="pyautogui",
        screen_width=1920,
        screen_height=1080,
        game_instance={"dummy_key": "dummy_value"}
    )

    print("environment initialized successfully!")

    time.sleep(20)

    game.env.close()
    print("environment closed successfully!")