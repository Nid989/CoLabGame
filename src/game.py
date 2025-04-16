import time

from .environment import EnvironmentFactory, Environment
from .utils.constants import OBSERVATION_TYPE, ACTION_SPACE


# TODO: move this to somewhere in utils


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
