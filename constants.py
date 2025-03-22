from enum import Enum
from typing import Literal


# Declared Literals
ACTION_SPACE = Literal["computer_13", "pyautogui"]
OBSERVATION_SPACE = Literal["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
HANDLER_TYPE = Literal["standard", "environment"]


# Have to make this more descriptive and use-case specific
# TODO: One idea would be to add more log categories.
class LogType(Enum):
    """Defines log categories for internal game master logging."""

    ACTION_INFO = "action_info"  # Successful action extractions
    ACTION_FAIL = "action_fail"  # Failed actions or errors
    ACTION_EXEC = "action_exec"  # Successful execution results
    TURN_PLAN = "turn_plan"  # Planning and thought processes
    TURN_SKIP = "turn_skip"  # No actions available
    TURN_FAIL = "turn_fail"  # Failures at the turn level
    VALIDATION = "validation"  # Validation-related messages
    GAME_STATE = "game_state"  # Tracking game state transitions
    SETUP_ERROR = "setup_error"  # Initialization errors


# Default environment configuration
DEFAULT_ENV_CONFIG = {
    "headless": False,
    "observation_type": "a11y_tree",
    "action_space": "pyautogui",
    "screen_width": 1920,
    "screen_height": 1080,
    "max_steps": 5,
    "max_trajectory_length": 3,
    "path_to_vm": "/Users/nidhirbhavsar/Desktop/WORK/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx",
    "sleep_after_execution": 0.0,
}

# Default roles
DEFAULT_ROLE = "executor"

# Action message templates
ACTION_RESULT_TEMPLATE = "Action: {action}, Reward: {reward}, Done: {done}"
