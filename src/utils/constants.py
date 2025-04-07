from enum import Enum
from typing import Literal, get_args


# Declared Literals
ACTION_SPACE = Literal["computer_13", "pyautogui"]
ACTION_SPACE_values = ", ".join(get_args(ACTION_SPACE))
OBSERVATION_TYPE = Literal["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
OBSERVATION_TYPE_values = ", ".join(get_args(OBSERVATION_TYPE))
HANDLER_TYPE = Literal["standard", "environment"]
HANDLER_TYPE_values = ", ".join(get_args(HANDLER_TYPE))


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
    "observation_type": "screenshot_a11y_tree",
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

# Default handler_type
DEFAULT_HANDLER_TYPE = "standard"

# Action message templates
ACTION_RESULT_TEMPLATE = "Action: {action}, Reward: {reward}, Done: {done}"

# Action space
# Computer13


def get_keyboard_keys():
    keyboard_keys = [
        "\t",
        "\n",
        "\r",
        " ",
        "!",
        '"',
        "#",
        "$",
        "%",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "[",
        "\\",
        "]",
        "^",
        "_",
        "`",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "{",
        "|",
        "}",
        "~",
        "accept",
        "add",
        "alt",
        "altleft",
        "altright",
        "apps",
        "backspace",
        "browserback",
        "browserfavorites",
        "browserforward",
        "browserhome",
        "browserrefresh",
        "browsersearch",
        "browserstop",
        "capslock",
        "clear",
        "convert",
        "ctrl",
        "ctrlleft",
        "ctrlright",
        "decimal",
        "del",
        "delete",
        "divide",
        "down",
        "end",
        "enter",
        "esc",
        "escape",
        "execute",
        "f1",
        "f10",
        "f11",
        "f12",
        "f13",
        "f14",
        "f15",
        "f16",
        "f17",
        "f18",
        "f19",
        "f2",
        "f20",
        "f21",
        "f22",
        "f23",
        "f24",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "final",
        "fn",
        "hanguel",
        "hangul",
        "hanja",
        "help",
        "home",
        "insert",
        "junja",
        "kana",
        "kanji",
        "launchapp1",
        "launchapp2",
        "launchmail",
        "launchmediaselect",
        "left",
        "modechange",
        "multiply",
        "nexttrack",
        "nonconvert",
        "num0",
        "num1",
        "num2",
        "num3",
        "num4",
        "num5",
        "num6",
        "num7",
        "num8",
        "num9",
        "numlock",
        "pagedown",
        "pageup",
        "pause",
        "pgdn",
        "pgup",
        "playpause",
        "prevtrack",
        "print",
        "printscreen",
        "prntscrn",
        "prtsc",
        "prtscr",
        "return",
        "right",
        "scrolllock",
        "select",
        "separator",
        "shift",
        "shiftleft",
        "shiftright",
        "sleep",
        "stop",
        "subtract",
        "tab",
        "up",
        "volumedown",
        "volumemute",
        "volumeup",
        "win",
        "winleft",
        "winright",
        "yen",
        "command",
        "option",
        "optionleft",
        "optionright",
    ]
    return keyboard_keys


X_MAX, Y_MAX = DEFAULT_ENV_CONFIG["screen_width"], DEFAULT_ENV_CONFIG["screen_height"]
KEYBOARD_KEYS = get_keyboard_keys()

ACTION_SPACE = [
    {
        "action_type": "MOVE_TO",
        "note": "move the cursor to the specified position",
        "parameters": {
            "x": {"type": float, "range": [0, X_MAX], "optional": False},
            "y": {"type": float, "range": [0, Y_MAX], "optional": False},
        },
    },
    {
        "action_type": "CLICK",
        "note": "click the specified button",
        "parameters": {
            "click_type": {
                "type": str,
                "range": ["LEFT", "MIDDLE", "RIGHT", "WHEEL_UP", "WHEEL_DOWN"],
                "optional": False,
            },
            "x": {"type": float, "range": [0, X_MAX], "optional": True},
            "y": {"type": float, "range": [0, Y_MAX], "optional": True},
            "num_clicks": {"type": int, "range": [1, 2, 3], "optional": True},
        },
    },
    {
        "action_type": "MOUSE_DOWN",
        "note": "press the specified button",
        "parameters": {
            "click_type": {
                "type": str,
                "range": ["LEFT", "MIDDLE", "RIGHT", "WHEEL_UP", "WHEEL_DOWN"],
                "optional": False,
            }
        },
    },
    {
        "action_type": "MOUSE_UP",
        "note": "release the specified button",
        "parameters": {
            "click_type": {
                "type": str,
                "range": ["LEFT", "MIDDLE", "RIGHT", "WHEEL_UP", "WHEEL_DOWN"],
                "optional": False,
            }
        },
    },
    {
        "action_type": "RIGHT_CLICK",
        "note": "right click at the specified or current position",
        "parameters": {
            "x": {"type": float, "range": [0, X_MAX], "optional": True},
            "y": {"type": float, "range": [0, Y_MAX], "optional": True},
        },
    },
    {
        "action_type": "DOUBLE_CLICK",
        "note": "double click at the specified or current position",
        "parameters": {
            "x": {"type": float, "range": [0, X_MAX], "optional": True},
            "y": {"type": float, "range": [0, Y_MAX], "optional": True},
        },
    },
    {
        "action_type": "DRAG_TO",
        "note": "drag the cursor to the specified position with the left button pressed",
        "parameters": {
            "x": {"type": float, "range": [0, X_MAX], "optional": False},
            "y": {"type": float, "range": [0, Y_MAX], "optional": False},
        },
    },
    {
        "action_type": "SCROLL",
        "note": "scroll the mouse wheel up or down",
        "parameters": {
            "dx": {"type": int, "range": None, "optional": False},
            "dy": {"type": int, "range": None, "optional": False},
        },
    },
    {
        "action_type": "TYPING",
        "note": "type the specified text",
        "parameters": {"text": {"type": str, "range": None, "optional": False}},
    },
    {
        "action_type": "PRESS",
        "note": "press the specified key(s) and release",
        "parameters": {"key": {"type": str, "range": KEYBOARD_KEYS, "optional": False}},
    },
    {
        "action_type": "KEY_DOWN",
        "note": "press the specified key",
        "parameters": {"key": {"type": str, "range": KEYBOARD_KEYS, "optional": False}},
    },
    {
        "action_type": "KEY_UP",
        "note": "release the specified key",
        "parameters": {"key": {"type": str, "range": KEYBOARD_KEYS, "optional": False}},
    },
    {
        "action_type": "HOTKEY",
        "note": "press the specified key combination",
        "parameters": {
            "keys": {
                "type": list,
                "range": KEYBOARD_KEYS,
                "optional": False,
            }
        },
    },
]
