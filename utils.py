import json
import xml.etree.ElementTree as ET
import tiktoken
import re

# OSWorld
from mm_agents.accessibility_tree_wrap.heuristic_retrieve import (
    filter_nodes,
    draw_bounding_boxes,
)

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"


def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):
    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = [
        "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"
    ]

    # Linearize the accessibility tree nodes into a table format
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text
                if '"' not in node.text
                else '"{:}"'.format(node.text.replace('"', '""'))
            )

        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith(
            "EditWrapper"
        ) and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (
                node_text
                if '"' not in node_text
                else '"{:}"'.format(node_text.replace('"', '""'))
            )
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag,
                node.get("name", ""),
                text,
                node.get("{{{:}}}class".format(_attributes_ns), "")
                if platform == "ubuntu"
                else node.get("{{{:}}}class".format(class_ns_windows), ""),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get("{{{:}}}screencoord".format(_component_ns), ""),
                node.get("{{{:}}}size".format(_component_ns), ""),
            )
        )

    return "\n".join(linearized_accessibility_tree)


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree


def tag_screenshot(screenshot, accessibility_tree, platform="ubuntu"):
    nodes = filter_nodes(
        ET.fromstring(accessibility_tree), platform=platform, check_image=True
    )
    # Make tag screenshot
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(
        nodes, screenshot
    )

    return marks, drew_nodes, tagged_screenshot, element_list


# NOTE: to-be deleted later, method already exists in clemcore.utils.file_utils (load_json)
def load_json(file_path: str) -> dict:
    """
    Load and parse a JSON file.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        dict: Parsed JSON content

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found at: {file_path}")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Invalid JSON format in file: {file_path}")


def parse_actions_from_string(input_string: str) -> list:
    """Parse actions from string when action_space is 'computer_13'."""
    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]

    actions = []
    # Try to find JSON in code blocks first
    matches = re.findall(r"```(?:json\s+)?(.*?)```", input_string, re.DOTALL)

    if matches:
        for match in matches:
            try:
                action_dict = json.loads(match)
                actions.append(action_dict)
            except json.JSONDecodeError:
                continue
        if actions:
            return actions

    # Try parsing entire string as JSON if no valid code blocks found
    try:
        action_dict = json.loads(input_string)
        return [action_dict]
    except json.JSONDecodeError:
        raise ValueError(f"Invalid response format: {input_string}")


def parse_code_from_string(input_string: str) -> list:
    """Parse actions from string when action_space is 'pyautogui'."""
    input_string = "\n".join(
        [line.strip() for line in input_string.split(";") if line.strip()]
    )

    if input_string.strip() in ["WAIT", "DONE", "FAIL"]:
        return [input_string.strip()]

    pattern = r"```(?:\w+\s+)?(.*?)```"
    matches = re.findall(pattern, input_string, re.DOTALL)
    codes = []

    for match in matches:
        match = match.strip()
        commands = ["WAIT", "DONE", "FAIL"]

        if match in commands:
            codes.append(match)
        elif match.split("\n")[-1] in commands:
            if len(match.split("\n")) > 1:
                codes.append("\n".join(match.split("\n")[:-1]))
            codes.append(match.split("\n")[-1])
        else:
            codes.append(match)

    return codes


def parse_code_from_som_string(input_string: str, masks: list) -> list:
    """Parse actions from string when observation_type is 'som'."""
    # Generate tag variables from masks
    tag_vars = "\n".join(
        f"tag_{i + 1}=({int(x + w // 2)}, {int(y + h // 2)})"
        for i, (x, y, w, h) in enumerate(masks)
    )

    actions = parse_code_from_string(input_string)

    # Add tag variables to non-command actions
    return [
        f"{tag_vars}\n{action}" if action not in ["WAIT", "DONE", "FAIL"] else action
        for action in actions
    ]


def extract_actions(
    response: str, observation_type: str, action_space: str, masks: list = None
) -> list:
    """
    Extract actions from response based on observation type and action space.

    Args:
        response: Response string from the LLM
        observation_type: Type of observation ('screenshot', 'a11y_tree', 'screenshot_a11y_tree', 'som')
        action_space: Type of action space ('computer_13', 'pyautogui')
        masks: List of masks for SOM parsing (optional)

    Returns:
        List of parsed actions
    """
    if observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
        if action_space == "computer_13":
            return parse_actions_from_string(response)
        elif action_space == "pyautogui":
            return parse_code_from_string(response)
        else:
            raise ValueError(
                f"Invalid action space: {action_space}"
            )  # FIXME; instead of ValueError log_to_self

    elif observation_type == "som":
        if action_space != "pyautogui":
            raise ValueError(f"Invalid action space for SOM: {action_space}")
        return parse_code_from_som_string(response, masks)

    else:
        raise ValueError(f"Invalid observation type: {observation_type}")
