import os
import json
import re
import logging
import shutil
from typing import Dict, List, Optional, Union, Tuple, Any
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

from clemcore import backends
from clemcore.clemgame import (
    GameMaster,
    GameBenchmark,
    GameScorer,
    GameSpec,
    metrics,
    ParseError,
    GameError,
    RuleViolationError,
)
from src.master import NetworkDialogueGameMaster, EdgeCondition
from src.environment import Environment, EnvironmentFactory
from src.player import RoleBasedPlayer
from src.message import MessageType, MessageState, PlayerContextFormatter
from src.utils.registry.parsers import parsers, get_parser_metadata, process_content
from src.utils.image_manager import ImageManager
from src.utils.s3_manager import S3Manager
from src.topologies.factory import TopologyFactory
from src.topologies.base import TopologyType

load_dotenv()

logger = logging.getLogger(__name__)


# Result monad implementation for chaining operations
@dataclass
class Ok:
    value: Any

    def __or__(self, func):
        try:
            return func(self.value)
        except Exception as e:
            return Error(e)


@dataclass
class Error:
    error: Exception

    def __or__(self, func):
        return self  # Short-circuit on error


Result = Union[Ok, Error]


class ComputerGame(NetworkDialogueGameMaster):
    def __init__(
        self,
        name: str,
        path: str,
        experiment: Dict,
        player_models: List[backends.Model],
    ):
        super().__init__(name, path, experiment, player_models)
        self.env: Environment = None
        self.game_instance: Dict = None
        self.game_config: Dict = None
        self.message_state = MessageState()
        self.player_context_formatter = None
        self.aborted: bool = False
        self.fail: bool = False
        self.success: bool = False
        self.env_terminated: bool = False
        self.request_count: int = 0
        self.request_count_parsed: int = 0
        self.request_count_violated: int = 0
        self.player_stats = {}
        self.round_stats = {}
        self._episode_score: float = 0.0

    def _on_setup(self, **game_instance) -> None:
        """Method executed at the start of the default setup method.

        Key Actions:
            - Prepares game configuration.
            - Sets up environment and loads initial observation + starts gameplay recording.
            - Constructs player interaction graph/network.
            - Sets up trigger pipeline (specific) 'parse func. -> after parse steps'

        Args:
            game_instance: Keyword arguments of the game_instance
        """
        self.game_instance = game_instance
        self.environment_type = self.experiment["environment_type"].lower()
        self._prepare_game_config()
        self._prepare_game_instance()
        self._initialize_formatter()
        self._initialize_environment()
        self._build_graph()
        self._initialize_topology_specific_components()

    def _prepare_game_config(self) -> None:
        """Prepare game configuration dictionary"""
        game_config = self.experiment["config"].copy()
        observation_type = game_config.get("observation_type", "a11y_tree")
        use_images = observation_type in ["screenshot", "screenshot_a11y_tree", "som"]
        require_a11y_tree = observation_type in [
            "a11y_tree",
            "screenshot_a11y_tree",
            "som",
        ]
        game_config.update({"use_images": use_images, "require_a11y_tree": require_a11y_tree})

        # Always initialize ImageManager for archival purposes
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION")
        s3_bucket = os.getenv("S3_BUCKET_NAME")

        if not all([aws_access_key_id, aws_secret_access_key, aws_region, s3_bucket]):
            raise ValueError("Missing required S3 environment variables.")

        game_config["image_manager"] = ImageManager(
            game_id=self.game_instance["game_id"],
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
            s3_bucket=s3_bucket,
        )
        self.log_key("image_manager_s3_prefix", game_config["image_manager"].s3_prefix)

        game_config["topology_type"] = TopologyType(game_config["topology_type"])
        self.game_config = game_config

    def _prepare_game_instance(self) -> None:
        """Prepare the game instance by applying templates and replacing placeholders"""
        templates = self.experiment["templates"]
        roles = templates["roles"]
        graph = templates["graph"]

        # Check if participants dictionary exists for dynamic graph generation
        participants = self.game_instance.get("participants")
        if participants:
            # Generate graph and roles dynamically based on participants
            graph, roles = self._generate_dynamic_graph_and_roles(participants, roles)

        self.game_instance["roles"] = roles
        self.game_instance["graph"] = graph

    def _generate_dynamic_graph_and_roles(self, participants: Dict, base_roles: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """Generate graph and roles dynamically based on participants configuration.

        Args:
            participants: Dictionary with participant configuration
            base_roles: Base role templates from experiment

        Returns:
            Tuple[Dict, List[Dict]]: (graph_config, updated_roles)
        """
        # Create topology instance
        topology = TopologyFactory.create_topology(self.game_config["topology_type"])

        # NEW: Load per-instance topology configuration
        topology.load_game_instance_config(self.game_instance)

        # Generate graph using topology (now with loaded config)
        graph_config = topology.generate_graph(participants)

        # Update roles based on topology (now using dynamic roles from topology config)
        updated_roles = self._update_roles_for_topology(base_roles, participants, topology)

        return graph_config, updated_roles

    def _update_roles_for_topology(self, base_roles: List[Dict], participants: Dict, topology) -> List[Dict]:
        """Create fresh roles based on dynamic topology configuration.

        Args:
            base_roles: Base role templates from experiment (ignored in new system)
            participants: Dictionary with participant configuration
            topology: Topology instance with loaded configuration

        Returns:
            List[Dict]: Fresh role configurations created from topology
        """
        # Get the node assignments from the generated graph
        graph_config = self.game_instance.get("graph", {})
        node_assignments = graph_config.get("node_assignments", {})

        if not node_assignments:
            logger.warning("No node assignments found in graph config, falling back to basic role creation")
            return self._create_basic_roles_from_topology(topology)

        # Get topology configuration
        role_definitions = topology.topology_config.get("role_definitions", {}) if topology.topology_config else {}

        # Create fresh roles from topology configuration
        fresh_roles = []

        for topology_role_name, role_nodes in node_assignments.items():
            if topology_role_name in role_definitions:
                role_def = role_definitions[topology_role_name]

                # Create role configuration based on topology definition
                role_config = {
                    "name": topology_role_name,
                    "handler_type": role_def.get("handler_type", "standard"),
                    "allowed_components": role_def.get("allowed_components", []),
                    "receives_goal": role_def.get("receives_goal", False),
                    "message_permissions": role_def.get("message_permissions", {"send": [], "receive": []}),
                    # Add domain information for this role
                    "domains": [node["domain"] for node in role_nodes],
                    "node_count": len(role_nodes),
                }

                fresh_roles.append(role_config)

        logger.info(f"Created {len(fresh_roles)} fresh roles from topology configuration")
        return fresh_roles

    def _create_basic_roles_from_topology(self, topology) -> List[Dict]:
        """Fallback method to create basic roles when node assignments are not available."""
        role_definitions = topology.topology_config.get("role_definitions", {}) if topology.topology_config else {}

        basic_roles = []
        for role_name, role_def in role_definitions.items():
            role_config = {
                "name": role_name,
                "handler_type": role_def.get("handler_type", "standard"),
                "allowed_components": role_def.get("allowed_components", []),
                "receives_goal": role_def.get("receives_goal", False),
                "message_permissions": role_def.get("message_permissions", {"send": [], "receive": []}),
                "domains": ["general"],
                "node_count": 1,
            }
            basic_roles.append(role_config)

        return basic_roles

    def _initialize_formatter(self) -> None:
        """Initialize the player context formatter with the current game configuration."""
        self.player_context_formatter = PlayerContextFormatter(game_config=self.game_config)

    def _process_screenshot(self, observation: Dict) -> None:
        """Process screenshot in observation: save to S3 and handle path replacement based on observation type."""
        # Handles screenshot processing: saves image to S3, updates observation with local path or removes screenshot key based on observation type.
        if "screenshot" not in observation or not isinstance(observation["screenshot"], bytes):
            return

        image_manager = self.game_config.get("image_manager")
        if not image_manager:
            return

        image_manager.save_image(observation["screenshot"])
        observation_type = self.game_config.get("observation_type", "a11y_tree")

        if observation_type in ["screenshot", "screenshot_a11y_tree", "som"]:
            local_path = image_manager.get_latest_image_path()
            observation["screenshot"] = local_path if local_path else None
        elif observation_type == "a11y_tree":
            observation.pop("screenshot", None)

        # Log the wget link for the latest uploaded screenshot image
        self.log_to_self("screenshot", {"image": [image_manager.get_latest_image_wget_link()]})

    def _initialize_environment(self) -> None:
        """Initializes game environment with recording capabilities and retrieves the initial state observation.

        Raises:
            RuntimeError: If environment or recording initialization fails
        """
        try:
            self.env = EnvironmentFactory.create_environment(self.environment_type, **self.game_config)
            observation = self.env.reset(task_config=self.game_instance["task_config"])
            self._process_screenshot(observation)
            self.message_state.update(observation=observation)
            if not self.env.start_recording():
                raise RuntimeError("Failed to start environment recording")
        except Exception as e:
            self.aborted = True
            error_message = (
                f"Environment initialization failed: {str(e)}" if "recording" not in str(e).lower() else f"Recording initialization failed: {str(e)}"
            )
            raise RuntimeError(error_message) from e

    def _build_graph(self) -> None:
        """Builds a player-network graph from the game instance configuration.

        Raises:
            RuntimeError: If graph building fails
            ValueError: If player model is not available for role index
            KeyError: If condition type is invalid
        """
        try:
            from src.message import RoleConfig
            from src.utils.template_manager import PromptTemplateManager

            # Initialize template manager
            template_manager = PromptTemplateManager()

            graph_config = self.game_instance.get("graph")
            roles = self.game_instance.get("roles", [])

            for node in graph_config.get("nodes", []):
                node_id = node.get("id")
                node_type = node.get("type")
                if node_type != "PLAYER" or not node_id:
                    continue

                role_index = node.get("role_index", 0)
                role_data = roles[role_index]

                # Create RoleConfig from role data
                role_config = RoleConfig.from_dict(role_data)

                # Generate dynamic prompt if no initial_prompt is provided
                if not role_config.initial_prompt:
                    participants = self.game_instance.get("participants")
                    goal = None
                    if role_config.receives_goal:
                        goal = self.game_instance["task_config"]["instruction"]
                    # Get topology type for template manager
                    topology_type_enum = self.game_config.get("topology_type")
                    # Convert string to enum if needed
                    if isinstance(topology_type_enum, str):
                        from src.topologies.base import TopologyType

                        topology_type_enum = TopologyType(topology_type_enum.upper())

                    role_config.initial_prompt = template_manager.generate_prompt(
                        role_config, self.game_config.get("observation_type"), participants, node_id, goal, topology_type_enum, graph_config
                    )

                print("sliding window size", self.game_config.get("sliding_window_size"))
                # Create player with message permissions
                player = RoleBasedPlayer(
                    self.player_models[0],
                    role=node_id,  # Use node_id as the role identifier
                    handler_type=role_config.handler_type,
                    allowed_components=role_config.allowed_components,
                    message_permissions=role_config.message_permissions,
                    sliding_window_size=self.game_config.get("sliding_window_size"),
                )

                self.add_player_to_graph(
                    player=player,
                    initial_prompt=role_config.initial_prompt,
                    node_id=node_id,
                )

            for edge in graph_config.get("edges", []):
                from_node = edge.get("from")
                to_node = edge.get("to")
                edge_type = edge.get("type")
                description = edge.get("description", "")
                if not from_node or not to_node or not edge_type:
                    continue
                if edge_type == "STANDARD":
                    self.add_standard_edge(from_node, to_node, description)
                elif edge_type == "DECISION":
                    condition_config = edge.get("condition", {})
                    message_type = condition_config.get("type")
                    if message_type not in MessageType.__members__:
                        raise KeyError(f"Invalid message-type field: {message_type}")
                    condition = EdgeCondition(message_type=message_type, description=description)
                    self.add_decision_edge(from_node, to_node, condition, description)

            anchor_node = graph_config.get("anchor_node")
            if anchor_node:
                self.set_anchor_node(anchor_node)

            logger.info("Graph building complete")
        except Exception as e:
            raise RuntimeError(f"Failed to build interaction graph: {str(e)}") from e

    def _initialize_topology_specific_components(self) -> None:
        """Initialize topology-specific components and configurations.

        This method delegates to topology classes to initialize their specific components.
        Each topology can set up any special components they need for their operation.
        """
        topology_type = self.game_config["topology_type"]

        # Create topology instance and delegate initialization
        topology = TopologyFactory.create_topology(topology_type)
        components = topology.initialize_game_components(self.game_instance, self.game_config)

        # Apply components to game state
        for component_name, component_value in components.items():
            setattr(self, component_name, component_value)

        # Handle special post-initialization logic
        if hasattr(self, "blackboard_manager") and self.blackboard_manager:
            self._get_blackboard_context()
        else:
            # Set default blackboard_manager to None for other topologies
            if not hasattr(self, "blackboard_manager"):
                self.blackboard_manager = None

    def _get_next_node(self) -> str:
        """Get next node in round-robin sequence based on current node.

        Returns:
            Next node ID in the sequence
        """
        if not hasattr(self, "node_sequence") or not self.node_sequence:
            return None

        # Find current node's position in the sequence
        try:
            current_index = self.node_sequence.index(self._current_node)
        except ValueError:
            # Current node not in sequence, fallback to first node
            logger.warning(f"Current node {self._current_node} not found in node_sequence, using first node")
            return self.node_sequence[0] if self.node_sequence else None

        # Calculate next node index (round-robin)
        next_index = (current_index + 1) % len(self.node_sequence)
        next_node = self.node_sequence[next_index]

        return next_node

    def _get_blackboard_context(self) -> None:
        """Get blackboard context and store raw data in message state."""
        if self.blackboard_manager:
            # Get raw blackboard entries (not formatted)
            raw_entries = self.blackboard_manager.get_history()

            # Store raw entries in message state
            self.message_state.update(blackboard=raw_entries)

    def _does_game_proceed(self) -> bool:
        """Determine if the game should continue to the next turn.

        Returns:
            bool: False if game is completed or aborted, True otherwise.
        """
        # Stop if a critical error has occurred.
        if self.aborted:
            print("aborted", self.aborted)
            return False

        # Stop if the environment has signaled the game is over.
        if self.env_terminated:
            print("env_terminated", self.env_terminated)
            self.log_to_self("info", "Environment signaled termination.")
            return False

        # Stop if the game has reached the maximum number of rounds.
        max_rounds = self.game_config.get("max_rounds", 1)
        if self.current_round >= max_rounds:
            self.aborted = True
            reason = f"Maximum rounds {max_rounds} reached"
            self.log_to_self("aborted", {"reason": reason})
            return False

        # Stop if the game has reached the maximum number of transitions per round.
        # We check the number of transitions in the current round.
        max_transitions = self.game_config.get("max_transitions_per_round", 10)
        if self.transition.total_transitions >= max_transitions:
            self.aborted = True
            reason = f"Maximum transitions per round {max_transitions} reached"
            self.log_to_self("aborted", {"reason": reason})
            return False

        # Stop if the current node is the designated end node.
        if self._current_node == "END":
            self.log_to_self("info", "Reached END node.")
            return False

        return True

    def _set_context_for(self, player: RoleBasedPlayer, formatted_context: Dict) -> None:
        """Sets context for a player based on formatted context data.

        Args:
            player: Player instance to set context for
            formatted_context: Dictionary containing content and optional image data
        """
        if "image" in formatted_context and formatted_context["image"]:
            self.set_context_for(
                player,
                formatted_context["content"],
                image=formatted_context["image"],
            )
        else:
            self.set_context_for(player, formatted_context["content"])

    def _on_before_game(self):
        """Executed once at the start, before entering the play loop.

        Key Actions:
            - Adds the initial game-context to the anchor player
        """
        super()._on_before_game()
        assert self._current_node == self.anchor_node, "Current node must be the anchor node at game start"
        context = self.player_context_formatter.create_context_for(self.message_state, self._current_player)
        self.message_state.reset(preserve=["observation", "blackboard"])  # NOTE: do we actually need to preserve blackboard?
        if context is None:
            logger.debug("No context generated for player; skipping inital context setup.")
            return
        self._set_context_for(self._current_player, context)
        logger.info(f"Set initial context for player at node {self._current_node}")

    def extract_json_codeblock(self, text: str) -> Tuple[bool, Optional[Dict[Any, Any]] | Exception]:
        """
        Extracts and parses JSON content from a string containing code blocks, only allowing no language identifier or 'json'.
        Args:
            text: Input string containing JSON within triple-backtick code blocks.
        Returns:
            Tuple[bool, Optional[Dict[Any, Any]] | Exception]: (success, result or error)
        """
        try:
            pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            match = re.search(pattern, text)

            if not match:
                invalid_pattern = r"```[a-zA-Z]+\s*([\s\S]*?)\s*```"
                if re.search(invalid_pattern, text):
                    return False, ParseError(
                        reason="Code block found with invalid language identifier (only 'json' or no identifier allowed)",
                        response=text,
                    )
                return False, ParseError(reason="No code block found in the input text", response=text)

            json_content = match.group(1).strip()

            if not json_content:
                return False, ParseError(reason="Empty code block found", response=text)

            result = json.loads(json_content)

            if not isinstance(result, dict):
                return False, ParseError(reason="Parsed content is not a JSON object", response=json_content)

            logger.info("Successfully parsed JSON from code block")
            return True, result

        except json.JSONDecodeError as e:
            return False, ParseError(reason=f"Invalid JSON format: {str(e)}", response=text)

    def check_json_message(self, data: Dict[Any, Any]) -> Tuple[bool, Optional[MessageType] | Exception]:
        """
        Validates the JSON message structure and returns the MessageType.
        Args:
            data: Parsed JSON dictionary to validate.
        Returns:
            Tuple[bool, Optional[MessageType] | Exception]: (success, result or error)
        """
        try:
            required_keys = {"type", "from", "content"}
            missing_keys = required_keys - set(data.keys())
            if missing_keys:
                return False, ParseError(reason=f"Missing required keys: {missing_keys}", response=str(data))

            try:
                message_type = MessageType[data["type"]]
            except KeyError:
                valid_types = ", ".join(mt.name for mt in MessageType)
                return False, ParseError(
                    reason=f"Invalid message type: {data['type']}. Must be one of {valid_types}",
                    response=str(data),
                )

            # Validate role-based permissions
            current_player = self._current_player
            if hasattr(current_player, "validate_outgoing_message"):
                is_valid, error_msg = current_player.validate_outgoing_message(message_type)
                if not is_valid:
                    return False, ParseError(reason=error_msg, response=str(data))

            if message_type in MessageType.requires_to():
                if "to" not in data:
                    return False, ParseError(
                        reason=f"'to' field is required for {message_type.name} messages",
                        response=str(data),
                    )
                if not isinstance(data["to"], str):
                    return False, ParseError(
                        reason="Invalid type for 'to' field: must be a string",
                        response=str(data),
                    )

                # Validate target role can receive this message type
                target_role = data["to"]
                target_player = self.get_player_by_role(target_role)
                if target_player and hasattr(target_player, "validate_incoming_message"):
                    is_valid, error_msg = target_player.validate_incoming_message(message_type)
                    if not is_valid:
                        return False, ParseError(reason=error_msg, response=str(data))

            elif message_type in MessageType.prohibits_to():
                if "to" in data:
                    return False, ParseError(
                        reason=f"'to' field must not be present for {message_type.name} messages",
                        response=str(data),
                    )

            if not isinstance(data["from"], str):
                return False, ParseError(
                    reason="Invalid type for 'from' field: must be a string",
                    response=str(data),
                )

            # Validate 'from' field matches current player role
            if hasattr(current_player, "role") and data["from"] != current_player.role:
                return False, ParseError(
                    reason=f"'from' field must match current player role. Expected '{current_player.role}', got '{data['from']}'",
                    response=str(data),
                )

            # Basic content type validation
            if message_type == MessageType.EXECUTE and self.game_config.get("action_space") == "computer13":
                if not isinstance(data["content"], list) or not all(isinstance(item, dict) for item in data["content"]):
                    return False, ParseError(
                        reason="Invalid 'content' field for computer13: must be a list of dictionaries",
                        response=str(data),
                    )
            elif message_type in {
                MessageType.EXECUTE,
                MessageType.REQUEST,
                MessageType.RESPONSE,
                MessageType.STATUS,
                MessageType.TASK,
                MessageType.WRITE_BOARD,
            }:
                if not isinstance(data["content"], str):
                    return False, ParseError(
                        reason=f"Invalid 'content' field for {message_type.name}: must be a string",
                        response=str(data),
                    )

            logger.info("JSON message validated successfully")

            return True, message_type

        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return False, ParseError(
                reason=f"Unexpected error during validation: {str(e)}",
                response=str(data),
            )

    def handle_json_content(
        self,
        data: Any,
        message_type: str | MessageType,
        environment_type: str,
        action_space: Optional[str],
    ) -> Tuple[bool, Optional[Any] | Exception]:
        """
        Processes the content field based on message type, environment type, and action space.
        Args:
            data: The content to process (str for pyautogui/REQUEST/RESPONSE/STATUS/TASK, List[Dict] for computer13).
            message_type: The message type (string or MessageType enum, e.g., 'EXECUTE' or MessageType.EXECUTE).
            environment_type: The environment type (e.g., 'osworld').
            action_space: The action space (e.g., 'computer13') for EXECUTE messages.
        Returns:
            Tuple[bool, Optional[Any] | Exception]: (success, result or error)
        """
        try:
            # Convert message_type to string for registry
            message_type_str = message_type.name if isinstance(message_type, MessageType) else message_type

            # Call registry to process content
            success, result = process_content(data, message_type_str, environment_type, action_space)

            if not success:
                if isinstance(result, GameError):
                    return False, result  # Propagate GameError from parsers
                return False, ParseError(reason=str(result), response=str(data))

            if result is None:
                return False, ParseError(
                    reason=f"Failed to parse content for {message_type_str} in {environment_type} with action_space {action_space}",
                    response=str(data),
                )

            # Get target field from parser metadata
            parser_key = (message_type_str, environment_type, action_space)
            if parser_key not in parsers:
                parser_key = (message_type_str, environment_type, None)

            metadata = get_parser_metadata(parser_key)
            target_field = metadata.get("target_field")

            if target_field:
                # TODO: Update the game state with result
                # Example: setattr(game_state, target_field, result)
                logger.info(f"Processed content for {target_field}: {result}")
            else:
                logger.warning(f"No target field defined for parser {parser_key}")

            return True, result

        except Exception as e:
            logger.error(f"Unexpected error during content processing: {str(e)}")
            return False, ParseError(
                reason=f"Unexpected error during content processing: {str(e)}",
                response=str(data),
            )

    def _parse_response(self, player: RoleBasedPlayer, response: str) -> str:
        """
        Chains the parsing, validation, and content processing steps to produce a parsed response.
        Focuses solely on parsing logic using class methods, returning a JSON serializable string.
        Args:
            player: Ignored Player object (part of class method signature).
            response: Input string containing JSON within code blocks.
        Returns:
            str: JSON string with validated fields (type, from, to, content).
        Raises:
            ParseError: If parsing or validation fails (e.g., invalid JSON, missing fields, invalid message type).
            GameError: If action content is invalid (e.g., invalid action parameters).
        """

        def construct_response(content: Any, message_type: MessageType, data: Dict[Any, Any]) -> Result:
            """
            Constructs the final JSON response from parsed content, message type, and original data.
            Args:
                content: The processed content (e.g., List[str], List[Dict], or str).
                message_type: The validated MessageType.
                data: The original parsed JSON dictionary.
            Returns:
                Result: Ok(str) with the JSON response.
            """
            parsed_response_dict = {
                "type": message_type.name,
                "from": data.get("from", ""),
            }
            if message_type in MessageType.requires_to() and "to" in data:
                parsed_response_dict["to"] = data.get("to", "")
            parsed_response_dict["content"] = content  # Preserves List[str], List[Dict], or str
            parsed_response = json.dumps(parsed_response_dict)
            return Ok(parsed_response)

        # Chain the operations using Result monad
        # TODO: the `handle_json_content` method is quite inefficient, as it provide `action_space` everytime, when it is not needed.
        self.request_count += 1
        player_id = str(self._current_player.name)
        self.player_stats.setdefault(
            player_id,
            {"requests": 0, "parsed": 0, "violated": 0, "violated_streak": 0},
        )
        self.player_stats[player_id]["requests"] += 1

        # Update round-level stats
        current_round = self.current_round
        self.round_stats.setdefault(current_round, {"requests": 0, "parsed": 0, "violated": 0, "players": {}})
        self.round_stats[current_round]["requests"] += 1
        self.round_stats[current_round]["players"].setdefault(player_id, {"requests": 0, "parsed": 0, "violated": 0, "violated_streak": 0})
        self.round_stats[current_round]["players"][player_id]["requests"] += 1

        # Chain the operations using Result monad
        # TODO: the `handle_json_content` method is quite inefficient, as it provide `action_space` everytime, when it is not needed.
        result = (
            Ok(response)
            | (
                lambda x: Ok(self.extract_json_codeblock(x))
                if isinstance(x, str)
                else Error(ParseError(reason="Expected string input", response=str(x)))
            )
            | (lambda x: Ok(x[1]) if x[0] else Error(x[1]))  # Ok(Dict) or Error(ParseError)
            | (lambda x: Ok((self.check_json_message(x), x)))
            | (lambda x: Ok((x[0][1], x[1])) if x[0][0] else Error(x[0][1]))  # Ok((MessageType, Dict)) or Error(ParseError)
            | (
                lambda x: Ok(
                    (
                        self.handle_json_content(
                            x[1]["content"],
                            x[0],
                            self.environment_type,
                            self.game_config["action_space"],
                        ),
                        x[0],
                        x[1],
                    )
                )
            )
            | (lambda x: Ok((x[0][1], x[1], x[2])) if x[0][0] else Error(x[0][1]))  # Ok((content, MessageType, Dict)) or Error(ParseError/GameError)
            | (lambda x: construct_response(x[0], x[1], x[2]))  # Ok(str)
        )

        # Extract final result and raise errors
        if isinstance(result, Ok):
            self.request_count_parsed += 1
            self.player_stats[player_id]["parsed"] += 1
            self.round_stats[current_round]["parsed"] += 1
            self.round_stats[current_round]["players"][player_id]["parsed"] += 1
            return result.value
        else:
            raise result.error

    def _execute_actions(self, actions: List[Union[str, Dict]]) -> Dict[str, Union[str, Image.Image, Dict]]:
        """Execute either pyautogui or computer13 actions and record observations.
        Args:
            actions: List of actions (pyautogui code strings or computer13 action dictionaries)
        Returns:
            Dict: Observation dictionary from the environment after executing actions
        Raises:
            GameError: If action execution fails or no observation is recorded
        """
        if not actions:
            raise GameError(reason="No actions to execute")

        observation = None
        for action in actions:
            try:
                # Assume self.env.step returns (observation, reward, done, info)
                observation, reward, done, info = self.env.step(action, self.game_config.get("sleep_after_execution", 0.0))
                if observation is None:
                    raise GameError(reason="Received None observation after action execution")

                self._process_screenshot(observation)

                if done:
                    self.env_terminated = True
                    logger.info("Game termination signal received (done=True)")
                    break
            except Exception as e:
                raise GameError(reason=f"Failed to execute action {str(action)}: {str(e)}")

        if observation is None:
            raise GameError(reason="No observation recorded after executing actions")
        return observation

    def _advance_game(self, player: RoleBasedPlayer, parsed_response: str):
        """Advance the game state based on the player's response.
        Processes the response to determine node transitions and handle messages.

        Args:
            player: The RoleBasedPlayer instance providing the response.
            parsed_response: JSON string containing the parsed message with keys like
                             'type', 'from', 'to', and 'content'.

        Raises:
            GameError: If the message type is unknown or action execution fails.
            RuleViolationError: If no valid transition is found or if the message type does not
                                match the edge condition.
        """
        # Step 1: Parse the JSON response
        data = json.loads(parsed_response)
        message_type = MessageType[data["type"]]
        content = data["content"]

        # Step 2: Apply topology-specific processing (NEW LAYER)
        processed_data = self._apply_topology_processing(data, message_type, player)

        # Step 3: Validate transition from current node
        current_node = self._current_node
        next_node = None
        from_role = player.role if hasattr(player, "role") else None
        to_role = processed_data.get("to") if "to" in processed_data else None

        if message_type == MessageType.STATUS:
            next_node = "END"
        else:
            # First, check decision edges for a valid transition
            decision_edges = self._get_decision_edges(current_node)
            if decision_edges:
                if "to" in processed_data:
                    target_node = processed_data["to"]
                    for to_node, condition in decision_edges:
                        if to_node == target_node and condition.validate(message_type.name, from_role, to_role):
                            next_node = target_node
                            break
                    if next_node is None:
                        raise RuleViolationError(
                            f"No valid transition found to target node {target_node} with message type {message_type.name} from role {from_role}"
                        )
                else:
                    # Check for self-loop or staying at current node
                    for to_node, condition in decision_edges:
                        if to_node == current_node and condition.validate(message_type.name, from_role, to_role):
                            next_node = current_node
                            break
                    if next_node is None:
                        raise RuleViolationError(
                            f"No valid self-loop transition found for message type {message_type.name} from role {from_role} at node {current_node}"
                        )

            # If no decision edge is found, fallback to standard edges
            if next_node is None:
                standard_edges = self._get_standard_edges(current_node)
                if standard_edges:
                    next_node = standard_edges[0][0]  # Take the first standard edge target node
                else:
                    raise RuleViolationError(
                        f"No valid transition (decision or standard) found for message type {message_type.name} from role {from_role} from node {current_node}"
                    )

        # Step 4: Update game state with the validated transition
        self._update_round_tracking(current_node, next_node)
        self.transition.next_node = next_node
        logger.info(f"Transitioned from {current_node} to {next_node} based on message type {message_type.name}")

        # Step 5: Process message content based on type
        if message_type == MessageType.EXECUTE:
            observation = self._execute_actions(content)
            self.message_state.update(observation=observation)
        elif message_type == MessageType.STATUS:
            # Content is a list with one string, e.g., ["DONE"] or ["FAIL"]
            # TODO: Investigate if observation should be recorded after STATUS execution
            _ = self._execute_actions(content)
        elif message_type == MessageType.REQUEST:
            self.message_state.update(request=content)
        elif message_type == MessageType.RESPONSE:
            self.message_state.update(response=content)
        elif message_type == MessageType.WRITE_BOARD:
            self._write_to_blackboard(player, content)
        else:
            raise GameError(reason=f"Unknown message type: {message_type}")

        # Step 6: Prepare context for the next player if transition occurred
        if self.transition.next_node:
            next_player = self.get_player_from_node(self.transition.next_node)
            if next_player:
                # Get blackboard context and store in message state
                self._get_blackboard_context()
                formatted_context = self.player_context_formatter.create_context_for(self.message_state, next_player)
                self._set_context_for(next_player, formatted_context)
                self.message_state.reset(preserve=["observation", "blackboard"])

        # A successful turn resets the consecutive violation counter for the current player
        player_id = str(self._current_player.name)
        if player_id in self.player_stats:
            self.player_stats[player_id]["violated_streak"] = 0

        # Reset round-level consecutive violations as well
        current_round = self.current_round
        if current_round in self.round_stats and player_id in self.round_stats[current_round]["players"]:
            self.round_stats[current_round]["players"][player_id]["violated_streak"] = 0

    def _apply_topology_processing(self, data: Dict, message_type: MessageType, player: RoleBasedPlayer) -> Dict:
        """
        Apply topology-specific processing to determine next node/agent.

        This method delegates to topology-specific processors implemented in topology classes.
        Each topology can implement its own logic for determining transitions.

        Args:
            data: Parsed JSON response data
            message_type: Type of message being processed
            player: Current player instance

        Returns:
            Dict: Modified data with topology-specific changes (e.g., added 'to' field)
        """
        topology_type = self.game_config.get("topology_type")

        # Create topology instance and delegate processing
        topology = TopologyFactory.create_topology(topology_type)

        # Prepare game context for topology processing
        game_context = {
            "current_node": self._current_node,
            "next_node_function": self._get_next_node,
            "node_sequence": getattr(self, "node_sequence", []),
        }

        return topology.process_message(data, message_type, player, game_context)

    def _write_to_blackboard(self, player: RoleBasedPlayer, content: str) -> None:
        """Write to the blackboard (WRITE_BOARD message).

        Args:
            player: The player writing to the blackboard
            content: Content to write to the blackboard
        """
        if self.blackboard_manager:
            role_id = player.role
            self.blackboard_manager.write_content(role_id, content)
            self.log_to_self("blackboard_write", {"role_id": role_id, "content": content, "entry_count": self.blackboard_manager.get_entry_count()})

    def _handle_player_violation(self):
        """Handles the logic for player violations, including counting and checking abortion limits."""
        self.request_count_violated += 1
        player_id = str(self._current_player.name)

        # This should have been set in _parse_response, but as a safeguard:
        self.player_stats.setdefault(
            player_id,
            {"requests": 0, "parsed": 0, "violated": 0, "violated_streak": 0},
        )

        # Increment violation counts
        self.player_stats[player_id]["violated"] += 1
        self.player_stats[player_id]["violated_streak"] += 1

        # Update round-level stats for violation
        current_round = self.current_round
        self.round_stats.setdefault(current_round, {"requests": 0, "parsed": 0, "violated": 0, "players": {}})
        self.round_stats[current_round]["violated"] += 1
        self.round_stats[current_round]["players"].setdefault(player_id, {"requests": 0, "parsed": 0, "violated": 0, "violated_streak": 0})
        self.round_stats[current_round]["players"][player_id]["violated"] += 1
        self.round_stats[current_round]["players"][player_id]["violated_streak"] += 1

        # Check for abortion conditions
        consecutive_limit = self.game_config.get("player_consecutive_violation_limit", 3)
        total_limit = self.game_config.get("player_total_violation_limit", 5)

        consecutive_violations = self.player_stats[player_id]["violated_streak"]
        total_violations = self.player_stats[player_id]["violated"]

        if consecutive_violations >= consecutive_limit:
            self.aborted = True
            reason = f"Player {player_id} exceeded consecutive violation limit ({consecutive_violations}/{consecutive_limit})."
            self.log_to_self("aborted", {"reason": reason})

        elif total_violations >= total_limit:
            self.aborted = True
            reason = f"Player {player_id} exceeded total violation limit ({total_violations}/{total_limit})."
            self.log_to_self("aborted", {"reason": reason})

    def _on_parse_error(self, error: ParseError):
        """Hook to implement consequences for parsing errors e.g. prepare re-prompting or set game state to abort."""
        self.log_to_self("parse_error", str(error))
        self._handle_player_violation()

    def _on_game_error(self, error: GameError):
        """Hook to implement consequences for game errors e.g. prepare re-prompting or set game state to failure."""
        self.log_to_self("game_error", str(error))
        self._handle_player_violation()

    def _on_after_game(self):
        """
        Called after the game ends (when _does_game_proceed returns False), before exiting the play loop.
        Evaluates the environment, sets success/fail state, and logs episode metrics.
        """
        # Step 1: Evaluate the episode and set success/fail/score flags
        # CHANGED: Episode score is now calculated independently of aborted state
        # Previously: if not self.aborted: self._episode_score = float(self.env.evaluate()) else: self._episode_score = 0.0
        # Now: Always evaluate environment to get true performance score
        self._episode_score = float(self.env.evaluate())
        self.success = self._episode_score == 1.0
        self.fail = not self.success
        print("episode_score", self._episode_score)
        print("success", self.success)
        print("fail", self.fail)

        # Step 2: Log all final summary data for the episode.
        log_keys = [
            ("success", self.success),
            ("fail", self.fail),
            ("aborted", self.aborted),
            ("episode_score", self._episode_score),
            ("request_count", self.request_count),
            ("request_count_parsed", self.request_count_parsed),
            ("request_count_violated", self.request_count_violated),
            ("player_stats", self.player_stats),
            ("round_stats", self.round_stats),
        ]
        for key, value in log_keys:
            self.log_key(key, value)
            if key in ["player_stats", "round_stats"]:
                continue
            self.log_to_self(key, value)

        # Cleanup image manager resources
        if "image_manager" in self.game_config:
            self.game_config["image_manager"].cleanup()

    def compute_episode_score(self):
        """
        Returns the score for the current episode.
        The score is pre-computed and stored in _on_after_game.

        Returns:
            float: A score of 1.0 for success or 0.0 for failure.
        """
        return self._episode_score

    def get_player_by_role(self, role_identifier: str) -> Optional[RoleBasedPlayer]:
        """Get a player by their role identifier (e.g., 'executor_1', 'advisor', etc.).

        Args:
            role_identifier: The role identifier to search for

        Returns:
            RoleBasedPlayer or None if not found
        """
        # Search through all players using the inherited players_by_names
        for player in self.players_by_names.values():
            if hasattr(player, "role") and player.role == role_identifier:
                return player

        # If exact match not found, try to find by base role type
        # This handles cases where we're looking for 'executor' but have 'executor_1'
        for player in self.players_by_names.values():
            if hasattr(player, "role"):
                # Extract base role type (e.g., 'executor' from 'executor_1')
                base_role = player.role.split("_")[0] if "_" in player.role else player.role
                if base_role == role_identifier:
                    return player

        return None


class ComputerGameScorer(GameScorer):
    def __init__(self, name: str, experiment: Dict, game_instance: Dict):
        super().__init__(name, experiment, game_instance)

        # Initialize S3 manager for downloading images
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION")
        s3_bucket = os.getenv("S3_BUCKET_NAME")

        self.s3_manager = S3Manager(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
        self.s3_bucket = s3_bucket
        self.image_manager_s3_prefix = None

    def _on_store_scores(self, file_path: str):
        interactions_dir = str(Path(file_path).parent)

        if self.image_manager_s3_prefix:
            data_dir = os.path.join(interactions_dir, "images")
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
            os.makedirs(data_dir)
            try:
                downloaded_files = self.s3_manager.download_directory(
                    bucket_name=self.s3_bucket, s3_prefix=self.image_manager_s3_prefix, local_dir=data_dir
                )
                logger.info(f"Successfully downloaded {len(downloaded_files)} files from S3")
            except Exception as e:
                logger.error(f"Failed to download S3 directory: {e}")
        else:
            logger.info("No image_manager_s3_prefix available, skipping S3 download")

    def score_episode(self, episode_interactions: Dict):
        # Step 1: Store the image manager s3 prefix for later use (_on_store_score)
        self.image_manager_s3_prefix = episode_interactions.get("image_manager_s3_prefix", None)

        # Step 2: Extract key episode-level data
        success = episode_interactions.get("success", False)
        aborted = episode_interactions.get("aborted", False)
        request_count = episode_interactions.get("request_count", 0)
        request_count_parsed = episode_interactions.get("request_count_parsed", 0)
        request_count_violated = episode_interactions.get("request_count_violated", 0)
        player_stats = episode_interactions.get("player_stats", {})

        # Step 3: Log episode-level binary outcomes and raw counts
        self.log_episode_score(metrics.METRIC_SUCCESS, 1 if success else 0)
        self.log_episode_score(metrics.METRIC_LOSE, 1 if not success else 0)
        self.log_episode_score(metrics.METRIC_ABORTED, 1 if aborted else 0)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT, request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_PARSED, request_count_parsed)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_VIOLATED, request_count_violated)

        # Step 4: Calculate and log the final BENCH_SCORE
        # BENCH_SCORE equals success * 100
        bench_score = success * 100

        self.log_episode_score(metrics.BENCH_SCORE, bench_score)

        # Step 6: Log per-player scores for detailed episode-level diagnostics
        for player_id, stats in player_stats.items():
            p_requests = stats.get("requests", 0)
            p_parsed = stats.get("parsed", 0)
            p_violated = stats.get("violated", 0)

            self.log_episode_score(f"{player_id}_{metrics.METRIC_REQUEST_COUNT}", p_requests)
            self.log_episode_score(f"{player_id}_{metrics.METRIC_REQUEST_COUNT_PARSED}", p_parsed)
            self.log_episode_score(f"{player_id}_{metrics.METRIC_REQUEST_COUNT_VIOLATED}", p_violated)
            self.log_episode_score(f"{player_id}_Violated_Streak", stats.get("violated_streak", 0))

    def score_rounds(self, episode_interactions: Dict):
        round_stats = episode_interactions.get("round_stats", {})
        for round_idx, stats in round_stats.items():
            # Log aggregated round scores
            self.log_round_score(round_idx, metrics.METRIC_REQUEST_COUNT, stats["requests"])
            self.log_round_score(round_idx, metrics.METRIC_REQUEST_COUNT_PARSED, stats["parsed"])
            self.log_round_score(round_idx, metrics.METRIC_REQUEST_COUNT_VIOLATED, stats["violated"])

            # Log per-player scores for the round
            for player_name, player_round_stats in stats["players"].items():
                self.log_round_score(
                    round_idx,
                    f"{player_name}_{metrics.METRIC_REQUEST_COUNT}",
                    player_round_stats["requests"],
                )
                self.log_round_score(
                    round_idx,
                    f"{player_name}_{metrics.METRIC_REQUEST_COUNT_PARSED}",
                    player_round_stats["parsed"],
                )
                self.log_round_score(
                    round_idx,
                    f"{player_name}_{metrics.METRIC_REQUEST_COUNT_VIOLATED}",
                    player_round_stats["violated"],
                )
                self.log_round_score(
                    round_idx,
                    f"{player_name}_Violated_Streak",
                    player_round_stats["violated_streak"],
                )


class ComputerGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[backends.Model]) -> GameMaster:
        return ComputerGame(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return ComputerGameScorer(self.game_name, experiment, game_instance)


if __name__ == "__main__":

    def load_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    experiments = load_json("./in/instances.json")
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"][0]
    master = ComputerGame("computergame", None, experiment_1, ["mock", "mock"])
    master.setup(**game_1)
