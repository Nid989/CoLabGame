import json
import re
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from PIL import Image

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
from src.utils.general import TemporaryImageManager
from dataclasses import dataclass

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
        self._prepare_game_instance()
        self._prepare_game_config()
        self._initialize_formatter()
        self._initialize_environment()
        self._build_graph()

    def _prepare_game_instance(self) -> None:
        """Prepare the game instance by applying templates and replacing placeholders"""
        templates = self.experiment["templates"]
        roles = templates["roles"]
        graph = templates["graph"]
        self.game_instance["roles"] = roles
        self.game_instance["graph"] = graph
        self.message_state.update(goal=self.game_instance["task_config"]["instruction"])

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
        if use_images:
            print("use_images")
            game_config["temporary_image_manager"] = TemporaryImageManager(game_id=self.game_instance["game_id"])
        self.game_config = game_config

    def _initialize_formatter(self) -> None:
        """Initialize the player context formatter with the current game configuration."""
        self.player_context_formatter = PlayerContextFormatter(game_config=self.game_config)

    def _initialize_environment(self) -> None:
        """Initializes game environment with recording capabilities and retrieves the initial state observation.

        Raises:
            RuntimeError: If environment or recording initialization fails
        """
        try:
            self.env = EnvironmentFactory.create_environment(self.environment_type, **self.game_config)
            observation = self.env.reset(task_config=self.game_instance["task_config"])
            print("--------------------------------")
            print(observation)
            print("--------------------------------")
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
                    role_config.initial_prompt = template_manager.generate_prompt(role_config, self.game_config.get("observation_type"))

                # Create player with message permissions
                player = RoleBasedPlayer(
                    self.player_models[0],
                    role=role_config.name,
                    handler_type=role_config.handler_type,
                    allowed_components=role_config.allowed_components,
                    message_permissions=role_config.message_permissions,
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

    def _does_game_proceed(self) -> bool:
        """Determine if the game should continue to the next turn.

        Returns:
            bool: False if game is completed or aborted, True otherwise.
        """
        # Stop if a critical error has occurred.
        if self.aborted:
            return False

        # Stop if the environment has signaled the game is over.
        if self.env_terminated:
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

        # Step 2: Validate transition from current node
        current_node = self._current_node
        next_node = None
        from_role = player.role if hasattr(player, "role") else None
        to_role = data.get("to") if "to" in data else None

        if message_type == MessageType.STATUS:
            next_node = "END"
        else:
            # First, check decision edges for a valid transition
            decision_edges = self._get_decision_edges(current_node)
            if decision_edges:
                if "to" in data:
                    target_node = data["to"]
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

        # Step 3: Update game state with the validated transition
        self._update_round_tracking(current_node, next_node)
        self.transition.next_node = next_node
        logger.info(f"Transitioned from {current_node} to {next_node} based on message type {message_type.name}")

        # Step 4: Process message content based on type
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
        else:
            raise GameError(reason=f"Unknown message type: {message_type}")

        # Step 5: Prepare context for the next player if transition occurred
        if self.transition.next_node:
            next_player = self.get_player_from_node(self.transition.next_node)
            if next_player:
                formatted_context = self.player_context_formatter.create_context_for(self.message_state, next_player)
                self._set_context_for(next_player, formatted_context)
                self.message_state.reset(preserve=["observation"])
                logger.info(f"Set context for next player at node {self.transition.next_node}")

        # A successful turn resets the consecutive violation counter for the current player
        player_id = str(self._current_player.name)
        if player_id in self.player_stats:
            self.player_stats[player_id]["violated_streak"] = 0

        # Reset round-level consecutive violations as well
        current_round = self.current_round
        if current_round in self.round_stats and player_id in self.round_stats[current_round]["players"]:
            self.round_stats[current_round]["players"][player_id]["violated_streak"] = 0

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
        # Step 1: Evaluate the environment to get the score, done only once.
        score = self.env.evaluate()
        self._episode_score = float(score)  # Store for compute_episode_score

        # Step 2: Determine final success/fail state.
        if self.aborted:
            self.fail = True
            self.success = False
        else:
            if self._episode_score == 1.0:
                self.success = True
                self.fail = False
            else:
                self.success = False
                self.fail = True

        # Step 3: Log all final summary data for the episode.
        self.log_key("success", self.success)
        self.log_key("fail", self.fail)
        self.log_key("aborted", self.aborted)
        self.log_key("episode_score", self._episode_score)
        self.log_key("request_count", self.request_count)
        self.log_key("request_count_parsed", self.request_count_parsed)
        self.log_key("request_count_violated", self.request_count_violated)
        self.log_key("player_stats", self.player_stats)
        self.log_key("round_stats", self.round_stats)

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
        # Search through all players in the network
        for node_id, player in self.players.items():
            if hasattr(player, "role") and player.role == role_identifier:
                return player

        # If exact match not found, try to find by base role type
        # This handles cases where we're looking for 'executor' but have 'executor_1'
        for node_id, player in self.players.items():
            if hasattr(player, "role"):
                # Extract base role type (e.g., 'executor' from 'executor_1')
                base_role = player.role.split("_")[0] if "_" in player.role else player.role
                if base_role == role_identifier:
                    return player

        return None

    def get_all_players_by_base_role(self, base_role: str) -> List[RoleBasedPlayer]:
        """Get all players that match a base role type (e.g., all executors).

        Args:
            base_role: The base role type to search for (e.g., 'executor', 'advisor')

        Returns:
            List of RoleBasedPlayer instances matching the base role
        """
        matching_players = []
        for node_id, player in self.players.items():
            if hasattr(player, "role"):
                # Extract base role type (e.g., 'executor' from 'executor_1')
                player_base_role = player.role.split("_")[0] if "_" in player.role else player.role
                if player_base_role == base_role:
                    matching_players.append(player)

        return matching_players


class ComputerGameScorer(GameScorer):
    def score_episode(self, episode_interactions: Dict):
        # Step 1: Extract key episode-level data
        success = episode_interactions.get("success", False)
        aborted = episode_interactions.get("aborted", False)
        request_count = episode_interactions.get("request_count", 0)
        request_count_parsed = episode_interactions.get("request_count_parsed", 0)
        request_count_violated = episode_interactions.get("request_count_violated", 0)
        player_stats = episode_interactions.get("player_stats", {})

        # Step 2: Log episode-level binary outcomes and raw counts
        self.log_episode_score(metrics.METRIC_SUCCESS, 1 if success else 0)
        self.log_episode_score(metrics.METRIC_LOSE, 1 if not success else 0)
        self.log_episode_score(metrics.METRIC_ABORTED, 1 if aborted else 0)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT, request_count)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_PARSED, request_count_parsed)
        self.log_episode_score(metrics.METRIC_REQUEST_COUNT_VIOLATED, request_count_violated)

        # Step 3: Calculate and log Efficiency and Robustness ratios
        if request_count > 0:
            efficiency = request_count_parsed / request_count
            robustness = 1 - (request_count_violated / request_count)
        else:
            efficiency = 0
            robustness = 0

        self.log_episode_score("Efficiency", efficiency)
        self.log_episode_score("Robustness", robustness)

        # Step 4: Calculate and log the final BENCH_SCORE
        # The harmonic mean is only valid if the game was a success
        if success and (efficiency + robustness) > 0:
            bench_score = (2 * efficiency * robustness) / (efficiency + robustness)
        else:
            bench_score = 0

        self.log_episode_score(metrics.BENCH_SCORE, bench_score)

        # Step 5: Log per-player scores for detailed episode-level diagnostics
        for player_id, stats in player_stats.items():
            p_requests = stats.get("requests", 0)
            p_parsed = stats.get("parsed", 0)
            p_violated = stats.get("violated", 0)

            if p_requests > 0:
                p_efficiency = p_parsed / p_requests
                p_robustness = 1 - (p_violated / p_requests)
            else:
                p_efficiency = 0
                p_robustness = 0

            self.log_episode_score(f"{player_id}_Efficiency", p_efficiency)
            self.log_episode_score(f"{player_id}_Robustness", p_robustness)
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
