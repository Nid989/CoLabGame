import json
import re
import logging
import numpy as np
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
from src.utils.constants import DEFAULT_HANDLER_TYPE
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
        self.lose: bool = False
        self.success: bool = False
        self.request_count: int = 0
        self.request_count_parsed: int = 0
        self.request_count_violated: int = 0
        self.invalid_response: bool = False

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
        game_config.update(
            {"use_images": use_images, "require_a11y_tree": require_a11y_tree}
        )
        if use_images:
            game_config["temporary_image_manager"] = TemporaryImageManager()
        self.game_config = game_config

    def _initialize_formatter(self) -> None:
        """Initialize the player context formatter with the current game configuration."""
        self.player_context_formatter = PlayerContextFormatter(
            game_config=self.game_config
        )

    def _initialize_environment(self) -> None:
        """Initializes game environment with recording capabilities and retrieves the initial state observation.

        Raises:
            RuntimeError: If environment or recording initialization fails
        """
        try:
            self.env = EnvironmentFactory.create_environment(
                self.environment_type, **self.game_config
            )
            observation = self.env.reset(task_config=self.game_instance["task_config"])
            self.message_state.update(observation=observation)
            print(self.message_state.preview())
            if not self.env.start_recording():
                raise RuntimeError("Failed to start environment recording")
        except Exception as e:
            error_message = (
                f"Environment initialization failed: {str(e)}"
                if "recording" not in str(e).lower()
                else f"Recording initialization failed: {str(e)}"
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
            graph_config = self.game_instance.get("graph")
            roles = self.game_instance.get("roles", [])
            for node in graph_config.get("nodes", []):
                node_id = node.get("id")
                node_type = node.get("type")
                if node_type != "PLAYER" or not node_id:
                    continue
                role_index = node.get("role_index", 0)
                if not (0 <= role_index < len(self.player_models)):
                    raise ValueError(
                        f"Player model not available for role index {role_index}"
                    )
                role_config = roles[role_index]
                player = RoleBasedPlayer(
                    self.player_models[role_index],
                    role=role_config.get("name"),
                    handler_type=role_config.get("handler_type", DEFAULT_HANDLER_TYPE),
                    allowed_components=role_config.get("allowed_components", []),
                )
                self.add_player_to_graph(
                    player=player,
                    initial_prompt=role_config.get("initial_prompt"),
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
                    condition = EdgeCondition(
                        message_type=message_type, description=description
                    )
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
            bool: False if game is completed or aborted, True otherwise
        """
        if self.invalid_response:
            self.aborted = True
            self.lose = True
            return False
        max_rounds = self.game_config.get("max_rounds", 5)
        if self.current_round + 1 >= max_rounds:
            self.aborted = True
            self.lose = True
            self.log_to_self("failure", f"Maximum rounds {max_rounds} reached")
            return False
        max_transitions_per_round = self.game_config.get("max_transitions_per_round", 5)
        if self.transition.total_transitions + 1 >= max_transitions_per_round:
            self.aborted = True
            self.log_to_self(
                "failure",
                f"Maximum transitions per round {max_transitions_per_round} reached",
            )
            return False
        if self._current_node == "END":
            self.game_config["temporary_image_manager"].cleanup()
            return False
        return True

    def _set_context_for(
        self, player: RoleBasedPlayer, formatted_context: Dict
    ) -> None:
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
        assert self._current_node == self.anchor_node, (
            "Current node must be the anchor node at game start"
        )
        context = self.player_context_formatter.create_context_for(
            self.message_state, self._current_player
        )
        if context is None:
            logger.debug(
                "No context generated for player; skipping inital context setup."
            )
            return
        self._set_context_for(self._current_player, context)
        logger.info(f"Set initial context for player at node {self._current_node}")

    def _on_after_game(self):
        """Executed once at the end, after exiting the play loop.

        Hook: Modify this method for game-specific functionality.

        This method is useful to process and log/record overall game results.
        """
        self.log_to_self("game_end", "Game completed")
        if (
            hasattr(self, "game_config")
            and "temporary_image_manager" in self.game_config
        ):
            self.game_config["temporary_image_manager"].cleanup()

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

        def construct_response(
            content: Any, message_type: MessageType, data: Dict[Any, Any]
        ) -> Result:
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
            parsed_response_dict["content"] = (
                content  # Preserves List[str], List[Dict], or str
            )
            parsed_response = json.dumps(parsed_response_dict)
            return Ok(parsed_response)

        # Chain the operations using Result monad
        # TODO: the `handle_json_content` method is quite inefficient, as it provide `action_space` everytime, when it is not needed.
        result = (
            Ok(response)
            | (
                lambda x: Ok(self.extract_json_codeblock(x))
                if isinstance(x, str)
                else Error(ParseError(reason="Expected string input", response=str(x)))
            )
            | (
                lambda x: Ok(x[1]) if x[0] else Error(x[1])
            )  # Ok(Dict) or Error(ParseError)
            | (lambda x: Ok((self.check_json_message(x), x)))
            | (
                lambda x: Ok((x[0][1], x[1])) if x[0][0] else Error(x[0][1])
            )  # Ok((MessageType, Dict)) or Error(ParseError)
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
            | (
                lambda x: Ok((x[0][1], x[1], x[2])) if x[0][0] else Error(x[0][1])
            )  # Ok((content, MessageType, Dict)) or Error(ParseError/GameError)
            | (lambda x: construct_response(x[0], x[1], x[2]))  # Ok(str)
        )

        # Extract final result and raise errors
        if isinstance(result, Ok):
            return result.value
        else:
            # TODO: Implement custom error handling for exceptions from extract_json_codeblock, check_json_message, or handle_json_content
            raise result.error

    def extract_json_codeblock(
        self, text: str
    ) -> Tuple[bool, Optional[Dict[Any, Any]] | Exception]:
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
                return False, ParseError(
                    reason="No code block found in the input text", response=text
                )

            json_content = match.group(1).strip()

            if not json_content:
                return False, ParseError(reason="Empty code block found", response=text)

            result = json.loads(json_content)

            if not isinstance(result, dict):
                return False, ParseError(
                    reason="Parsed content is not a JSON object", response=json_content
                )

            logger.info("Successfully parsed JSON from code block")
            return True, result

        except json.JSONDecodeError as e:
            return False, ParseError(
                reason=f"Invalid JSON format: {str(e)}", response=text
            )

    def check_json_message(
        self, data: Dict[Any, Any]
    ) -> Tuple[bool, Optional[MessageType] | Exception]:
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
                return False, ParseError(
                    reason=f"Missing required keys: {missing_keys}", response=str(data)
                )

            try:
                message_type = MessageType[data["type"]]
            except KeyError:
                valid_types = ", ".join(mt.name for mt in MessageType)
                return False, ParseError(
                    reason=f"Invalid message type: {data['type']}. Must be one of {valid_types}",
                    response=str(data),
                )

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

            # TODO: Add player ID validation for 'from' field
            # Example: validate_player_id(json_data["from"])

            # Validate 'to' field if present (basic string check)
            if "to" in data:
                # TODO: Add player ID validation for 'to' field
                # Example: validate_player_id(json_data["to"])
                pass

            # Basic content type validation
            if (
                message_type == MessageType.EXECUTE
                and self.game_config.get("action_space") == "computer13"
            ):
                if not isinstance(data["content"], list) or not all(
                    isinstance(item, dict) for item in data["content"]
                ):
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
            message_type_str = (
                message_type.name
                if isinstance(message_type, MessageType)
                else message_type
            )

            # Call registry to process content
            success, result = process_content(
                data, message_type_str, environment_type, action_space
            )

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

    def _execute_actions(
        self, actions: List[Union[str, Dict]]
    ) -> Dict[str, Union[str, Image.Image, Dict]]:
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
                observation, reward, done, info = self.env.step(
                    action, self.game_config.get("sleep_after_execution", 0.0)
                )
                if observation is None:
                    raise GameError(
                        reason="Received None observation after action execution"
                    )
                if done:
                    logger.info("Game termination signal received (done=True)")
                    break
            except Exception as e:
                raise GameError(
                    reason=f"Failed to execute action {str(action)}: {str(e)}"
                )

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
        print("current_node", current_node)
        next_node = None

        # First, check decision edges for a valid transition
        decision_edges = self._get_decision_edges(current_node)
        if decision_edges:
            if "to" in data:
                target_node = data["to"]
                for to_node, condition in decision_edges:
                    print("to_node", to_node)
                    print("target_node", target_node)
                    print("message_type.name", message_type.name)
                    print("condition.message_type", condition.message_type)
                    print("condition", condition.validate(message_type.name))
                    if to_node == target_node and condition.validate(message_type.name):
                        next_node = target_node
                        break
                if next_node is None:
                    raise RuleViolationError(
                        f"No valid transition found to target node {target_node} with message type {message_type.name}"
                    )
            else:
                # Check for self-loop or staying at current node
                for to_node, condition in decision_edges:
                    if to_node == current_node and condition.validate(
                        message_type.name
                    ):
                        next_node = current_node
                        break
                if next_node is None:
                    raise RuleViolationError(
                        f"No valid self-loop transition found for message type {message_type.name} at node {current_node}"
                    )

        # If no decision edge is found, fallback to standard edges
        if next_node is None:
            standard_edges = self._get_standard_edges(current_node)
            if standard_edges:
                next_node = standard_edges[0][
                    0
                ]  # Take the first standard edge target node
            else:
                raise RuleViolationError(
                    f"No valid transition (decision or standard) found for message type {message_type.name} from node {current_node}"
                )

        # Step 3: Update game state with the validated transition
        self._update_round_tracking(current_node, next_node)
        self.transition.next_node = next_node
        logger.info(
            f"Transitioned from {current_node} to {next_node} based on message type {message_type.name}"
        )

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

        print(self.message_state.preview())

        # Step 5: Prepare context for the next player if transition occurred
        if self.transition.next_node:
            next_player = self.get_player_from_node(self.transition.next_node)
            if next_player:
                formatted_context = self.player_context_formatter.create_context_for(
                    self.message_state, next_player
                )
                self._set_context_for(next_player, formatted_context)
                self.message_state.reset(preserve=["observation"])
                logger.info(
                    f"Set context for next player at node {self.transition.next_node}"
                )

    def compute_episode_score(self):
        """
        Computes the score for the current episode.

        Returns:
            float: A score of 1.0 for success or 0.0 for failure.
        """
        result = self.env.evaluate()
        self.log_to_self("episode_score", result)
        return result

    def _on_parse_error(self, error: ParseError):
        """Hook to implement consequences for parsing errors e.g. prepare re-prompting or set game state to abort."""
        self.log_to_self("parse_error", str(error))
        self.invalid_response = True
        self.request_count_violated += 1

    def _on_game_error(self, error: GameError):
        """Hook to implement consequences for game errors e.g. prepare re-prompting or set game state to failure."""
        self.log_to_self("game_error", str(error))
        self.invalid_response = True
        self.request_count_violated += 1


class ComputerGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        """Temporary method to compute scores for Computer Game."""

        aborted = True
        success = False
        all_turn_scores = []
        # Events are logged properly.
        # I need to iterate over the events and identify the types and their respected values, and correspond them with METRICS available.
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score_dict = {
                "request_count": 0,
                "request_count_parsed": 0,
                "request_count_violated": 0,
            }

            for event in turn:
                action = event["action"]
                turn_score_dict["request_count"] += 1
                if action["type"] == "validation_error":
                    turn_score_dict["request_count_violated"] += 1
                elif action["type"] == "parsed_response":
                    turn_score_dict["request_count_parsed"] += 1
                elif action["type"] == "episode_score":
                    aborted = False
                    success = bool(action["content"] == 1.0)

            self.log_turn_score(
                turn_idx,
                metrics.METRIC_REQUEST_COUNT_VIOLATED,
                turn_score_dict["request_count_violated"],
            )
            self.log_turn_score(
                turn_idx,
                metrics.METRIC_REQUEST_COUNT_PARSED,
                turn_score_dict["request_count_parsed"],
            )
            self.log_turn_score(
                turn_idx, metrics.METRIC_REQUEST_COUNT, turn_score_dict["request_count"]
            )
            all_turn_scores.append(turn_score_dict)

        ep_request_count = 0
        ep_request_count_violated = 0
        ep_request_count_parsed = 0
        for s in all_turn_scores:
            ep_request_count += s["request_count"]
            ep_request_count_violated += s["request_count_violated"]
            ep_request_count_parsed += s["request_count_parsed"]

        self.log_episode_score(metrics.METRIC_REQUEST_COUNT, ep_request_count)
        self.log_episode_score(
            metrics.METRIC_REQUEST_COUNT_VIOLATED, ep_request_count_violated
        )
        self.log_episode_score(
            metrics.METRIC_REQUEST_COUNT_PARSED, ep_request_count_parsed
        )

        if aborted:
            self.log_episode_score(metrics.METRIC_ABORTED, 1)
            self.log_episode_score(metrics.METRIC_SUCCESS, 0)
            self.log_episode_score(metrics.METRIC_LOSE, 0)
            self.log_episode_score(metrics.BENCH_SCORE, np.nan)
            self.log_episode_score("Player Score", np.nan)
        else:
            self.log_episode_score(metrics.METRIC_ABORTED, 0)
            self.log_episode_score(metrics.METRIC_SUCCESS, int(success))
            self.log_episode_score(metrics.METRIC_LOSE, int(not success))
            self.log_episode_score(metrics.BENCH_SCORE, 100)
            self.log_episode_score("Player Score", 100)


class ComputerGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(
        self, experiment: Dict, player_models: List[backends.Model]
    ) -> GameMaster:
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
