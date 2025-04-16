import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from PIL import Image

from clemcore import backends
from clemcore.clemgame import Player, GameMaster, GameBenchmark
from src.master import (
    NetworkDialogueGameMaster,
    EdgeCondition,
    EdgeType,
    ConditionType,
)
from src.environment import Environment, create_osworld_environment
from src.player import RoleBasedPlayer
from src.message import MessageState, PlayerContextFormatter, PipeManager, PipeStage
from src.utils.registry.parsers import parsers, get_parser_metadata
from src.utils.constants import (
    DEFAULT_ENV_CONFIG,
    DEFAULT_HANDLER_TYPE,
)
from src.utils.general import TemporaryImageManager

logger = logging.getLogger(__name__)


class ComputerGameMaster(NetworkDialogueGameMaster):
    def __init__(
        self,
        name: str,
        path: str,
        experiment: Dict,
        player_models: List[backends.Model],
    ):
        super().__init__(name, path, experiment, player_models)
        self.experiment: str = experiment["name"]
        self.env: Environment = None
        self.game_instance: Dict = None
        self.game_config: Dict = None
        self.message_state = MessageState()
        self.player_context_formatter = None
        self.pipe_manager = PipeManager()

    def _on_setup(self, **game_instance) -> None:
        """Method executed at the start of the default setup method.

        Key Actions:
            - Sets up environment and loads initial observation + starts gameplay recording.
            - Constructs player interaction graph/network.
            - Sets up trigger pipeline (specific) 'parse func. -> after parse steps'

        Args:
            game_instance: Keyword arguments of the game_instance

        Returns:
            None: No return value
        """
        self.game_instance = game_instance
        self.game_config = self._prepare_game_config()
        self._initialize_environment()
        self.player_context_formatter = PlayerContextFormatter(
            game_config=self.game_config
        )
        self._build_graph()
        self._setup_after_parse_pipelines()

    def _prepare_game_config(self) -> Dict:
        """Prepare game configuration

        Returns:
            Dict: Complete game configuration dictionary
        """
        game_config = DEFAULT_ENV_CONFIG.copy()
        observation_type = game_config.get("observation_type", "a11y_tree")
        use_images = observation_type in ["screenshot", "screenshot_a11y_tree", "som"]
        require_a11y_tree = observation_type in [
            "a11y_tree",
            "screenshot_a11y_tree",
            "som",
        ]
        game_config["use_images"] = use_images
        game_config["require_a11y_tree"] = require_a11y_tree
        if use_images:
            game_config["temporary_image_manager"] = TemporaryImageManager()
        return game_config

    def _initialize_environment(self) -> None:
        """Initializes game environment with recording capabilities and retrieves the initial state observation.

        Returns:
            None: No return value

        Raises:
            RuntimeError: If environment or recording initialization fails
        """
        try:
            self.env = create_osworld_environment(**self.game_config)
            observation = self.env.reset(task_config=self.game_instance["task_config"])
            self.message_state.update(observation=observation)
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
        """Builds a dialogic-player network graph from the game instance configuration.

        Returns:
            None: No return value

        Raises:
            RuntimeError: If graph building fails
            ValueError: If player model is not available for role index
            KeyError: If condition type is invalid
        """
        try:
            graph_config = self.game_instance.get("graph")
            for node in graph_config.get("nodes", []):
                node_id = node.get("id")
                node_type = node.get("type")
                if not node_id or not node_type:
                    continue
                if node_type == "PLAYER":
                    role_index = node.get("role_index", 0)
                    if role_index >= len(self.player_models):
                        raise ValueError(
                            f"Player model not available for role index {role_index}"
                        )
                    roles = self.game_instance.get("roles", [])
                    role_config = roles[role_index]
                    role_name = role_config.get("name")
                    handler_type = role_config.get("handler_type", DEFAULT_HANDLER_TYPE)
                    allowed_components = role_config.get("allowed_components", [])
                    initial_prompt = role_config.get("initial_prompt")
                    player = RoleBasedPlayer(
                        self.player_models[role_index],
                        role=role_name,
                        handler_type=handler_type,
                        allowed_components=allowed_components,
                    )
                    self.add_player(
                        player=player, initial_prompt=initial_prompt, node_id=node_id
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
                    condition_type = condition_config.get("condition_type")
                    if (
                        not condition_type
                        or condition_type not in ConditionType._value2member_map_
                    ):
                        raise KeyError(
                            f"Invalid condition-type field: {condition_type}"
                        )
                    condition = EdgeCondition(
                        condition_type=condition_type, description=description
                    )
                    self.add_decision_edge(from_node, to_node, condition, description)
            anchor_node = graph_config.get("anchor_node")
            if anchor_node:
                self.set_anchor_node(anchor_node)
            logger.info("Graph building complete")
        except Exception as e:
            raise RuntimeError(f"Failed to build interaction graph: {str(e)}") from e

    def _setup_after_parse_pipelines(self) -> None:
        """Initialize processing pipelines for different parsers.

        Returns:
            None: No return value
        """
        self.pipe_manager.register_pipeline(
            "pyautogui_actions",
            [
                PipeStage(
                    self._execute_actions,
                    output_field="observation",
                    description="applies player actions to environment, producing state-based observations.",
                )
            ],
        )
        self.pipe_manager.register_pipeline(
            "done_or_fail",
            [
                PipeStage(
                    self._execute_actions,
                    output_field="observation",
                    description="applies player actions (either done or fail) to environment, producing final-state-based observation",
                )
            ],
        )

    def _does_game_proceed(self) -> bool:
        """Determine if the game should continue to the next turn.

        Returns:
            bool: False if game is completed or max steps reached, True otherwise
        """
        return self.current_node != "END"

    def _on_before_game(self):
        """Executed once at the start, before entering the play loop.

        Key Actions:
            - Adds the initial game-context to the anchor player
        """
        super()._on_before_game()
        assert self.current_node == self.anchor_node, (
            "Current node must be the anchor node at game start"
        )
        current_player = self.get_player_from_node(self.current_node)
        formatted_context = self.player_context_formatter.create_context_for(
            self.message_state, current_player
        )
        if "image" in formatted_context and formatted_context["image"]:
            self.set_context_for(
                current_player,
                formatted_context["content"],
                image=formatted_context["image"],
            )
        else:
            self.set_context_for(current_player, formatted_context["content"])
        logger.info(f"Set initial context for player at node {self.current_node}")

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        """Method executed after a player response has been parsed and validated.

        Key Actions:
            - Updates the message state with the parsed response
            - Sets context for the next node's player using the current message state

        Args:
            player: The Player instance that produced the response
            parsed_response: The parsed and valid response of the current player
        """
        if self.transition.next_node:
            next_player = self.get_player_from_node(self.transition.next_node)
            if next_player:
                formatted_context = self.player_context_formatter.create_context_for(
                    self.message_state, next_player
                )
                if "image" in formatted_context and formatted_context["image"]:
                    self.set_context_for(
                        next_player,
                        formatted_context["content"],
                        image=formatted_context["image"],
                    )
                else:
                    self.set_context_for(next_player, formatted_context["content"])
                logger.info(
                    f"Set context for next player at node {self.transition.next_node}"
                )

    def _validate_player_response(self, player: Player, response: str) -> bool:
        """Decide if a response should be added to the conversation history.

        Args:
            player: The Player instance for which the response is added
            response: The text content of the message to be added

        Returns:
            bool: True if the response is valid, False otherwise
        """
        print("::VALIDAION UTTERANCE::", response)
        if response is None:
            return False
        player_node = self.get_node_from_player(player)
        decision_edges = [
            (to_node, edge_data["condition"])
            for _, to_node, edge_data in self.graph.out_edges(player_node, data=True)
            if edge_data.get("type") == EdgeType.DECISION and edge_data.get("condition")
        ]
        # If no decision edges, check for exactly one standard edge
        if not decision_edges:
            standard_edges = [
                (to_node, edge_data)
                for _, to_node, edge_data in self.graph.out_edges(
                    player_node, data=True
                )
                if edge_data.get("type") == EdgeType.STANDARD
            ]
            if len(standard_edges) != 1:
                logger.warning(
                    f"Node {player_node} has {len(standard_edges)} standard edges, expected exactly 1"
                )
                return False
            return True
        # Check each decision edge's condition
        for to_node, condition in decision_edges:
            validation_result = condition.validate(response)
            # Case 1: Valid and intended format - successful validation
            if validation_result.is_valid and validation_result.intended_format:
                return True
            # Case 2: Invalid but intended format - validation failed for intended format
            elif not validation_result.is_valid and validation_result.intended_format:
                if validation_result.error:
                    logger.warning(
                        f"Validation failed for edge to {to_node}: {validation_result.error.message}"
                    )
                return False
            # Case 3: Not intended format - continue to next edge
            elif not validation_result.intended_format:
                continue
        # Finally, resort to a fallback procedure (TODO)
        logger.warning(
            "Validation failed, response did not comply with available conditions for connecting edges"
        )
        return False

    def _parse_response_for_decision_routing(
        self, player: Player, response: str
    ) -> Tuple[str, bool, Optional[str], Optional[Any]]:
        """Parse player response and evaluate decision edge conditions.

        Key Actions:
            1. Parse the player's response for relevant content
            2. Evaluate decision edge conditions based on the parsed content
            3. Determine which decision edge (if any) should be taken

        Args:
            player: The Player instance that produced the response
            response: The text content of the response

        Returns:
            Tuple[str, bool, Optional[str], Optional[Any]]: Modified response, logging flag, next node ID, extracted content
        """
        player_node = self.get_node_from_player(player)
        decision_edges = [
            (to_node, edge_data["condition"])
            for _, to_node, edge_data in self.graph.out_edges(player_node, data=True)
            if edge_data.get("type") == EdgeType.DECISION and edge_data.get("condition")
        ]
        print("::DECISION_EDGES::", decision_edges)
        if not decision_edges:
            return response, False, None, None
        # Evaluate each decision edge condition
        for to_node, condition in decision_edges:
            print("::TO_NODE::", to_node, "::CONDITION::", condition)
            try:
                parse_result = condition.parse(response)
                print(
                    "::IS_SUCCESSFUL, EXTRACTED_CONTENT::",
                    parse_result.is_successful,
                    parse_result.content,
                )
                if parse_result.is_successful and parse_result.content:
                    parser_id = None
                    parse_func_name = condition.function_pair.parse_func.__name__
                    for pid in parsers:
                        if (
                            parse_func_name.startswith(f"parse_{pid}")
                            or pid in parse_func_name
                        ):
                            parser_id = pid
                            break
                    if parser_id:
                        metadata = get_parser_metadata(parser_id)
                        target_field = metadata.get("target_field")
                        if target_field and hasattr(self.message_state, target_field):
                            self.message_state.update(
                                **{target_field: parse_result.content}
                            )
                        success, result = self.pipe_manager.execute_pipeline(
                            parser_id=parser_id,
                            content=parse_result.content,
                            message_state=self.message_state,
                        )
                        if success:
                            logger.info(
                                f"Processing pipeline executed for parser {parser_id}"
                            )
                    logger.info(
                        f"Decision edge condition met: {self.current_node} → {to_node}"
                    )
                    return (
                        response,
                        True,
                        to_node,
                        (
                            parse_result.content,
                            result if "result" in locals() else None,
                        ),
                    )
            except Exception as e:
                logger.error(
                    f"Error evaluating condition for edge to {to_node}: {str(e)}"
                )
        return response, False, None, None

    def _execute_actions(
        self, content: str
    ) -> Optional[Dict[str, Union[str, Image.Image, Dict]]]:
        """Execute either pyautogui or computer13 actions and record observations (upon environment-state change).

        Args:
            content: Parser extracted actions typically either pyautogui python-code (as str.), or computer13 actions in JSON (as str.)

        Returns:
            Optional[Dict]: Observation dictionary or None if execution fails
        """
        print("::EXECUTE ACTION::", content)
        try:
            if not content:
                logger.error("No actions to execute")
                return None
            observation = None
            for action in content:
                try:
                    observation, reward, done, info = self.env.step(
                        action, self.game_config.get("sleep_after_execution", 0.0)
                    )
                    print(reward, done, info)
                    if observation is None:
                        logger.error("Recieved None observation after game execution")
                        return None
                    self.current_observation = observation
                    if done:
                        logger.info("Game termination signal recieved (done=True)")
                        break
                except Exception as e:
                    logger.error(f"Failed to execute action {str(action)}: {str(e)}")
            return observation
        except Exception as e:
            logger.error(f"Action execution failed: {str(e)}")
            return None


class ComputerGameBenchmark(GameBenchmark):
    def create_game_master(
        self, experiment: Dict, player_models: List[backends.Model]
    ) -> GameMaster:
        return ComputerGameMaster(
            self.game_name, self.game_path, experiment, player_models
        )


if __name__ == "__main__":

    def load_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    experiments = load_json("./in/instances.json")
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"][0]
    master = ComputerGameMaster("computergame", None, experiment_1, ["mock", "mock"])
    master.setup(**game_1)
