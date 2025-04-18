import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from PIL import Image

from clemcore import backends
from clemcore.clemgame import Player, GameMaster, GameBenchmark
from src.master import (
    NetworkDialogueGameMaster,
    EdgeCondition,
    ConditionType,
)
from src.environment import Environment, create_osworld_environment
from src.player import RoleBasedPlayer
from src.message import MessageState, PlayerContextFormatter, PipeManager, PipeStage
from src.utils.registry.parsers import parsers, get_parser_metadata
from src.utils.registry.validators import raise_unrecognized_format_error
from src.utils.constants import DEFAULT_GAME_CONFIG, DEFAULT_HANDLER_TYPE, LogType
from src.utils.general import TemporaryImageManager

logger = logging.getLogger(__name__)


class ComputerGame(NetworkDialogueGameMaster):
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
        self.aborted: bool = False
        self.failure: bool = False
        self.success: bool = False
        self.request_count: int = 0
        self.parsed_request_count: int = 0
        self.violated_request_count: int = 0
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
        self._prepare_game_config()
        self._initialize_formatter()
        self._initialize_environment()
        self._build_graph()
        self._setup_after_parse_pipelines()

    def _prepare_game_config(self) -> Dict:
        """Prepare game configuration dictionary"""
        game_config = DEFAULT_GAME_CONFIG.copy()
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
                self.add_player(
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
                    condition_type = condition_config.get("condition_type")
                    if condition_type not in ConditionType._value2member_map_:
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
        """Initialize processing pipelines for different parsers."""
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
            bool: False if game is completed or aborted, True otherwise
        """
        # NOTE: might want to change the below check?
        if self.invalid_response and self.aborted:
            self.failure = True
            return False
        max_rounds = self.game_config.get("max_rounds", 5)
        if self.current_round + 1 >= max_rounds:
            self.aborted = True
            self.failure = True
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
        if self.current_node == "END":
            return False

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
        assert self.current_node == self.anchor_node, (
            "Current node must be the anchor node at game start"
        )
        current_player = self.get_player_from_node(self.current_node)
        formatted_context = self.player_context_formatter.create_context_for(
            self.message_state, current_player
        )
        self._set_context_for(current_player, formatted_context)
        logger.info(f"Set initial context for player at node {self.current_node}")

    def _validate_player_response(self, player: Player, response: str) -> bool:
        """Decide if player response is valid.

        A valid response enables the invocation of additional methods, such as _parse_response and _on_valid_parse_response.

        Args:
            player: The player that gave the response.
            response: The response of the current player.

        Returns:
            bool: True, if the response is fine. Otherwise, False.
        """

        def handle_retry(error_msg: str):
            if player.retries < self.game_config["max_retries"]:
                player.retries += 1
                self.message_state.update(tagged_content={"error": error_msg})
                formatted_context = self.player_context_formatter.create_context_for(
                    self.message_state, player
                )
                self._set_context_for(player, formatted_context)
            else:
                # player exceeded invalid-response <-> re-prompt attempts thus abort game.
                self.aborted = True

        self.request_count += 1
        self.invalid_response = False
        if response is None:
            return False
        player_node = self.get_node_from_player(player)
        decision_edges = self._get_decision_edges(player_node)
        # If no decision edges, check for exactly one standard edge
        if not decision_edges:
            return bool(self._get_standard_edges(player_node))
        # Check each decision edge's condition
        for to_node, condition in decision_edges:
            result = condition.validate(response)
            # Case 1: Valid and intended format - successful validation
            if result.is_valid and result.intended_format:
                return True
            # Case 2: Invalid but intended format - validation failed for intended format
            elif not result.is_valid and result.intended_format:
                self.log_to_self(LogType.VALIDATION_ERROR, result.error.get_dict())
                self.invalid_response = True
                self.violated_request_count += 1
                handle_retry(result.error.message)
                return False
            # Case 3: Not intended format - continue to next edge
            elif not result.intended_format:
                continue

        # Fallback procedure
        # TODO: deal with how to handle the observation <-> pyautogui or computer13 substitution.
        error = raise_unrecognized_format_error(player.allowed_components)
        self.log_to_self(LogType.VALIDATION_ERROR, error.get_dict())
        self.invalid_response = True
        self.violated_request_count += 1
        handle_retry(error.message)
        # TODO: Should we expand how we handle the fallback procedure?
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
        self.parsed_request_count += 1
        player_node = self.get_node_from_player(player)
        decision_edges = self._get_decision_edges(player_node)
        if not decision_edges:
            return response, False, None, None
        # Evaluate each decision edge condition
        for to_node, condition in decision_edges:
            try:
                parse_result = condition.parse(response)
                if not (parse_result.is_successful and parse_result.content):
                    continue
                parse_func_name = condition.function_pair.parse_func.__name__
                parser_id = next(
                    (
                        pid
                        for pid in parsers
                        if parse_func_name.startswith(f"parse_{pid}")
                        or pid in parse_func_name
                    ),
                    None,
                )
                result = None
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
                    logger.info(f"Processing pipeline executed for parser {parser_id}")

                logger.info(
                    f"Decision edge condition met: {self.current_node} → {to_node}"
                )
                return response, True, to_node, (parse_result.content, result)
            except Exception as e:
                logger.error(f"Error evaluating condition for edge to {to_node}: {e}")

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
                self._set_context_for(next_player, formatted_context)
                # self.message_state.reset(preserve=["observation"]) # Somehow this does not work.
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
        return result


class ComputerGameBenchmark(GameBenchmark):
    def create_game_master(
        self, experiment: Dict, player_models: List[backends.Model]
    ) -> GameMaster:
        return ComputerGame(self.game_name, self.game_path, experiment, player_models)


if __name__ == "__main__":

    def load_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    experiments = load_json("./in/instances.json")
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"][0]
    master = ComputerGame("computergame", None, experiment_1, ["mock", "mock"])
    master.setup(**game_1)

# _on_before_round -> something should be logged via log_to_self
# _does_game_proceed -> add more depenedent variables including aborted, lost etc & log_to_self once proceeding-of-game-stops
# _validate_player_response -> obviously we log_to_self `the ValidationErros` but something more?
# _parse_response -> log_to_self successful outcomes.
# _on_valid_response -> log_to_self not sure think
