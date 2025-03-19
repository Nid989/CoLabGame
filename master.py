import json
import logging
from typing import Dict, List, Tuple

from clemcore import backends
from clemcore.clemgame import (
    Player,
    DialogicNetworkGameMaster,
    GameMaster,
    GameBenchmark,
)
from clemcore.clemgame.master import EdgeCondition
from game import ComputerGame, RoleBasedPlayer
from utils import extract_actions
from parsing import parse_function_registry
from constants import LogType, DEFAULT_ENV_CONFIG, DEFAULT_ROLE, ACTION_RESULT_TEMPLATE

logger = logging.getLogger(__name__)


class ComputerGameMaster(DialogicNetworkGameMaster):
    def __init__(
        self,
        name: str,
        path: str,
        experiment: Dict,
        player_models: List[backends.Model],
    ):
        super().__init__(name, path, experiment, player_models)

        self.experiment: str = experiment["name"]
        self.game: ComputerGame = None
        self.game_instance: Dict = None
        self.terminated: bool = False

    # NOTE: (Confused) log_to_self + self.terminated = True; does not seem ideal for setup related issues, good idea will be to replace it with logger.error (if encountered)
    # however, we need to set self.terminated = True since game must not proceed further.
    def _on_setup(self, **game_instance) -> None:
        """Method executed at the start of the default setup method.
        - Sets up environment and loads initial observation.
        - Starts gameplay recording.
        - Constructs player interaction network.
        """
        self.game_instance = game_instance

        self._initialize_environment()
        self._build_graph()

    def _initialize_environment(self) -> None:
        """Initializes game environment with recording capabilities"""
        try:
            env_config = DEFAULT_ENV_CONFIG.copy()
            self.game = ComputerGame(
                **env_config, game_instance=self.game_instance["task_config"]
            )
            self.current_observation = self.game.env.reset(
                task_config=self.game_instance["task_config"]
            )
            if not self.game.env.start_recording():
                raise RuntimeError("Failed to start environment recording")

        except Exception as e:
            self.terminated = True
            error_message = (
                f"Environment initialization failed: {str(e)}"
                if "recording" not in str(e).lower()
                else f"Recording initialization failed: {str(e)}"
            )
            self.log_to_self(LogType.SETUP_ERROR.value, error_message)
            self.log_to_self(
                LogType.GAME_STATE.value,
                "Game terminated: failed to initialize game environment",
            )

    def _build_graph(self) -> None:
        """Builds a dialogic-player network graph from the game instance configuration."""
        try:
            graph_config = self.game_instance.get("graph")
            if not graph_config:
                self.terminated = True
                self.log_to_self(
                    LogType.SETUP_ERROR.value,
                    "Game instance missing required 'graph' field",
                )
                self.log_to_self(
                    LogType.GAME_STATE.value,
                    "Game terminated: missing graph configuration",
                )
                return

            for node in graph_config.get("nodes", []):
                node_id = node.get("id")
                node_type = node.get("type")
                if not node_id or not node_type:
                    continue
                if node_type == "PLAYER":
                    role_index = node.get("role_index", 0)
                    if role_index >= len(self.player_models):
                        self.log_to_self(
                            LogType.SETUP_ERROR.value,
                            f"Player model not available for role index {role_index}",
                        )
                        continue
                    roles = self.game_instance.get("roles", [])
                    role = (
                        roles[role_index] if role_index < len(roles) else DEFAULT_ROLE
                    )
                    player = RoleBasedPlayer(self.player_models[role_index], role=role)
                    self.add_player(player, node_id)

            for index, edge in enumerate(graph_config.get("edges", [])):
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
                    parse_function_id = condition_config.get("parse_function_id")
                    if (
                        not parse_function_id
                        or parse_function_id not in parse_function_registry
                    ):
                        self.log_to_self(
                            LogType.SETUP_ERROR.value,
                            f"Invalid parse function ID: {parse_function_id}",
                        )
                        continue
                    parse_func = parse_function_registry[parse_function_id]
                    condition = EdgeCondition(
                        parse_func=parse_func, description=description
                    )
                    self.add_decision_edge(from_node, to_node, condition, description)

            anchor_node = graph_config.get("anchor_node")
            if anchor_node:
                self.set_anchor_node(anchor_node)
            logger.info("Graph building complete")

        except Exception as e:
            self.terminated = True
            self.log_to_self(
                LogType.SETUP_ERROR.value, f"Error building graph: {str(e)}"
            )
            self.log_to_self(
                LogType.GAME_STATE.value,
                "Game terminated: failed to build interaction graph",
            )

    def _does_game_proceed(self) -> bool:
        """Determine if the game should continue to the next turn.
        Returns:
            bool: False if game is completed or max steps reached, True otherwise
        """
        return not self.terminated and self.current_turn < self.game.max_steps

    def _on_before_game(self) -> None:
        """Initializes game instruction and adds system message to all players' dialogue history."""
        self.instruction = self.game_instance.get("instruction")
        if not self.instruction:
            self.terminated = True
            self.log_to_self(
                LogType.SETUP_ERROR.value,
                "Game instance missing required 'instruction' field",
            )
            self.log_to_self(
                LogType.GAME_STATE.value, "Game terminated: missing instruction"
            )
            return

        for player in self.get_players():
            initial_context = self.game.prompt_handler._get_turn_context(
                self.current_observation,
                turn=self.current_turn,
                instruction=self.instruction,
            )
            image = initial_context.get("image", [])
            self.add_user_message(player, initial_context["content"], image=image)
            logger.info("Initial instruction prompt added for %s", player.descriptor)

    def _on_before_turn(self, turn_idx: int):
        """Updates the game's turn counter and player context before each turn.

        Args:
            turn_idx: The current turn index.
        """
        self.game._current_turn = turn_idx

        # Skip player context update for turn 0 as it's handled in _on_before_game
        if turn_idx > 0:
            for player in self.get_players():
                self._update_player_context(player)

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """Basic format validation for player responses.
        Args:
            utterance (str): Response/utterance to validate
        Returns:
            bool: Validation result
        """
        if not utterance or not isinstance(utterance, str):
            self.terminated = True
            self.log_to_self(
                LogType.VALIDATION.value,
                "Invalid response: empty or non-string message",
            )
            self.log_to_self(
                LogType.GAME_STATE.value,
                "Game terminated: received invalid/empty response",
            )
            return False

        return True

    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        """Extracts executable actions from player response.
        Args:
            utterance (str): Response text to parse
        Returns:
            Tuple[str, bool]: (original utterance, parsing success)
        Note: Extracted actions stored in self._temp_extracted_actions
        """
        try:
            masks = (
                self.current_observation.get("masks")
                if self.game.observation_type == "som"
                else None
            )
            extracted_actions = extract_actions(
                utterance, self.game.observation_type, self.game.action_space, masks
            )
            self._temp_extracted_actions = extracted_actions
            self.log_to_self(
                LogType.ACTION_INFO.value,
                f"Extracted actions: {', '.join(str(a) for a in extracted_actions)}",
            )
            return utterance, True

        except ValueError as e:
            self._temp_extracted_actions = []
            self.terminated = True
            self.log_to_self(
                LogType.ACTION_FAIL.value,
                f"Error: {str(e)}, Response preview: {utterance[:100]}",
            )
            self.log_to_self(
                LogType.GAME_STATE.value, "Game terminated: failed to parse actions"
            )
            return utterance, False

    def _execute_action(self, action) -> bool:
        """Execute a single action and process its results.

        Args:
            action: The action to execute

        Returns:
            bool: True if execution was successful, False if game should terminate
        """
        try:
            self.current_observation, reward, done, info = self.game.env.step(
                action, self.game.sleep_after_execution
            )

            if self.current_observation is None:
                print("none current observation")
                self.terminated = True
                self.log_to_self(
                    LogType.ACTION_FAIL.value,
                    "Received None observation after action execution",
                )
                self.log_to_self(
                    LogType.GAME_STATE.value,
                    "Game terminated: invalid observation state",
                )
                return False

            action_result = ACTION_RESULT_TEMPLATE.format(
                action=str(action), reward=reward, done=done
            )
            print(action_result)
            if info:
                action_result += f", Additional info: {str(info)}"
            self.log_to_self(LogType.ACTION_EXEC.value, action_result)

            if done:
                self.terminated = True
                self.log_to_self(
                    LogType.GAME_STATE.value,
                    "Game termination signal received (done=True)",
                )
                return False

            return True

        except Exception as e:
            print(e)
            self.terminated = True
            self.log_to_self(
                LogType.ACTION_FAIL.value,
                f"Failed to execute action {str(action)}: {str(e)}",
            )
            self.log_to_self(
                LogType.GAME_STATE.value, "Game terminated: action execution failed"
            )
            return False

    def _execute_all_actions(self, extracted_actions):
        """Execute all extracted actions in sequence.

        Args:
            extracted_actions: List of actions to execute
        """
        if not extracted_actions:
            self.terminated = True
            self.log_to_self(
                LogType.ACTION_FAIL.value, "No actions extracted from response"
            )
            self.log_to_self(
                LogType.GAME_STATE.value, "Game terminated: no actions to execute"
            )
            return

        for action in extracted_actions:
            print(action)
            if not self._execute_action(action):
                break

    def _update_player_context(self, player: Player) -> None:
        """Updates player's context with the current observation.

        Args:
            player (Player): The player to update context for
        """
        try:
            turn_context = self.game.prompt_handler._get_turn_context(
                self.current_observation, turn=self.current_turn
            )
            message = turn_context["content"]
            image = turn_context.get("image", [])
            self.add_user_message(player, message, image=image)
        except Exception as e:
            self.terminated = True
            self.log_to_self(
                LogType.TURN_FAIL.value, f"Failed to update player context: {str(e)}"
            )
            self.log_to_self(
                LogType.GAME_STATE.value,
                "Game terminated: failed to update player context",
            )

    def _after_add_player_response(self, player: Player, utterance: str):
        """Updates interaction history with response and extracted actions, then executes actions.

        Args:
            utterance (str): Validated response text
        Note: Game terminates on history update failure or action execution failure
        """
        if self.terminated:
            return

        extracted_actions = getattr(self, "_temp_extracted_actions", [])

        print(extracted_actions)

        try:
            self.game.prompt_handler.update_interaction_history(
                thought=utterance,
                action=extracted_actions,
                obs=self.current_observation,
            )

            self.log_to_self(
                LogType.TURN_PLAN.value,
                f"Turn {self.current_turn}: Thought preview: {utterance[:100]}, Actions: {', '.join(str(a) for a in extracted_actions)}",
            )

        except Exception as e:
            self.terminated = True
            self.log_to_self(
                LogType.ACTION_FAIL.value,
                f"Failed to update interaction history: {str(e)}",
            )
            self.log_to_self(
                LogType.GAME_STATE.value,
                "Game terminated: failed to update interaction history",
            )
            return

        if self.terminated:
            return

        try:
            self._execute_all_actions(extracted_actions)
        except Exception as e:
            self.terminated = True
            self.log_to_self(
                LogType.TURN_FAIL.value, f"Turn {self.current_turn} failed: {str(e)}"
            )
            self.log_to_self(
                LogType.GAME_STATE.value, "Game terminated: turn execution failed"
            )
        finally:
            if hasattr(self, "_temp_extracted_actions"):
                del self._temp_extracted_actions

    def add_message(
        self, player: Player, utterance: str, role: str, image: List[str] = None
    ):
        """Adds a message to the conversation history.
        Args:
            player: The Player instance that produced the message.
            utterance: The text content of the message to be added.
            role: The chat/instruct conversation role ('user', 'assistant', or 'system')
            image: Optional list of image data or paths to include with the message.
        """
        message = {"role": role, "content": utterance}
        if image and len(image) > 0:
            message["image"] = image
        history = self.messages_by_names[player.descriptor]
        history.append(message)

    def add_user_message(self, player: Player, utterance: str, image: List[str] = None):
        """Adds a message with the 'user' role to the conversation history.
        Args:
            player: The Player instance that produced the message.
            utterance: The text content of the message to be added.
            image: Optional list of image data or paths to include with the message.
        """
        self.add_message(player, utterance, role="user", image=image)


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
