import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from PIL import Image
import os

from clemcore import backends
from clemcore.clemgame import Player, GameMaster, GameBenchmark
from src.game_master import (
    NetworkDialogueGameMaster,
    EdgeCondition,
    NodeType,
    EdgeType,
    ConditionType,
)
from src.game import ComputerGame, RoleBasedPlayer
from src.utils.registry.parsers import parsers, get_parser_metadata
from src.utils.constants import (
    DEFAULT_ENV_CONFIG,
    DEFAULT_HANDLER_TYPE,
)

logger = logging.getLogger(__name__)


@dataclass
class MessageState:
    """Dynamic state container for message components with reset capability"""

    observation: Optional[Dict[str, Union[str, Image.Image, Dict]]] = None
    query: Optional[str] = None
    response: Optional[str] = None
    plan: Optional[str] = None
    task: Optional[str] = None
    actions: Optional[List[str]] = None
    additional: Optional[Dict[str, str]] = None  # {tag: content}

    def reset_except_observation(self) -> None:
        """Reset all fields to None except observation using dynamic field iteration"""
        for field in self.__dataclass_fields__:
            if field != "observation":
                setattr(self, field, None)

    def update(self, **kwargs) -> None:
        """Update state fields with new values
        Args:
            **kwargs: Field names and values to update
        """
        for field, value in kwargs.items():
            if field in self.__dataclass_fields__:
                setattr(self, field, value)


class PipeStage:
    """Defines a individual processing stage within a parser pipeline."""

    def __init__(self, processor_func, output_field=None, description=""):
        self.processor_func = processor_func
        self.output_field = output_field
        self.description = description

    def execute(self, content, message_state):
        """Execute the processing step
        Args:
            content: The input content to process
            message_state: MessageState instance to update
        Returns:
            The result of the processor function execution
        """
        # Check if this is a bound method (instance method of a class)
        is_bound_method = (
            hasattr(self.processor_func, "__self__")
            and self.processor_func.__self__ is not None
        )
        try:
            result = self.processor_func(content)
        except Exception as e:
            logger.error(
                f"{'Bound' if is_bound_method else 'Unbound'} processor function failed: {str(e)}"
            )
            raise
        if self.output_field and hasattr(message_state, self.output_field):
            message_state.update(**{self.output_field: result})

        return result


class PipeManager:
    """Manages and executes parser-specific processing pipelines"""

    def __init__(self):
        self.parser_pipelines = {}

    def register_pipeline(self, parser_id: str, steps: List[PipeStage]):
        """Register a processing pipeline for a parser"""
        self.parser_pipelines[parser_id] = steps

    def get_pipeline(self, parser_id: str) -> List[PipeStage]:
        """Get processing pipeline for a parser"""
        return self.parser_pipelines.get(parser_id, [])

    def execute_pipeline(
        self, parser_id: str, content: Any, message_state
    ) -> Tuple[bool, Any]:
        """Execute the entire processing pipeline for a parser"""
        if parser_id not in self.parser_pipelines:
            return False, content

        current_content = content
        result = None
        for step in self.parser_pipelines[parser_id]:
            result = step.execute(current_content, message_state)
            current_content = result

        return True, result


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
        self.game: ComputerGame = None
        self.game_instance: Dict = None
        self.terminated: bool = False
        self.message_state = MessageState()
        self.pipe_manager = PipeManager()

    def _on_setup(self, **game_instance) -> None:
        """Method executed at the start of the default setup method.
        Key Actions:
            - Sets up environment and loads initial observation + starts gameplay recording.
            - Constructs player interaction graph/ network.
            - Sets up trigger pipeline (specific) "parse func. -> after parse steps"
        Args:
            game_instance: Keyword arguments of the game_instance
        """
        self.game_instance = game_instance

        self._initialize_environment()
        self._build_graph()
        self._setup_after_parse_pipelines()

    def _initialize_environment(self) -> None:
        """Initializes game environment with recording capabilities and retrieves the inital state observation"""
        try:
            env_config = DEFAULT_ENV_CONFIG.copy()
            self.game = ComputerGame(
                **env_config, game_instance=self.game_instance["task_config"]
            )
            observation = self.game.env.reset(
                task_config=self.game_instance["task_config"]
            )
            self.message_state.update(observation=observation)
            if not self.game.env.start_recording():
                raise RuntimeError("Failed to start environment recording")

        except Exception as e:
            error_message = (
                f"Environment initialization failed: {str(e)}"
                if "recording" not in str(e).lower()
                else f"Recording initialization failed: {str(e)}"
            )
            raise RuntimeError(error_message) from e

    def _build_graph(self) -> None:
        """Builds a dialogic-player network graph from the game instance configuration."""
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
                    valid_entries = role_config.get("valid_entries", [])
                    prompt_header = role_config.get("prompt_header")
                    player = RoleBasedPlayer(
                        self.player_models[role_index],
                        role=role_name,
                        prompt_header=prompt_header,
                        handler_type=handler_type,
                        valid_entries=valid_entries,
                        **DEFAULT_ENV_CONFIG,
                    )
                    self.add_player(player, node_id)

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
        """Initialize processing pipelines for different parser"""
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
        return not self.terminated and self.current_turn < self.game.max_steps

    def add_message(self, player: Player, utterance: str, role: str):
        """Overrides parent class method (not used in this implementation)."""
        pass

    def add_user_message(self, player: Player, utterance: str, image: List[str] = None):
        """Overrides parent class method (implemented in RoleBasedPlayer)."""
        pass

    def add_assistant_message(self, player: Player, utterance: str):
        """Overrides parent class method (implemented in RoleBasedPlayer)."""
        pass

    def _on_before_game(self):
        """Executed once at the start, before entering the play loop.
        Key Actions
            - Adds the initial game-context to the anchor player
        """
        anchor_player = self.get_player_from_node(self.anchor_node)
        anchor_player.add_user_message(**self.message_state.__dict__)
        logger.info(
            f"Added the initial game-context for anchor player: {anchor_player.descriptor}"
        )

    def _on_before_node_transition(self, from_node: str, to_node: str):
        """Executed right before transitioning from one node to another.
        Key Actions:
            - Updates the player at the target node with the current message state.
        Args:
            from_node: The node ID that the system is transitioning from.
            to_node: The node ID that the system is transitioning to.
        """
        to_player = self.get_player_from_node(to_node)
        if not to_player:
            return
        to_player.add_user_message(**self.message_state.__dict__)
        logger.info(
            f"Added message state to player at node {to_node} (player: {to_player.descriptor})"
        )

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """Decide if an utterance should be added to the conversation history.
        Args:
            player: The Player instance for which the response is added.
            utterance: The text content of the message to be added.
        Returns:
            True, if the utterance is fine; False, if the response should not be added to the history.
        """
        if utterance is None:
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
            validation_result = condition.validate(utterance)
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
            "Validation failed, utterance did not comply with available conditions for connecting edges"
        )
        return False

    def _parse_response_for_decision_routing(
        self, player: Player, utterance: str
    ) -> Tuple[str, bool, Optional[str], Optional[Any]]:
        """Parse player response and evaluate decision edge conditions.
        Key Actions:
            1. Parse the player's utterance for relevant content
            2. Evaluate decision edge conditions based on the parsed content
            3. Determine which decision edge (if any) should be taken
        Args:
            player: The Player instance that produced the response.
            utterance: The text content of the response.
        Returns:
            Tuple containing:
            - Modified utterance (or original if no modification)
            - Boolean flag for logging
            - Next node ID from a decision edge, or None if no decision edge condition is met
            - Extracted content (if any)
        """
        player_node = self.get_node_from_player(player)
        decision_edges = [
            (to_node, edge_data["condition"])
            for _, to_node, edge_data in self.graph.out_edges(player_node, data=True)
            if edge_data.get("type") == EdgeType.DECISION and edge_data.get("condition")
        ]
        print("::DECISION_EDGES::", decision_edges)
        if not decision_edges:
            return utterance, False, None, None
        # Evaluate each decision edge condition
        for to_node, condition in decision_edges:
            print("::TO_NODE::", to_node, "::CONDITION::", condition)
            try:
                parse_result = condition.parse(utterance)
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
                        utterance,
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

        return utterance, False, None, None

    def _execute_actions(
        self, content: str
    ) -> Optional[Dict[str, Union[str, Image.Image, Dict]]]:
        """Execute either pyautogui or computer13 actions and record observations (upon environment-state change)
        Args:
            content: Parser extracted actions typically either pyautogui python-code (as str.), or computer13 actions in JSON (as str.)
        """
        print(content, "EXECUTE_ACTIONS")
        try:
            if not content:
                logger.error("No actions to execute")
                return None
            observation = None
            for action in content:
                try:
                    observation, reward, done, info = self.game.env.step(
                        action, self.game.sleep_after_execution
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
    master._on_before_game()

    # --- DEBUG_HARNESS: Testing and validation code (safe to remove) ---
    def run_debug_tests():
        print("\n" + "=" * 50)
        print("DEBUG HARNESS: Comprehensive System Validation")
        print("=" * 50)

        print("\n[TEST 1] Graph Structure Validation")
        print("-" * 40)
        try:
            # Visualize graph and save to debug folder
            os.makedirs("debug_output", exist_ok=True)
            master.visualize_graph(
                figsize=(12, 10), save_path="debug_output/interaction_graph.png"
            )
            print("✓ Graph visualization saved to debug_output/interaction_graph.png")

            # Print node and edge information
            player_nodes = [
                node
                for node, data in master.graph.nodes(data=True)
                if data.get("type") == NodeType.PLAYER
            ]
            print(f"✓ Player nodes: {player_nodes}")
            print(f"✓ Total nodes: {len(master.graph.nodes())}")
            print(f"✓ Total edges: {len(master.graph.edges())}")
            print(f"✓ Anchor node: {master.anchor_node}")
        except Exception as e:
            print(f"✗ Graph structure test failed: {str(e)}")

        print("\n[TEST 2] Player Configuration")
        print("-" * 40)
        try:
            for node_id in player_nodes:
                player = master.get_player_from_node(node_id)
                if player:
                    print(f"Player '{node_id}' configuration:")
                    print(f"  - Role: {player.role}")
                    print(f"  - Handler type: {player.prompt_handler.handler_type}")
                    print(f"  - Valid entries: {player.prompt_handler.valid_entries}")
                    if player.prompt_handler.handler_type == "environment":
                        print(
                            f"  - Observation type: {getattr(player.prompt_handler, 'observation_type', 'N/A')}"
                        )
            print("✓ Player configuration validation complete")
        except Exception as e:
            print(f"✗ Player configuration test failed: {str(e)}")

        print("\n[TEST 3] Environment Initialization")
        print("-" * 40)
        try:
            env = master.game.env
            print(f"✓ Environment type: {type(env).__name__}")
            print(f"✓ Action space type: {master.game.action_space}")
            print(f"✓ Observation type: {master.game.observation_type}")
            print(f"✓ Max steps: {master.game.max_steps}")
            print("✓ Environment initialization validation complete")
        except Exception as e:
            print(f"✗ Environment test failed: {str(e)}")

        print("\n[TEST 4] Decision Edge Routing and Pipeline Processing")
        print("-" * 40)
        try:
            # First verify the decision edges are properly set up
            decision_edges = [
                (from_node, to_node, data)
                for from_node, to_node, data in master.graph.edges(data=True)
                if data.get("type") == EdgeType.DECISION
            ]
            print(f"Found {len(decision_edges)} decision edges:")
            for from_node, to_node, data in decision_edges:
                print(
                    f"  - {from_node} → {to_node} ({data.get('condition').function_pair.parse_func.__name__})"
                )

            # Test parsing and pipeline execution for different message types
            test_cases = {
                "pyautogui": {
                    "utterance": "EXECUTE```python\nimport pyautogui\npyautogui.click(100, 100)```",
                    "parser": "pyautogui_actions",
                    "expected_field": "actions",
                    "from_node": "executor",  # Add source node
                    "to_node": "executor",  # Add target node
                },
                "query": {
                    "utterance": "QUERY```How do I change search settings?```",
                    "parser": "query",
                    "expected_field": "query",
                    "from_node": "executor",
                    "to_node": "advisor",
                },
                "response": {
                    "utterance": "RESPONSE```Click the three dots menu and go to Settings```",
                    "parser": "response",
                    "expected_field": "response",
                    "from_node": "advisor",
                    "to_node": "executor",
                },
                "done": {
                    "utterance": "Task completed successfully. DONE",
                    "parser": "done_or_fail",
                    "expected_field": "actions",
                    "from_node": "executor",
                    "to_node": "END",
                },
            }

            for test_name, test_data in test_cases.items():
                print(f"\nTesting {test_name} parsing and pipeline:")

                # Set current node before testing
                master.current_node = test_data["from_node"]
                test_player = master.get_player_from_node(master.current_node)

                if not test_player:
                    print(f"  ✗ No player found at node {master.current_node}")
                    continue

                # Test parsing
                utterance, log_flag, next_node, extracted = (
                    master._parse_response_for_decision_routing(
                        test_player, test_data["utterance"]
                    )
                )
                print(f"  ✓ Current node: {master.current_node}")
                print(f"  ✓ Expected next node: {test_data['to_node']}")
                print(f"  ✓ Actual next node: {next_node}")
                print(f"  ✓ Extracted content available: {extracted is not None}")

                # Test pipeline execution if content was extracted
                if extracted:
                    content, result = extracted
                    success, pipeline_result = master.pipe_manager.execute_pipeline(
                        parser_id=test_data["parser"],
                        content=content,
                        message_state=master.message_state,
                    )
                    print(f"  ✓ Pipeline execution success: {success}")
                    print(
                        f"  ✓ Target field '{test_data['expected_field']}' updated: "
                        f"{hasattr(master.message_state, test_data['expected_field'])}"
                    )

            print("\n✓ Decision routing and pipeline testing complete")
        except Exception as e:
            print(f"✗ Decision routing test failed: {str(e)}")

        print("\n[TEST 5] Action Execution Pipeline")
        print("-" * 40)
        try:
            # Test action execution pipeline
            test_actions = ["pyautogui.click(100, 100)", "pyautogui.press('enter')"]
            print("Testing action execution pipeline:")

            # Get pipeline for pyautogui actions
            pipeline = master.pipe_manager.get_pipeline("pyautogui_actions")
            print(f"  ✓ Pipeline stages found: {len(pipeline)}")
            for stage in pipeline:
                print(f"    - Stage: {stage.description}")
                print(f"    - Output field: {stage.output_field}")

            # Test direct execution of actions
            print("\nTesting direct action execution:")
            result = master._execute_actions(content=test_actions)
            print(f"  ✓ Action execution returned observation: {result is not None}")
            if result:
                print(f"  ✓ Observation type: {type(result).__name__}")

            # Test pipeline integration
            print("\nTesting pipeline integration:")
            initial_state = {
                k: getattr(master.message_state, k)
                for k in master.message_state.__dataclass_fields__
            }
            success, pipeline_result = master.pipe_manager.execute_pipeline(
                parser_id="pyautogui_actions",
                content=test_actions,
                message_state=master.message_state,
            )
            print(f"  ✓ Pipeline execution success: {success}")
            print(f"  ✓ Pipeline result available: {pipeline_result is not None}")

            # Check state changes
            changed_fields = []
            for field in master.message_state.__dataclass_fields__:
                if getattr(master.message_state, field) != initial_state[field]:
                    changed_fields.append(field)
            print(f"  ✓ Changed state fields: {changed_fields}")

            print("\n✓ Action execution pipeline testing complete")
        except Exception as e:
            print(f"✗ Action execution test failed: {str(e)}")

        print("\n[TEST 6] Message State Management")
        print("-" * 40)
        try:
            # Test message state updates through pipeline
            print("Testing message state management:")

            # Reset message state
            master.message_state.reset_except_observation()
            print("  ✓ Message state reset")

            # Test state updates through different pipelines
            test_query = "QUERY```How do I access settings?```"
            test_player = master.get_player_from_node(master.anchor_node)

            _, _, _, extracted = master._parse_response_for_decision_routing(
                test_player, test_query
            )
            if extracted:
                content, _ = extracted
                success, _ = master.pipe_manager.execute_pipeline(
                    parser_id="query",
                    content=content,
                    message_state=master.message_state,
                )
                print(f"  ✓ Query pipeline updated message state: {success}")
                print(
                    f"  ✓ Query field populated: {master.message_state.query is not None}"
                )

            print("\n✓ Message state management testing complete")
        except Exception as e:
            print(f"✗ Message state test failed: {str(e)}")

        print("\n" + "=" * 50)
        print("DEBUG HARNESS: Summary")
        print("=" * 50)
        print("All tests completed. Check output for any issues marked with ✗")
        print("=" * 50 + "\n")

    # Run the debugging tests
    run_debug_tests()
    # --- END_DEBUG_HARNESS ---


# TODO
# 1. add the log_to_self messaging everywhere
# 2 (a). _validate_xxx function is not yet implemented--do it today!
# 2 (b). connect the parts with proper logging and other related issues.
# 3. run one instance, the current instances.json file should work just fine. [DID THIS]
# 4. maybe need to rewrite the prompt-header
# 5. think about the re-prompting part.
