import json
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from PIL import Image
import os

from clemcore import backends
from clemcore.clemgame import Player, GameMaster, GameBenchmark
from game_master import NetworkDialogueGameMaster, EdgeCondition, NodeType, EdgeType
from game import ComputerGame, RoleBasedPlayer
from registry import parsers
from constants import (
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

    def reset_except_observation(self) -> None:
        """Reset all fields to None except observation using dynamic field iteration"""
        for field in self.__dataclass_fields__:
            if field != "observation":
                setattr(self, self, field, None)

    def update(self, **kwargs) -> None:
        """Update state fields with new values
        Args:
            **kwargs: Field names and values to update
        """
        for field, value in kwargs.items():
            if field in self.__dataclass_fields__:
                setattr(self, field, value)


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

    def _on_setup(self, **game_instance) -> None:
        """Method executed at the start of the default setup method.
        Key Actions:
            - Sets up environment and loads initial observation + starts gameplay recording.
            - Constructs player interaction graph/ network.
        Args:
            game_instance: Keyword arguments of the game_instance
        """
        self.game_instance = game_instance

        self._initialize_environment()
        self._build_graph()

        self.message_state.update(observation=self.current_observation)

    def _initialize_environment(self) -> None:
        """Initializes game environment with recording capabilities and retrieves the inital state observation"""
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
                    parse_function_id = condition_config.get("parse_function_id")
                    if not parse_function_id or parse_function_id not in parsers:
                        raise KeyError(
                            f"Invalid parse function ID: {parse_function_id}"
                        )
                    parse_func = parsers[parse_function_id]
                    condition = EdgeCondition(
                        parse_func=parse_func, description=description
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

    def _parse_response_for_decision_routing(
        self, player: Player, utterance: str
    ) -> Tuple[str, bool, Optional[str], Optional[str]]:
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
        decision_edges = [
            (to_node, edge_data["condition"])
            for _, to_node, edge_data in self.graph.out_edges(
                self.current_node, data=True
            )
            if edge_data.get("type") == EdgeType.DECISION and edge_data.get("condition")
        ]
        if not decision_edges:
            return utterance, False, None, None
        # Evaluate each decision edge condition
        for to_node, condition in decision_edges:
            try:
                is_match, extracted_content = condition.parse(player, utterance, self)
                if is_match and extracted_content:
                    logger.info(
                        f"Decision edge condition met: {self.current_node} → {to_node} with content: {extracted_content[:50]}..."
                        if len(extracted_content) > 50
                        else extracted_content
                    )
                    return utterance, True, to_node, extracted_content

            except Exception as e:
                logger.error(
                    f"Error evaluating condition for edge to {to_node}: {str(e)}"
                )

        return utterance, False, None, None

    def _test_parse_response_for_decision_routing(
        self, test_node_id=None, test_utterance=None
    ):
        """Test function for validating the decision edge routing logic.

        Args:
            test_node_id: Optional node ID to set as current node for testing
            test_utterance: Optional test utterance to parse

        Returns:
            Dict containing test results and diagnostic information
        """
        # Store original current node to restore later
        original_node = self.current_node

        # Use provided node or default to anchor node
        if test_node_id and test_node_id in self.graph:
            self.current_node = test_node_id
        elif self.anchor_node:
            self.current_node = self.anchor_node

        # Get the player associated with the current node
        test_player = self.get_player_from_node(self.current_node)
        if not test_player:
            self.current_node = original_node
            return {"error": f"No player found at node {self.current_node}"}

        # Use provided utterance or create test utterances for different parsers
        test_results = {}

        if test_utterance:
            # Test with the specific utterance provided
            utterance, log_flag, next_node, extracted = (
                self._parse_response_for_decision_routing(test_player, test_utterance)
            )
            test_results["custom_test"] = {
                "utterance": test_utterance[:100] + "..."
                if len(test_utterance) > 100
                else test_utterance,
                "next_node": next_node,
                "extracted_content": extracted[:100] + "..."
                if extracted and len(extracted) > 100
                else extracted,
                "success": next_node is not None,
            }
        else:
            # Create and test sample utterances for each parser
            test_utterances = {
                "pyautogui_actions": "EXECUTE```python\nimport pyautogui\npyautogui.moveTo(100, 100)\npyautogui.click()```",
                "query": "QUERY```How do I open Chrome settings?```",
                "response": "RESPONSE```Click on the three dots in the top right corner and select Settings.```",
                "done_or_fail": "I've completed the task successfully. DONE",
                "computer13_actions": 'EXECUTE```json\n[{"action": "click", "position": [100, 100]}]```',
            }

            # Execute tests for each utterance type
            for parser_id, utterance_text in test_utterances.items():
                utterance, log_flag, next_node, extracted = (
                    self._parse_response_for_decision_routing(
                        test_player, utterance_text
                    )
                )
                test_results[parser_id] = {
                    "utterance": utterance_text[:100] + "..."
                    if len(utterance_text) > 100
                    else utterance_text,
                    "next_node": next_node,
                    "extracted_content": extracted[:100] + "..."
                    if extracted and len(extracted) > 100
                    else extracted,
                    "success": next_node is not None,
                }

        # Collect information about available edges for diagnostics
        available_edges = []
        for _, to_node, edge_data in self.graph.out_edges(self.current_node, data=True):
            edge_type = edge_data.get("type", "UNKNOWN")
            condition = None
            if edge_type == EdgeType.DECISION and "condition" in edge_data:
                condition = (
                    edge_data["condition"].description
                    if hasattr(edge_data["condition"], "description")
                    else "No description"
                )
            available_edges.append(
                {
                    "to_node": to_node,
                    "edge_type": str(edge_type),
                    "condition": condition,
                }
            )

        # Restore original node
        self.current_node = original_node

        # Return comprehensive results
        return {
            "test_node": test_node_id or self.current_node,
            "available_edges": available_edges,
            "test_results": test_results,
        }


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

        print("\n[TEST 4] Player Message History")
        print("-" * 40)
        try:
            # Loop through all player nodes
            for node_id in player_nodes:
                player = master.get_player_from_node(node_id)
                if player:
                    # Mark if this is the anchor node
                    anchor_indicator = (
                        " (ANCHOR NODE)" if node_id == master.anchor_node else ""
                    )
                    print(f"Player '{node_id}'{anchor_indicator} message history:")

                    if hasattr(player, "prompt_handler") and hasattr(
                        player.prompt_handler, "history"
                    ):
                        for i, msg in enumerate(player.prompt_handler.history):
                            print(
                                f"  [{i}] Role: {msg.get('role')}, Length: {len(msg.get('content', ''))}"
                            )
                            if i < 2:  # Show preview of first messages only
                                content_preview = (
                                    msg.get("content", "")[:100] + "..."
                                    if len(msg.get("content", "")) > 100
                                    else msg.get("content", "")
                                )
                                print(f"      Preview: {content_preview}")
                        print(f"  Total messages: {len(player.prompt_handler.history)}")
                    else:
                        print("  No message history available")
                    print("-" * 30)  # Separator between players

            print("✓ Player message history validation complete")
        except Exception as e:
            print(f"✗ Message history test failed: {str(e)}")

        print("\n[TEST 5] Game State and Turn Tracking")
        print("-" * 40)
        try:
            print(f"✓ Current turn: {master.current_turn}")
            print(f"✓ Game terminated: {master.terminated}")
            if hasattr(master, "turn_path"):
                print(f"✓ Turn path: {master.turn_path}")
            if hasattr(master, "turn_visited_nodes"):
                print(f"✓ Visited nodes: {master.turn_visited_nodes}")
            print("✓ Game state validation complete")
        except Exception as e:
            print(f"✗ Game state test failed: {str(e)}")

        print("\n[TEST 6] Node Transition Message Update")
        print("-" * 40)
        try:
            # Test node transition behavior
            if len(player_nodes) >= 2:
                # Find a node that's not the anchor node to test transition to
                test_node = next(
                    (node for node in player_nodes if node != master.anchor_node),
                    player_nodes[0],
                )

                # Get the player's message count before transition
                test_player = master.get_player_from_node(test_node)
                msg_count_before = (
                    len(test_player.prompt_handler.history) if test_player else 0
                )

                # Add a test value to message state
                master.message_state.update(query="This is a test transition query")

                # Simulate transition
                current_node = (
                    master.anchor_node if master.anchor_node else player_nodes[0]
                )
                print(f"✓ Simulating transition: {current_node} -> {test_node}")
                master._on_before_node_transition(current_node, test_node)

                # Check if message was added
                msg_count_after = (
                    len(test_player.prompt_handler.history) if test_player else 0
                )
                print(
                    f"✓ Messages before: {msg_count_before}, after: {msg_count_after}"
                )

                # Verify message content
                if msg_count_after > msg_count_before:
                    newest_msg = test_player.prompt_handler.history[-1]
                    preview = (
                        newest_msg.get("content", "")[:100] + "..."
                        if len(newest_msg.get("content", "")) > 100
                        else newest_msg.get("content", "")
                    )
                    print(f"✓ New message content: {preview}")

                    # Check if our test query was included
                    if "This is a test transition query" in newest_msg.get(
                        "content", ""
                    ):
                        print("✓ Test query successfully passed to destination player")
                    else:
                        print("✗ Test query not found in destination player message")

                print("✓ Node transition messaging test complete")
            else:
                print("✓ Skipping node transition test (insufficient player nodes)")
        except Exception as e:
            print(f"✗ Node transition test failed: {str(e)}")

        print("\n[TEST 7] Decision Edge Routing")
        print("-" * 40)
        try:
            # Test the decision edge routing logic
            if hasattr(master, "_test_parse_response_for_decision_routing"):
                routing_test_results = (
                    master._test_parse_response_for_decision_routing()
                )

                print(
                    f"Testing decision edge routing from node: {routing_test_results.get('test_node', 'Unknown')}"
                )
                print("\nAvailable edges:")
                for edge in routing_test_results.get("available_edges", []):
                    edge_desc = f"→ {edge['to_node']} ({edge['edge_type']})"
                    if edge.get("condition"):
                        edge_desc += f": {edge['condition']}"
                    print(f"  {edge_desc}")

                print("\nRouting test results:")
                for parser_id, result in routing_test_results.get(
                    "test_results", {}
                ).items():
                    status = "✓" if result.get("success") else "✗"
                    print(f"\n  {status} Parser: {parser_id}")
                    print(f"    Utterance: {result.get('utterance', 'N/A')}")
                    print(f"    Next node: {result.get('next_node', 'None')}")
                    if result.get("extracted_content"):
                        print(
                            f"    Extracted: {result.get('extracted_content', 'N/A')}"
                        )

                # Test with a specific utterance if we have a query edge
                query_edge = next(
                    (
                        edge
                        for edge in routing_test_results.get("available_edges", [])
                        if edge.get("condition")
                        and "query" in str(edge.get("condition")).lower()
                    ),
                    None,
                )
                if query_edge:
                    specific_test = master._test_parse_response_for_decision_routing(
                        test_utterance="QUERY```Can you help me find the settings menu?```"
                    )
                    result = specific_test.get("test_results", {}).get(
                        "custom_test", {}
                    )
                    status = "✓" if result.get("success") else "✗"
                    print(f"\n  {status} Custom Query Test")
                    print(f"    Utterance: {result.get('utterance', 'N/A')}")
                    print(f"    Next node: {result.get('next_node', 'None')}")
                    if result.get("extracted_content"):
                        print(
                            f"    Extracted: {result.get('extracted_content', 'N/A')}"
                        )

                print("\n✓ Decision edge routing test complete")
            else:
                print("✗ Test method not found on master instance")
        except Exception as e:
            print(f"✗ Decision edge routing test failed: {str(e)}")

        print("\n" + "=" * 50)
        print("DEBUG HARNESS: Summary")
        print("=" * 50)
        print("All tests completed. Check output for any issues marked with ✗")
        print("=" * 50 + "\n")

    # Run the debugging tests
    run_debug_tests()
    # --- END_DEBUG_HARNESS ---
