import json
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from PIL import Image
import os

from clemcore import backends
from clemcore.clemgame import (
    Player,
    DialogicNetworkGameMaster,
)
from clemcore.clemgame.master import EdgeCondition
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
        """Initializes game environment with recording capabilities.
        Also retrieves initial environment-state observation"""
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
        """Initializes the game by prompting the anchor node with initial observation."""
        anchor_player = self.get_player_from_node(self.anchor_node)
        try:
            anchor_player.add_user_message(**self.message_state.__dict__)
            logger.info(
                f"Initial observation added for anchor player: {anchor_player.descriptor}"
            )
        except Exception as e:
            self.terminated = True
            raise RuntimeError(
                f"Failed to initialize game with anchor player: {str(e)}"
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
                if data.get("type") == "PLAYER"
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
            anchor_player = master.get_player_from_node(master.anchor_node)
            if anchor_player:
                print(f"Anchor player '{master.anchor_node}' message history:")
                for i, msg in enumerate(anchor_player.prompt_handler.history):
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
                print(
                    f"✓ Total messages in history: {len(anchor_player.prompt_handler.history)}"
                )

                # Check if any other players have message history
                for node_id in player_nodes:
                    if node_id != master.anchor_node:
                        player = master.get_player_from_node(node_id)
                        if player and player.prompt_handler.history:
                            print(
                                f"Player '{node_id}' has {len(player.prompt_handler.history)} messages in history"
                            )
            else:
                print("✗ No anchor player found")
            print("✓ Message history validation complete")
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

        print("\n" + "=" * 50)
        print("DEBUG HARNESS: Summary")
        print("=" * 50)
        print("All tests completed. Check output for any issues marked with ✗")
        print("=" * 50 + "\n")

    # Run the debugging tests
    run_debug_tests()
    # --- END_DEBUG_HARNESS ---
