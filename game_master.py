import collections
import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx

from clemcore import backends
from clemcore.clemgame import Player, GameMaster

module_logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Node types in the Network."""

    START = auto()
    PLAYER = auto()
    END = auto()


class EdgeType(Enum):
    """Edge types in the Network."""

    STANDARD = (
        auto()
    )  # Direct connection, always traversed if no other decision edges are taken
    DECISION = (
        auto()
    )  # Conditional connection, traversed only if condition evaluates to True


class EdgeCondition:
    """Condition for transitioning between nodes in the Network with content extraction."""

    def __init__(
        self,
        parse_func: Callable[
            [Player, str, "NetworkDialogueGameMaster"], Tuple[bool, Optional[str]]
        ],
        description: str = "",
    ):
        """Args:
        parse_func: Function that takes (player, utterance, game_master) and returns
                   a tuple containing (is_match, parsed_content). The is_match indicates
                   if parsing was successful, and parsed_content contains the extracted text
                   (or None if parsing failed).
        description: Human-readable description of the condition for visualization.
        """
        self.parse_func = parse_func
        self.description = description

    def parse(
        self,
        player: Player,
        utterance: str,
        game_master: "NetworkDialogueGameMaster",
    ) -> Tuple[bool, Optional[str]]:
        """Parse the utterance and determine if the condition is satisfied.
        Args:
            player: The Player instance that produced the response.
            utterance: The text content of the response.
            game_master: The DialogicNetworkGameMaster instance.
        Returns:
            A tuple containing (is_match, parsed_content). The is_match indicates
            if parsing was successful, and parsed_content contains the extracted text
            (or None if parsing failed).
        """
        return self.parse_func(player, utterance, game_master)


@dataclass
class NodeTransition:
    """Temporary storage for node transition data."""

    next_node: Optional[str] = None
    extracted_content: Optional[str] = None


class NetworkDialogueGameMaster(GameMaster):
    """Extended GameMaster, implements a graph-based approach for player interaction flow.
    Players are represented as nodes in a directed graph, with edges representing possible
    transitions between players.
    The graph supports two types of edges:
        - Standard edges: Direct connections that are always traversed if no decision edges are taken
        - Decision edges: Conditional connections that are traversed only if their condition evaluates to True
    This allows for complex interaction patterns including branching paths, self-loops, and
    conditional transitions based on player responses.
    Turn Definition:
        A turn is defined as the sequence of state transitions that:
            - Begins when the system enters the anchor node
            - Progresses through sequential interactions, potentially including repeated anchor node interactions
            - Requires at least one transition to a non-anchor node
            - Concludes when the system returns to the anchor node after visiting at least one non-anchor node
    """

    def __init__(
        self,
        name: str,
        path: str,
        experiment: dict,
        player_models: List[backends.Model],
    ):
        """Args:
        name: The name of the game (as specified in game_registry).
        path: Path to the game (as specified in game_registry).
        experiment: The experiment (set of instances) to use.
        player_models: Player models to use for one or more players.
        """
        super().__init__(name, path, experiment, player_models)

        # the logging works with an internal mapping  of "Player N" -> Player
        self.players_by_names: Dict[str, Player] = collections.OrderedDict()
        self.messages_by_names: Dict[str, List] = dict()
        self.current_turn: int = 0

        self.graph = nx.MultiDiGraph()
        self.graph.add_node("START", type=NodeType.START)
        self.graph.add_node("END", type=NodeType.END)

        self.current_node = "START"
        self.last_player: Optional[Player] = None
        self.last_utterance: Optional[str] = None

        self.node_positions = None
        self.edge_labels = {}

        self.anchor_node = None
        self.current_turn_nodes = []
        self.non_anchor_visited = False
        self.turn_complete = False

        self.transition = NodeTransition()

    def get_players(self) -> List[Player]:
        """Get a list of the players.
        Returns:
            List of Player instances in the order they are added.
        """
        return list(self.players_by_names.values())

    def add_player(self, player: Player, node_id: str = None):
        """Add a player to the game and the graph.
        Args:
            player: The player to be added to the game.
            node_id: Optional custom node ID for the player. If None, the player's descriptor will be used.
        """
        idx = len(self.players_by_names)
        player.descriptor = f"Player {idx + 1}"
        self.players_by_names[player.descriptor] = player
        self.messages_by_names[player.descriptor] = []

        node_id = node_id
        self.graph.add_node(node_id, type=NodeType.PLAYER, player=player)

    def add_standard_edge(self, from_node: str, to_node: str, label: str = ""):
        """Add a standard edge between nodes in the graph.
        Standard edges are always traversed if no decision edges are taken.
        Args:
            from_node: The ID of the source node.
            to_node: The ID of the target node.
            label: Optional label for the edge (for visualization).
        """
        if from_node not in self.graph:
            raise ValueError(f"Node '{from_node}' does not exist in the graph")
        if to_node not in self.graph:
            raise ValueError(f"Node '{to_node}' does not exist in the graph")

        self.graph.add_edge(
            from_node,
            to_node,
            type=EdgeType.STANDARD,
            condition=None,
            key=f"standard_{from_node}_{to_node}",
        )
        if label:
            edge_key = (from_node, to_node, f"standard_{from_node}_{to_node}")
            self.edge_labels[edge_key] = label

    def add_decision_edge(
        self, from_node: str, to_node: str, condition: EdgeCondition, label: str = ""
    ):
        """Add a decision edge between nodes in the graph.
        Decision edges are traversed only if their condition evaluates to True.
        Args:
            from_node: The ID of the source node.
            to_node: The ID of the target node.
            condition: The condition for the edge transition.
            label: Optional label for the edge (for visualization).
        """
        if from_node not in self.graph:
            raise ValueError(f"Node '{from_node}' does not exist in the graph")
        if to_node not in self.graph:
            raise ValueError(f"Node '{to_node}' does not exist in the graph")

        edge_count = sum(1 for edge in self.graph.edges(from_node, to_node, keys=True))
        edge_key = f"decision_{from_node}_{to_node}_{edge_count}"
        self.graph.add_edge(
            from_node,
            to_node,
            type=EdgeType.DECISION,
            condition=condition,
            key=edge_key,
        )
        if label:
            self.edge_labels[(from_node, to_node, edge_key)] = label
        elif condition and condition.description:
            self.edge_labels[(from_node, to_node, edge_key)] = condition.description

    def set_anchor_node(self, node_id: str):
        """Set the anchor node for turn tracking.
        The anchor node marks the beginning and end of a turn.
        Args:
            node_id: The ID of the node to set as anchor.
        """
        if node_id not in self.graph:
            raise ValueError(f"Node '{node_id}' does not exist in the graph")

        self.anchor_node = node_id
        module_logger.info(f"Anchor node set to '{node_id}'")

    def setup(self, **kwargs):
        """Load resources and prepare everything to play the game.
        Needs to log the players dictionary via self.log_players(players_dict).
        Intended to be left as-is by inheriting classes. Implement game-specific setup functionality in the _on_setup
        method.
        Called by the game's GameBenchmark run method for each game instance.
        Args:
            kwargs: Keyword arguments used to set up the GameMaster instance. This is usually a game instance object
                read from the game's instances.json.
        """
        self._on_setup(**kwargs)
        # log players
        players_descriptions = collections.OrderedDict(
            GM=f"Game master for {self.game_name}"
        )
        for name, player in self.players_by_names.items():
            players_descriptions[name] = player.get_description()
        # log player ID and description dict:
        self.log_players(players_descriptions)

    def play(self) -> None:
        """Main play loop method.
        This method is called to run the game for benchmarking.
        Intended to be left as-is by inheriting classes. Implement game-specific functionality in the
        _on_before_game, _does_game_proceed, _on_before_turn, _should_reprompt, _on_before_reprompt, _on_after_turn and
        _on_after_game methods.
        """
        self._on_before_game()
        self.current_node = "START"
        while self.current_node != "END" and self._does_game_proceed():
            # Check if a turn has completed based on the anchor node definition
            if self._is_turn_complete():
                self.log_next_turn()
                self._on_before_turn(self.current_turn)
                module_logger.info(
                    f"{self.game_name}: %s turn: %d", self.game_name, self.current_turn
                )
                self._reset_turn_tracking()
            next_node = self.transition.next_node
            self.transition = NodeTransition()
            if next_node is None:
                module_logger.warning(
                    f"No valid transitions from node '{self.current_node}'"
                )
                break
            self._on_before_node_transition(self.current_node, next_node)
            prev_node = self.current_node
            self.current_node = next_node
            self._update_turn_tracking(prev_node, next_node)
            if self.current_node == "END":
                break
            node_data = self.graph.nodes[self.current_node]
            if node_data["type"] == NodeType.PLAYER:
                player = node_data["player"]
                self.prompt(player)
                while self._should_reprompt(player):
                    self._on_before_reprompt(player)
                    self.prompt(player, is_reprompt=True)
                self.last_player = player
                self.last_utterance = self.messages_by_names[player.descriptor][-1][
                    "content"
                ]
            if self._is_turn_complete():
                self._on_after_turn(self.current_turn)
                self.current_turn += 1
        self._on_after_game()

    def prompt(self, player: Player, is_reprompt=False):
        """Prompt a player model.
        Includes logging of 'send message' and 'get message' actions.
        Intended to be left as-is by inheriting classes. Implement game-specific functionality in the
        _should_reprompt, _on_before_reprompt, _after_add_player_response, _validate_player_response methods.
        Args:
            player: The Player instance to be prompted.
            is_reprompt: If this is a reprompt attempt. This is intended for re-prompting with modified prompts.
        """
        # GM -> Player
        history = self.messages_by_names[player.descriptor]
        assert history, f"messages history must not be empty for {player.descriptor}"

        last_entry = history[-1]
        assert last_entry["role"] != "assistant", (
            "Last entry should not be assistant "
            "b.c. this would be the role of the current player"
        )
        message = last_entry["content"]

        action_type = "send message" if not is_reprompt else "send message (reprompt)"
        action = {"type": action_type, "content": message}
        self.log_event(from_="GM", to=player.descriptor, action=action)

        _prompt, _response, response_message = player(history, self.current_turn)

        # Player -> GM
        action = {"type": "get message", "content": response_message}
        # log 'get message' event including backend/API call:
        self.log_event(
            from_=player.descriptor, to="GM", action=action, call=(_prompt, _response)
        )

        # GM -> GM
        self.__validate_process_and_add_player_response(player, response_message)

    def _should_reprompt(self, player: Player):
        """Method to check if a Player should be re-prompted.
        This is intended to check for invalid responses.
        Args:
            player: The Player instance to re-prompt.
        """
        return False

    def _on_before_reprompt(self, player: Player):
        """Method executed before reprompt is passed to a Player.
        Hook
        Change the prompt to reprompt the player on e.g. an invalid response.
        Add the new prompt to the players message via self.add_user_message(player, new_prompt)
        Args:
            player: The Player instance that produced the invalid response.
        """
        pass

    def log_message_to(self, player: Player, message: str):
        """Logs a 'send message' action from GM to Player.
        This is a logging method, and will not add the message to the conversation history on its own!
        Args:
            player: The Player instance the message is targeted at.
            message: The message content sent to the Player instance.
        """
        action = {"type": "send message", "content": message}
        self.log_event("GM", player.descriptor, action)

    def log_message_to_self(self, message: str):
        """Logs a 'metadata' action from GM to GM.
        This is a logging method, and will not add anything to the conversation history.
        Args:
            message: The message content logged as metadata.
        """
        action = {"type": "metadata", "content": message}
        self.log_event("GM", "GM", action)

    def log_to_self(self, type_: str, value: str):
        """Logs an action of the passed type from GM to GM.
        This is a logging method, and will not add anything to the conversation history.
        Args:
            type_: The type of the action to be logged.
            value: The content value of the action to be logged.
        """
        action = {"type": type_, "content": value}
        self.log_event("GM", "GM", action)

    def add_message(self, player: Player, utterance: str, role: str):
        """Adds a message to the conversation history.
        This method is used to iteratively create the conversation history, but will not log/record messages
        automatically.
        Args:
            player: The Player instance that produced the message.
            utterance: The text content of the message to be added.
            role: The chat/instruct conversation role to use for this message. Either 'user' or 'assistant', or 'system'
                for models/templates that support it.
        """
        message = {"role": role, "content": utterance}
        history = self.messages_by_names[player.descriptor]
        history.append(message)

    def add_user_message(self, player: Player, utterance: str):
        """Adds a message with the 'user' role to the conversation history.
        Args:
            player: The Player instance that produced the message.
            utterance: The text content of the message to be added.
        """
        self.add_message(player, utterance, role="user")

    def add_assistant_message(self, player: Player, utterance: str):
        """Adds a message with the 'assistant' role to the conversation history.
        Args:
            player: The Player instance that produced the message.
            utterance: The text content of the message to be added.
        """
        self.add_message(player, utterance, role="assistant")

    def __validate_process_and_add_player_response(
        self, player: Player, utterance: str
    ):
        """Checks player response validity, parses it and adds it to the conversation history.
        Part of the play loop, not intended to be modified - modify _validate_player_response,
        _on_process_response_and_determine_routing and/or _after_add_player_response instead.
        Args:
            player: The Player instance that produced the response.
            utterance: The text content of the response.
        """
        if self._validate_player_response(player, utterance):
            utterance = self.__process_response(player, utterance)
            self.add_assistant_message(player, utterance)
            self._after_add_player_response(player, utterance)

    def _after_add_player_response(self, player: Player, utterance: str):
        """Method executed after a player response has been validated and added to the conversation history.
        Hook: Modify this method for game-specific functionality.
        Add the utterance to other player's history, if necessary.
        Args:
            player: The Player instance that produced the response.
            utterance: The text content of the message that was added.
        """
        pass

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """Decide if an utterance should be added to the conversation history.
        Hook: Modify this method for game-specific functionality.
        Args:
            player: The Player instance for which the response is added.
            utterance: The text content of the message to be added.
        Returns:
            True, if the utterance is fine; False, if the response should not be added to the history.
        """
        return True

    def __process_response(self, player: Player, utterance: str) -> str:
        """Parses a response, determines next node, and stores extracted content.
        Part of the validate-process loop, not intended to be modified.
        Args:
            player: The Player instance that produced the response.
            utterance: The text content of the response.
        Returns:
            The response content, potentially modified.
        """
        self.transition = NodeTransition()
        _utterance, log_action, next_node, extracted_content = (
            self._parse_response_for_decision_routing(player, utterance)
        )
        # If no decision edge was taken, fall back to standard edges
        if next_node is None:
            for _, to_node, edge_data in self.graph.out_edges(
                self.current_node, data=True
            ):
                if edge_data.get("type") == EdgeType.STANDARD:
                    next_node = to_node
                    break
        # Store transition data in temporary registry
        if next_node:
            self.transition.next_node = next_node
        if extracted_content:
            self.transition.extracted_content = extracted_content
        if _utterance != utterance and log_action:
            action = {"type": "parse", "content": _utterance}
            self.log_event(from_="GM", to="GM", action=action)
        return _utterance

    def _parse_response_for_decision_routing(
        self, player: Player, utterance: str
    ) -> Tuple[str, bool, Optional[str], Optional[str]]:
        """Parse player response and evaluate decision edge conditions.
        Hook: Modify this method for game-specific functionality.
        This method should:
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
        # Default implementation: no parsing or decision edge evaluation
        return utterance, True, None, None

    def _on_before_turn(self, turn_idx: int):
        """Executed in play loop after turn advance and before proceed check and prompting.
        Hook: Modify this method for game-specific functionality.
        Args:
            turn_idx: The current turn index.
        """
        pass

    def _on_after_turn(self, turn_idx: int):
        """Executed in play loop after prompting.
        Hook: Modify this method for game-specific functionality.
        Args:
            turn_idx: The current turn index.
        """
        pass

    def _on_before_game(self):
        """Executed once at the start, before entering the play loop.
        Hook: Modify this method for game-specific functionality.
        """
        pass

    def _on_after_game(self):
        """Executed once at the end, after exiting the play loop.
        Hook: Modify this method for game-specific functionality.
        """
        pass

    def visualize_graph(self, figsize=(12, 10), save_path=None, dpi=100):
        """Visualize the Network structure with professional styling.
        Args:
            figsize: Size of the figure (width, height) in inches.
            save_path: Optional path to save the visualization. If None, the visualization is displayed.
            dpi: Resolution for the output figure.
        """
        plt.figure(figsize=figsize, dpi=dpi)

        # Better node positioning - hierarchical layout works well for directed graphs
        if not self.node_positions:
            try:
                # Try to use a hierarchical layout for better flow visualization
                self.node_positions = nx.nx_pydot.pydot_layout(self.graph, prog="dot")
            except Exception:
                # Fall back to spring layout with better parameters
                self.node_positions = nx.spring_layout(
                    self.graph,
                    k=0.5,  # Optimal distance between nodes
                    iterations=100,  # More iterations for better layout
                    seed=42,  # Consistent layout between runs
                )

        # Professional color scheme
        node_colors = {
            NodeType.START: "#2ECC71",  # Emerald green
            NodeType.PLAYER: "#3498DB",  # Blue
            NodeType.END: "#E74C3C",  # Red
        }

        # Draw nodes by type for better styling
        for node_type in NodeType:
            nodes = [
                node
                for node in self.graph.nodes()
                if self.graph.nodes[node].get("type") == node_type
            ]
            if not nodes:
                continue

            nx.draw_networkx_nodes(
                self.graph,
                self.node_positions,
                nodelist=nodes,
                node_color=node_colors[node_type],
                node_size=3000,
                alpha=0.9,
                edgecolors="#2C3E50",  # Dark border
                linewidths=2,
            )

        # Create better node labels with appropriate text wrapping
        node_labels = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get("type")
            if node_type == NodeType.PLAYER:
                player = self.graph.nodes[node].get("player")

                # Handle either model name or role
                if hasattr(player, "role") and player.role:
                    role_text = player.role
                else:
                    # Use string representation if no role available
                    role_text = str(player.model)

                # Limit label length to avoid overflow
                node_labels[node] = f"{node}\n({role_text})"
            else:
                node_labels[node] = node

        # Draw edge types with distinctive styling
        standard_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("type") == EdgeType.STANDARD
        ]
        decision_edges = [
            (u, v)
            for u, v, d in self.graph.edges(data=True)
            if d.get("type") == EdgeType.DECISION
        ]

        # Standard edges with solid lines
        nx.draw_networkx_edges(
            self.graph,
            self.node_positions,
            edgelist=standard_edges,
            arrowsize=25,
            width=2.5,
            edge_color="#34495E",  # Dark gray
            connectionstyle="arc3,rad=0.1",  # Curved edges for better visibility
        )

        # Decision edges with dashed lines
        nx.draw_networkx_edges(
            self.graph,
            self.node_positions,
            edgelist=decision_edges,
            arrowsize=25,
            width=2,
            edge_color="#8E44AD",  # Purple
            style="dashed",
            connectionstyle="arc3,rad=0.1",  # Curved edges
        )

        # Draw node labels with better font
        nx.draw_networkx_labels(
            self.graph,
            self.node_positions,
            labels=node_labels,
            font_size=10,
            font_family="sans-serif",
            font_weight="bold",
            font_color="#FFFFFF",  # White text for better contrast
        )

        # Prepare edge labels with better formatting
        edge_labels_dict = {}
        for (u, v, k), label in self.edge_labels.items():
            # Limit edge label length and add line breaks
            if label and len(label) > 20:
                words = label.split()
                chunks = []
                current_chunk = []
                current_length = 0

                for word in words:
                    if current_length + len(word) > 20:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_length = len(word)
                    else:
                        current_chunk.append(word)
                        current_length += len(word) + 1

                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                label = "\n".join(chunks)

            edge_labels_dict[(u, v)] = label

        # Draw edge labels with better positioning
        nx.draw_networkx_edge_labels(
            self.graph,
            self.node_positions,
            edge_labels=edge_labels_dict,
            font_size=8,
            font_family="sans-serif",
            bbox=dict(
                facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.3"
            ),
            label_pos=0.4,  # Adjust label position along the edge
        )

        # Mark the anchor node with a special border if it exists
        if self.anchor_node and self.anchor_node in self.graph:
            anchor_pos = {self.anchor_node: self.node_positions[self.anchor_node]}
            nx.draw_networkx_nodes(
                self.graph,
                anchor_pos,
                nodelist=[self.anchor_node],
                node_color="none",  # Transparent fill
                node_size=3300,  # Slightly larger
                alpha=1.0,
                edgecolors="#FFD700",  # Gold border
                linewidths=4,
            )

        plt.title(
            f"Interaction Network for {self.game_name}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.axis("off")
        plt.tight_layout()

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                color=node_colors[NodeType.START],
                marker="o",
                linestyle="None",
                markersize=15,
                label="Start Node",
            ),
            plt.Line2D(
                [0],
                [0],
                color=node_colors[NodeType.PLAYER],
                marker="o",
                linestyle="None",
                markersize=15,
                label="Player Node",
            ),
            plt.Line2D(
                [0],
                [0],
                color=node_colors[NodeType.END],
                marker="o",
                linestyle="None",
                markersize=15,
                label="End Node",
            ),
            plt.Line2D([0], [0], color="#34495E", linewidth=2.5, label="Standard Edge"),
            plt.Line2D(
                [0],
                [0],
                color="#8E44AD",
                linewidth=2,
                linestyle="dashed",
                label="Decision Edge",
            ),
        ]
        if self.anchor_node:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    markerfacecolor="none",
                    markeredgecolor="#FFD700",
                    marker="o",
                    linestyle="None",
                    markersize=15,
                    markeredgewidth=2,
                    label="Anchor Node",
                )
            )

        plt.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            fontsize=10,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        else:
            plt.show()

    def set_node_positions(self, positions: Dict[str, Tuple[float, float]]):
        """Set custom positions for nodes in the visualization.
        Args:
            positions: Dictionary mapping node IDs to (x, y) positions.
        """
        self.node_positions = positions

    def _update_turn_tracking(self, prev_node: str, next_node: str):
        """Update the turn tracking state based on node transitions.
        Part of the core turn tracking system, not intended to be modified by inheriting classes.
        Args:
            prev_node: The node the system is transitioning from.
            next_node: The node the system is transitioning to.
        """
        if self.anchor_node is None:
            return
        self.current_turn_nodes.append(next_node)
        if next_node != self.anchor_node:
            self.non_anchor_visited = True
        if next_node == self.anchor_node and self.non_anchor_visited:
            self.turn_complete = True
            module_logger.debug(f"Turn complete: {self.current_turn_nodes}")

    def _is_turn_complete(self) -> bool:
        """Check if a turn is complete based on the formal definition.
        Part of the core turn tracking system, not intended to be modified by inheriting classes.
        A turn is complete when:
        1. The system has visited at least one non-anchor node
        2. The system has returned to the anchor node
        Returns:
            A bool indicating if a turn is complete.
        """
        return self.turn_complete

    def _reset_turn_tracking(self):
        """Reset the turn tracking state.
        Part of the core turn tracking system, not intended to be modified by inheriting classes.
        """
        self.current_turn_nodes = []
        self.non_anchor_visited = False
        self.turn_complete = False
        if self.current_node:
            self.current_turn_nodes.append(self.current_node)

    def get_current_turn_path(self) -> List[str]:
        """Get the path of nodes visited in the current turn.
        Returns:
            A list of node IDs representing the path of the current turn.
        """
        return self.current_turn_nodes.copy()

    def get_turn_completion_status(self) -> Dict[str, Any]:
        """Get the current status of turn completion.
        Returns:
            A dictionary containing turn completion status information.
        """
        return {
            "anchor_node": self.anchor_node,
            "visited_nodes": self.current_turn_nodes.copy(),
            "non_anchor_visited": self.non_anchor_visited,
            "turn_complete": self.turn_complete,
        }

    def _does_game_proceed(self) -> bool:
        """Check if game should proceed.
        Template method: Must be implemented by subclasses!
        Returns:
            A bool, True if game continues, False if game should stop.
        """
        raise NotImplementedError()

    def _on_setup(self, **kwargs):
        """Method executed at the start of the default setup method.
        Template method: Must be implemented by subclasses!
        Args:
            kwargs: Keyword arguments of the game instance.
        """
        raise NotImplementedError()

    def _on_before_node_transition(self, from_node: str, to_node: str):
        """Executed right before transitioning from one node to another.
        Hook: Modify this method for game-specific functionality.
        This method allows you to prepare messages or perform other actions
        before a specific node (especially player nodes) is reached.

        Args:
            from_node: The node ID that the system is transitioning from.
            to_node: The node ID that the system is transitioning to.
        """
        pass

    def get_player_from_node(self, node_id: str) -> Optional[Player]:
        """Get the Player instance associated with a node ID.
        Args:
            node_id: The ID of the node to get the player from.
        Returns:
            The Player instance if the node is a player node, None otherwise.
        """
        if node_id not in self.graph:
            return None

        node_data = self.graph.nodes[node_id]
        if node_data.get("type") == NodeType.PLAYER:
            return node_data.get("player")
        return None
