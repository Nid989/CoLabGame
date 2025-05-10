import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union, Optional, Tuple, NamedTuple
import matplotlib.pyplot as plt
import networkx as nx

from clemcore import backends
from clemcore.clemgame import DialogueGameMaster
from .player import RoleBasedPlayer
from .utils.registry.parsers import (
    parse_computer13_actions,
    parse_pyautogui_actions,
    parse_done_or_fail,
    parse_request,
    parse_response,
    parse_task,
)
from .utils.registry.validators import (
    validate_computer13_actions,
    validate_pyautogui_actions,
    validate_done_or_fail,
    validate_request,
    validate_response,
    validate_task,
    ValidationError,
)

module_logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Enum representing different types of nodes in the network."""

    START = auto()
    PLAYER = auto()
    END = auto()


class EdgeType(Enum):
    """Enum representing different types of edges in the network."""

    STANDARD = (
        auto()
    )  # Direct connection, always traversed if no other decision edges are taken
    DECISION = (
        auto()
    )  # Conditional connection, traversed only if condition evaluates to True


@dataclass
class ParseResult:
    """Container for parsing results.

    Attributes:
        is_successful: Whether the parsing operation succeeded.
        content: The extracted content from parsing, None if parsing failed.
    """

    is_successful: bool
    content: Optional[Any] = None


@dataclass
class ValidationResult:
    """Container for validation results.

    Attributes:
        is_valid: Whether the response passed all validation checks.
        intended_format: Whether the response was intended to follow this format.
        error: Validation error details if validation failed.
    """

    is_valid: bool
    intended_format: bool
    error: Optional[ValidationError] = None


class ConditionType(str, Enum):
    """Enum for pre-defined types of edge conditions."""

    COMPUTER13_ACTIONS = "computer13_actions"
    PYAUTOGUI_ACTIONS = "pyautogui_actions"
    DONE_OR_FAIL = "done_or_fail"
    REQUEST = "request"
    RESPONSE = "response"
    TASK = "task"


class ConditionPair(NamedTuple):
    """Pair of parse and validate functions for a condition.

    Attributes:
        parse_func: Function to parse the response.
        validate_func: Function to validate the response.
    """

    parse_func: Callable
    validate_func: Callable


# Pre-defined pairs of parse and validate functions
CONDITION_PAIRS: Dict[ConditionType, ConditionPair] = {
    ConditionType.COMPUTER13_ACTIONS: ConditionPair(
        parse_computer13_actions, validate_computer13_actions
    ),
    ConditionType.PYAUTOGUI_ACTIONS: ConditionPair(
        parse_pyautogui_actions, validate_pyautogui_actions
    ),
    ConditionType.DONE_OR_FAIL: ConditionPair(
        parse_done_or_fail, validate_done_or_fail
    ),
    ConditionType.REQUEST: ConditionPair(parse_request, validate_request),
    ConditionType.RESPONSE: ConditionPair(parse_response, validate_response),
    ConditionType.TASK: ConditionPair(parse_task, validate_task),
}


class EdgeCondition:
    """Condition for transitioning between nodes in the network with paired parsing and validation."""

    def __init__(
        self, condition_type: Union[ConditionType, str], description: str = ""
    ):
        """Initialize an edge condition with paired parser and validator.

        Args:
            condition_type: Type of condition that determines which function pair to use.
            description: Human-readable description of the condition.

        Raises:
            KeyError: If no function pair is found for the condition type.
        """
        if isinstance(condition_type, str):
            condition_type = ConditionType(condition_type)
        if condition_type not in CONDITION_PAIRS:
            raise KeyError(
                f"No function pair found for condition type {condition_type}"
            )
        self.function_pair = CONDITION_PAIRS[condition_type]
        self.description = description

    def parse(self, response: str) -> ParseResult:
        """Parse the response using the paired parse function.

        Args:
            response: The text content to parse.

        Returns:
            ParseResult containing match status and parsed content.
        """
        is_successful, content = self.function_pair.parse_func(response)
        return ParseResult(is_successful=is_successful, content=content)

    def validate(self, response: str) -> ValidationResult:
        """Validate the response using the paired validate function.

        Args:
            response: The text content to validate.

        Returns:
            ValidationResult containing validation status and any error.
        """
        is_valid, intended_format, error = self.function_pair.validate_func(response)
        return ValidationResult(
            is_valid=is_valid, intended_format=intended_format, error=error
        )


@dataclass
class NodeTransition:
    """Temporary storage for node transition data.

    Attributes:
        next_node: ID of the next node to transition to, if any.
        total_transitions: Counter for total transitions in current round.
    """

    next_node: Optional[str] = None
    total_transitions: int = 0


class NetworkDialogueGameMaster(DialogueGameMaster):
    """This class extends DialogueGameMaster, to implement a graph-based interaction model,
    where nodes represent game states or players, and edges represent transitions (standard and decision-based)
    """

    def __init__(
        self,
        name: str,
        path: str,
        experiment: dict,
        player_models: List[backends.Model],
    ):
        """Initialize the network dialogue game master.

        Args:
            name: The name of the game (as specified in game_registry).
            path: Path to the game (as specified in game_registry).
            experiment: The experiment (set of instances) to use.
            player_models: Player models to use for one or two players.
        """
        super().__init__(name, path, experiment, player_models)
        self.graph = nx.MultiDiGraph()
        self.graph.add_node("START", type=NodeType.START)
        self.graph.add_node("END", type=NodeType.END)
        self.current_node = "START"
        self.node_positions = None
        self.edge_labels = {}
        self.anchor_node = None
        self.current_round_nodes = []
        self.non_anchor_visited = False
        self.round_complete = False
        self.transition = NodeTransition()

    def add_player(
        self,
        player: RoleBasedPlayer,
        initial_prompt: Union[str, Dict] = None,
        initial_context: Union[str, Dict] = None,
        node_id: str = None,
    ):
        """Add a player to the game and the graph.

        Args:
            player: The player instance to add.
            initial_prompt: Initial prompt for the player, if any.
            initial_context: Initial context for the player, if any.
            node_id: Optional custom node ID for the player. If None, player's name is used.
        """
        super().add_player(player, initial_prompt, initial_context)
        node_id = node_id if node_id else player.name
        self.graph.add_node(node_id, type=NodeType.PLAYER, player=player)

    def add_standard_edge(self, from_node: str, to_node: str, label: str = ""):
        """Add a standard edge between nodes in the graph, ensuring only one standard edge exists.

        Standard edges are traversed when no decision edges are taken.

        Args:
            from_node: ID of the source node.
            to_node: ID of the target node.
            label: Optional label for the edge (for visualization).

        Raises:
            ValueError: If either node does not exist in the graph or a standard edge already exists.
        """
        if from_node not in self.graph:
            raise ValueError(f"Node '{from_node}' does not exist in the graph")
        if to_node not in self.graph:
            raise ValueError(f"Node '{to_node}' does not exist in the graph")
        existing_edges = [
            (u, v, data)
            for u, v, data in self.graph.edges(data=True)
            if u == from_node and v == to_node and data.get("type") == EdgeType.STANDARD
        ]
        if existing_edges:
            raise ValueError(
                f"A standard edge already exists from '{from_node}' to '{to_node}'"
            )
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

        Decision edges are traversed only if their condition is satisfied.

        Args:
            from_node: ID of the source node.
            to_node: ID of the target node.
            condition: Condition for the edge transition.
            label: Optional label for the edge (for visualization).

        Raises:
            ValueError: If either node does not exist in the graph.
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
        """Set the anchor node for round tracking.

        Args:
            node_id: ID of the node to set as anchor.

        Raises:
            ValueError: If the node does not exist in the graph.
        """
        if node_id not in self.graph:
            raise ValueError(f"Node '{node_id}' does not exist in the graph")
        self.anchor_node = node_id
        module_logger.info(f"Anchor node set to '{node_id}'")

    def _should_pass_turn(self) -> bool:
        """Check if transition to a different node is needed.

        Returns:
            bool: True if transition to a different node is required.
        """
        return (
            self.transition.next_node is not None
            and self.transition.next_node != self.current_node
        )

    def _next_player(self) -> RoleBasedPlayer:
        """Determine the next player based on graph transitions.

        Returns:
            RoleBasedPlayer: The player at the next node if next node; otherwise, the current player.
        """
        next_node = self.transition.next_node
        if next_node and next_node in self.graph:
            node_data = self.graph.nodes[next_node]
            if node_data["type"] == NodeType.END:
                self.current_node = next_node
                # _next_player expects to return a Player instance.
                # eventhough we transit to the END node, which is a non-Player node.
                return self.current_player
            elif node_data["type"] == NodeType.PLAYER:
                self.current_node = next_node
                return node_data["player"]
        return self.current_player

    def _start_next_round(self) -> bool:
        """Check if a new round should start based on anchor node.

        Returns:
            bool: True if a new round should start.
        """
        return self.round_complete

    def _on_after_round(self):
        """Handle post-round cleanup and preparation for the next round.

        This method performs cleanup by resetting round tracking state and
        preparing for the next round by logging the round transition.
        """
        self._reset_round_tracking()
        self.log_next_round()

    def _on_before_game(self):
        """Handle setup before entering the main play loop.

        Handles the initial transition from the START node,
        Extend this method for game-specific functionality

        Raises:
            ValueError: If START node has no standard edges.
        """
        standard_edges = [
            (_, to_node)
            for _, to_node, edge_data in self.graph.out_edges("START", data=True)
            if edge_data.get("type") == EdgeType.STANDARD
        ]
        if not standard_edges:
            raise ValueError("No standard edges found from START node")
        self.current_node = standard_edges[0][1]
        self._update_round_tracking("START", self.current_node)

    def _parse_response(self, player: RoleBasedPlayer, response: str) -> str:
        """Parse current-player response and determine the next node transition.

        This method processes the response using decision routing, falling back to standard edges if needed,
        and updates round tracking when node transitions occur.

        Args:
            player: Player instance that produced the response. (usually self.current_player)
            response: The response of the player.

        Returns:
            str: The parsed response.
        """
        self.transition = NodeTransition()
        _response, log_action, next_node, _ = self._parse_response_for_decision_routing(
            player, response
        )
        if next_node is None:
            next_node = self._get_standard_edges(self.current_node)
        if next_node:
            # Handles both cases: (a) self-loops and (b) transitions between nodes
            self._update_round_tracking(self.current_node, next_node)
            self.transition.next_node = next_node
        if _response != response and log_action:
            self.log_to_self("parse", f"Parsed response: {_response}")
        return _response

    def _parse_response_for_decision_routing(
        self, player: RoleBasedPlayer, response: str
    ) -> Tuple[str, bool, Optional[str], Optional[str]]:
        """Parse a response and evaluate decision edge conditions.

        This hook method is intended for subclass override to implement game-specific logic.

        Args:
            player: The player who produced the response.
            response: The text content of the response.

        Returns:
            Tuple containing:
                - Modified response (or original if unchanged).
                - Boolean indicating whether to log the parsing.
                - Next node ID if a decision edge is taken, None otherwise.
                - Extracted content, if any.

        Notes:
            Default implementation returns the response unchanged with no transition.
        """
        return response, True, None, None

    def _update_round_tracking(self, prev_node: str, next_node: str):
        """Update round tracking state based on node transitions.

        Tracks visited nodes, transitions count, and marks a round complete when returning
        to the anchor after other nodes, or when transitioning to the END node.

        Args:
            prev_node: Node being transitioned from.
            next_node: Node being transitioned to.
        """
        if self.anchor_node is None:
            return
        self.current_round_nodes.append(next_node)
        self.transition.total_transitions += 1
        if next_node != self.anchor_node:
            self.non_anchor_visited = True
        node_data = self.graph.nodes.get(next_node, {})
        is_end_node = node_data.get("type") == NodeType.END
        is_anchor_return = next_node == self.anchor_node and self.non_anchor_visited
        if is_end_node or is_anchor_return:
            self.round_complete = True

    def _reset_round_tracking(self):
        """Reset the round tracking state variables.

        Resets all round-related tracking variables to their initial states and
        adds the current node to the new round's tracking list if one exists.
        """
        round_path = " » ".join(str(node) for node in self.current_round_nodes)
        self.log_to_self(
            "round-complete",
            f"Round completed: {round_path} (Total transitions: {self.transition.total_transitions})",
        )
        self.current_round_nodes = []
        self.non_anchor_visited = False
        self.round_complete = False
        self.transition.total_transitions = 0
        if self.current_node:
            self.current_round_nodes.append(self.current_node)

    def get_player_from_node(self, node_id: str) -> Optional[RoleBasedPlayer]:
        """Get player associated with a node ID.

        Args:
            node_id: ID of the node to check.

        Returns:
            Optional[RoleBasedPlayer]: Player instance if node is a player node, None otherwise.
        """
        if node_id not in self.graph:
            return None
        node_data = self.graph.nodes[node_id]
        if node_data.get("type") == NodeType.PLAYER:
            return node_data.get("player")
        return None

    def get_node_from_player(self, player: RoleBasedPlayer) -> Optional[str]:
        """Get node ID associated with a player instance.

        Args:
            player: Player instance to find.

        Returns:
            Optional[str]: Node ID if player is found, None otherwise.
        """
        for node_id, node_data in self.graph.nodes(data=True):
            if (
                node_data.get("type") == NodeType.PLAYER
                and node_data.get("player") == player
            ):
                return node_id
        return None

    def _get_decision_edges(self, node_id: str) -> List[Tuple[str, EdgeCondition]]:
        """Retrieve all decision edges originating from a specified node.

        Args:
            node_id: ID of the source node.

        Returns:
            List[Tuple[str, EdgeCondition]]: A list of tuples, each containing the
                target node ID and the EdgeCondition object associated with the decision edge.

        Raises:
            ValueError: If the specified node_id does not exist in the graph.
        """
        if node_id not in self.graph:
            raise ValueError(f"Node '{node_id}' does not exist in the graph")
        decision_edges = [
            (to_node, edge_data["condition"])
            for _, to_node, edge_data in self.graph.out_edges(node_id, data=True)
            if edge_data.get("type") == EdgeType.DECISION and edge_data.get("condition")
        ]
        return decision_edges

    def _get_standard_edges(self, node_id: str) -> List[Tuple[str, Dict]]:
        """Retrieve all standard edges originating from a specified node.

        Args:
            node_id: ID of the source node.

        Returns:
            List[Tuple[str, Dict]]: A list of tuples, each containing the target node ID
                and the edge data dictionary for the standard edge.

        Raises:
            ValueError: If the specified node_id does not exist in the graph.
        """
        if node_id not in self.graph:
            raise ValueError(f"Node '{node_id}' does not exist in the graph")
        return [
            (to_node, edge_data)
            for _, to_node, edge_data in self.graph.out_edges(node_id, data=True)
            if edge_data.get("type") == EdgeType.STANDARD
        ]

    def visualize_graph(self, figsize=(12, 10), save_path=None, dpi=100):
        """Visualize the network structure with professional styling.

        Args:
            figsize: Tuple of (width, height) for figure size in inches.
            save_path: Optional path to save the visualization.
            dpi: Resolution for the output figure.
        """
        plt.figure(figsize=figsize, dpi=dpi)
        if not self.node_positions:
            try:
                self.node_positions = nx.nx_pydot.pydot_layout(self.graph, prog="dot")
            except Exception:
                self.node_positions = nx.spring_layout(
                    self.graph, k=0.5, iterations=100, seed=42
                )

        # Professional color palette
        node_colors = {
            NodeType.START: "#4CAF50",  # Muted green
            NodeType.PLAYER: "#0288D1",  # Deep blue
            NodeType.END: "#D81B60",  # Muted magenta
        }

        # Define node size - we'll use this value consistently
        base_node_size = 3300

        # Edges - Draw edges BEFORE nodes so nodes appear on top
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

        nx.draw_networkx_edges(
            self.graph,
            self.node_positions,
            edgelist=standard_edges,
            arrowsize=25,
            width=2.5,
            edge_color="#455A64",
            connectionstyle="arc3,rad=0.1",
            arrowstyle="-|>",
            node_size=base_node_size,
        )
        nx.draw_networkx_edges(
            self.graph,
            self.node_positions,
            edgelist=decision_edges,
            arrowsize=25,
            width=2,
            edge_color="#7B1FA2",
            style="dashed",
            connectionstyle="arc3,rad=0.1",
            arrowstyle="-|>",
            node_size=base_node_size,
        )

        # Now draw nodes on top of edges
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
                node_size=base_node_size,
                alpha=0.9,
                edgecolors="#37474F",
                linewidths=2,
            )

        # Node labels
        node_labels = {}
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get("type")
            if node_type == NodeType.PLAYER:
                player = self.graph.nodes[node].get("player")
                role_text = (
                    player.role
                    if hasattr(player, "role") and player.role
                    else str(player.model)
                )
                node_labels[node] = f"{node}\n({role_text})"
            else:
                node_labels[node] = node

        # Node labels drawing
        nx.draw_networkx_labels(
            self.graph,
            self.node_positions,
            labels=node_labels,
            font_size=10,
            font_family="sans-serif",
            font_weight="bold",
            font_color="#FFFFFF",
        )

        # Edge labels
        edge_labels_dict = {}
        for (u, v, k), label in self.edge_labels.items():
            if label:
                if len(label) > 20:
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

        nx.draw_networkx_edge_labels(
            self.graph,
            self.node_positions,
            edge_labels=edge_labels_dict,
            font_size=8,
            font_family="sans-serif",
            font_weight="normal",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
                boxstyle="round,pad=0.4",
            ),
            label_pos=0.4,
        )

        # Anchor node
        if self.anchor_node and self.anchor_node in self.graph:
            anchor_pos = {self.anchor_node: self.node_positions[self.anchor_node]}
            nx.draw_networkx_nodes(
                self.graph,
                anchor_pos,
                nodelist=[self.anchor_node],
                node_color="none",
                node_size=base_node_size * 1.1,
                alpha=1.0,
                edgecolors="#FBC02D",
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

        # Legend with increased padding
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
            plt.Line2D([0], [0], color="#455A64", linewidth=2.5, label="Standard Edge"),
            plt.Line2D(
                [0],
                [0],
                color="#7B1FA2",
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
                    markeredgecolor="#FBC02D",
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
            frameon=True,
            edgecolor="#37474F",
            facecolor="#FAFAFA",
            framealpha=0.9,
            bbox_transform=plt.gcf().transFigure,
            borderpad=1,
            labelspacing=1,
            handletextpad=1,
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
