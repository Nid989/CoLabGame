"""
Blackboard topology implementation with shared memory and round-robin turn-taking.
"""

import random
import logging
from typing import Dict, Any, List

from .base import BaseTopology, TopologyConfig, TopologyType
from src.message import MessagePermissions, MessageType

logger = logging.getLogger(__name__)


class BlackboardTopology(BaseTopology):
    """Blackboard topology implementation with shared memory."""

    def __init__(self):
        self.config = TopologyConfig(
            topology_type=TopologyType.BLACKBOARD,
            anchor_selection="random",  # Random node selection
            transition_strategy="round_robin",  # Round-robin after blackboard updates
            message_permissions={
                "executor": MessagePermissions(
                    send=[MessageType.EXECUTE, MessageType.STATUS, MessageType.WRITE_BOARD],
                    receive=[],  # No direct communication, only through blackboard
                )
            },
        )

    def generate_graph(self, participants: Dict) -> Dict:
        """Generate blackboard topology graph with round-robin role transitions."""
        self.validate_participants(participants)

        # Generate node sequence using actual participant role names
        node_sequence = []
        for role_name, config in participants.items():
            count = config["count"]
            for i in range(count):
                node_id = f"{role_name}_{i + 1}" if count > 1 else role_name
                node_sequence.append(node_id)

        nodes = [{"id": "START", "type": "START"}]
        nodes.extend({"id": node_id, "type": "PLAYER", "role_index": 0} for node_id in node_sequence)
        nodes.append({"id": "END", "type": "END"})

        edges = [{"from": "START", "to": node_id, "type": "STANDARD", "description": ""} for node_id in node_sequence]
        for i, node_id in enumerate(node_sequence):
            next_node_id = node_sequence[(i + 1) % len(node_sequence)]
            edges.append({"from": node_id, "to": node_id, "type": "DECISION", "condition": {"type": "EXECUTE"}, "description": "EXECUTE"})
            edges.append(
                {"from": node_id, "to": next_node_id, "type": "DECISION", "condition": {"type": "WRITE_BOARD"}, "description": "WRITE_BOARD"}
            )
            edges.append({"from": node_id, "to": "END", "type": "DECISION", "condition": {"type": "STATUS"}, "description": "STATUS"})

        anchor_node = random.choice(node_sequence)
        return {"nodes": nodes, "edges": edges, "anchor_node": anchor_node, "node_sequence": node_sequence}

    def validate_participants(self, participants: Dict) -> None:
        """Validate blackboard topology participants.

        Args:
            participants: Dictionary with participant configuration

        Raises:
            ValueError: If participant configuration is invalid for blackboard topology
        """
        if len(participants) < 1:
            raise ValueError("Blackboard topology requires at least 1 participant")

        total_executors = sum(config["count"] for config in participants.values())
        if total_executors < 2:
            raise ValueError("Blackboard topology requires at least 2 executors total")

    def get_config(self) -> TopologyConfig:
        """Return blackboard topology configuration.

        Returns:
            TopologyConfig instance for blackboard topology
        """
        return self.config

    def process_message(self, data: Dict, message_type: Any, player: Any, game_context: Dict) -> Dict:
        """Process blackboard topology message transitions.

        For WRITE_BOARD, sets the 'to' field to the next node in round-robin and logs the transition.
        For other message types, returns data unchanged.

        Args:
            data: Parsed JSON response data
            message_type: Type of message being processed
            player: Current player instance
            game_context: Dictionary containing game state context

        Returns:
            Dict: Data with topology-specific modifications
        """
        if message_type.name == "WRITE_BOARD":
            next_node_function = game_context.get("next_node_function")
            current_node = game_context.get("current_node")

            if next_node_function:
                next_node = next_node_function()
                if next_node:
                    data["to"] = next_node
                    logger.info(f"Blackboard: {player.name} at node {current_node} wrote to board, transitioning to {next_node}")
        return data

    def initialize_game_components(self, game_instance: Dict, game_config: Dict) -> Dict:
        """Initialize blackboard-specific components.

        Sets up the blackboard manager, node sequence for round-robin, and writes
        the initial goal to the blackboard.

        Args:
            game_instance: Dictionary containing game instance configuration
            game_config: Dictionary containing game configuration

        Returns:
            Dict: Dictionary of component names to component instances
        """
        from src.utils.blackboard_manager import BlackboardManager

        # Create blackboard manager
        blackboard_manager = BlackboardManager()

        # Get node_sequence from graph metadata
        graph_config = game_instance.get("graph", {})
        node_sequence = graph_config.get("node_sequence", [])

        # Write the goal to the blackboard
        goal = game_instance.get("task_config", {}).get("instruction")
        if goal:
            blackboard_manager.write_content(role_id="Goal", content=goal)

        return {"blackboard_manager": blackboard_manager, "node_sequence": node_sequence}

    def get_template_name(self, role_name: str) -> str:
        """Get template name for blackboard topology roles.

        Args:
            role_name: Name of the role (e.g., 'executor_1')

        Returns:
            str: Template filename to use for this role
        """
        base_role = role_name.split("_")[0] if "_" in role_name else role_name

        if base_role == "executor":
            return "blackboard_topology_executor_prompt.j2"
        else:
            # Fallback to default implementation
            return super().get_template_name(role_name)

    def validate_experiment_config(self, experiment_config: Dict) -> List[str]:
        """Validate experiment configuration for blackboard topology.

        Args:
            experiment_config: Dictionary containing experiment configuration

        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        participants = experiment_config.get("participants", {})

        # Check for at least one participant
        if len(participants) < 1:
            errors.append("Blackboard topology requires at least 1 participant")

        # Check total executor count across all participants
        total_executors = sum(config.get("count", 0) for config in participants.values())
        if total_executors < 2:
            errors.append(f"Blackboard topology requires at least 2 executors total for meaningful collaboration, got {total_executors}")

        return errors
