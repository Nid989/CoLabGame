"""
Mesh topology implementation with peer-to-peer communication and negotiated transitions.
"""

import random
import logging
from typing import Dict, Any, List

from .base import BaseTopology, TopologyConfig, TopologyType
from src.message import MessagePermissions, MessageType

logger = logging.getLogger(__name__)


class MeshTopology(BaseTopology):
    """Mesh topology implementation with full peer-to-peer communication."""

    def __init__(self):
        self.config = TopologyConfig(
            topology_type=TopologyType.MESH,
            anchor_selection="random",  # Random anchor selection like blackboard
            transition_strategy="negotiated",  # Peer-decided transitions
            message_permissions={
                "executor": MessagePermissions(
                    send=[MessageType.EXECUTE, MessageType.REQUEST, MessageType.RESPONSE, MessageType.STATUS],
                    receive=[MessageType.REQUEST, MessageType.RESPONSE],
                )
            },
        )

    def generate_graph(self, participants: Dict) -> Dict:
        """Generate mesh topology graph with full peer-to-peer connectivity."""
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

        edges = []

        # Standard edges from START to all executors
        edges.extend([{"from": "START", "to": node_id, "type": "STANDARD", "description": ""} for node_id in node_sequence])

        # Create full mesh connectivity between all executors
        for i, node_id in enumerate(node_sequence):
            # Self-loop for EXECUTE
            edges.append({"from": node_id, "to": node_id, "type": "DECISION", "condition": {"type": "EXECUTE"}, "description": "EXECUTE"})

            # Peer-to-peer REQUEST/RESPONSE edges to all other executors
            for j, other_node_id in enumerate(node_sequence):
                if i != j:  # Don't create REQUEST/RESPONSE self-loops
                    edges.append(
                        {"from": node_id, "to": other_node_id, "type": "DECISION", "condition": {"type": "REQUEST"}, "description": "REQUEST"}
                    )
                    edges.append(
                        {"from": node_id, "to": other_node_id, "type": "DECISION", "condition": {"type": "RESPONSE"}, "description": "RESPONSE"}
                    )

            # STATUS edge to END from each executor
            edges.append({"from": node_id, "to": "END", "type": "DECISION", "condition": {"type": "STATUS"}, "description": "STATUS"})

        # Random anchor selection
        anchor_node = random.choice(node_sequence)

        return {"nodes": nodes, "edges": edges, "anchor_node": anchor_node}

    def validate_participants(self, participants: Dict) -> None:
        """Validate mesh topology participants.

        Args:
            participants: Dictionary with participant configuration

        Raises:
            ValueError: If participant configuration is invalid for mesh topology
        """
        if len(participants) < 1:
            raise ValueError("Mesh topology requires at least 1 participant")

        total_executors = sum(config["count"] for config in participants.values())
        if total_executors < 2:
            raise ValueError("Mesh topology requires at least 2 executors total for meaningful peer-to-peer communication")

    def get_config(self) -> TopologyConfig:
        """Return mesh topology configuration.

        Returns:
            TopologyConfig instance for mesh topology
        """
        return self.config

    def process_message(self, data: Dict, message_type: Any, player: Any, game_context: Dict) -> Dict:
        """Process mesh topology message transitions.

        Mesh topology allows flexible peer-to-peer communication where executors coordinate
        their own handoffs through REQUEST/RESPONSE messages. No additional processing
        is needed as 'to' fields are already set by the players for peer communication.

        Args:
            data: Parsed JSON response data
            message_type: Type of message being processed
            player: Current player instance
            game_context: Dictionary containing game state context

        Returns:
            Dict: Data unchanged (mesh uses direct peer communication)
        """
        current_node = game_context.get("current_node")
        logger.info(f"Mesh: {player.name} at node {current_node} sending {message_type.name} message")
        return data

    def get_template_name(self, role_name: str) -> str:
        """Get template name for mesh topology roles.

        Args:
            role_name: Name of the role (e.g., 'executor_1')

        Returns:
            str: Template filename to use for this role
        """
        base_role = role_name.split("_")[0] if "_" in role_name else role_name

        if base_role == "executor":
            return "mesh_topology_executor_prompt.j2"
        else:
            # Fallback to default implementation
            return super().get_template_name(role_name)

    def validate_experiment_config(self, experiment_config: Dict) -> List[str]:
        """Validate experiment configuration for mesh topology.

        Args:
            experiment_config: Dictionary containing experiment configuration

        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        participants = experiment_config.get("participants", {})

        # Check for at least one participant
        if len(participants) < 1:
            errors.append("Mesh topology requires at least 1 participant")

        # Check total executor count across all participants
        total_executors = sum(config.get("count", 0) for config in participants.values())
        if total_executors < 2:
            errors.append(f"Mesh topology requires at least 2 executors total for meaningful peer-to-peer communication, got {total_executors}")

        return errors
