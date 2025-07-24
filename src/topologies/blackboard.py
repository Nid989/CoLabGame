"""
Blackboard topology implementation with shared memory and round-robin turn-taking.
"""

import random
from typing import Dict

from .base import BaseTopology, TopologyConfig, TopologyType
from src.message import MessagePermissions, MessageType


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
