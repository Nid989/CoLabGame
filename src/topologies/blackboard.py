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
            anchor_selection="random",  # Random agent selection
            transition_strategy="round_robin",  # Round-robin after blackboard updates
            message_permissions={
                "executor": MessagePermissions(
                    send=[MessageType.EXECUTE, MessageType.STATUS, MessageType.WRITE_BOARD],
                    receive=[],  # No direct communication, only through blackboard
                )
            },
        )

    def generate_graph(self, participants: Dict) -> Dict:
        """Generate blackboard topology graph with round-robin executor transitions."""
        self.validate_participants(participants)
        total_executors = sum(config["count"] for config in participants.values())
        nodes = [{"id": "START", "type": "START"}]
        executor_ids = [f"executor_{i + 1}" for i in range(total_executors)]
        nodes.extend({"id": eid, "type": "PLAYER", "role_index": i} for i, eid in enumerate(executor_ids))
        nodes.append({"id": "END", "type": "END"})
        edges = [{"from": "START", "to": eid, "type": "STANDARD", "description": ""} for eid in executor_ids]
        for i, eid in enumerate(executor_ids):
            next_eid = executor_ids[(i + 1) % total_executors]
            edges.append({"from": eid, "to": eid, "type": "DECISION", "condition": {"type": "EXECUTE"}, "description": "EXECUTE"})
            edges.append({"from": eid, "to": next_eid, "type": "DECISION", "condition": {"type": "WRITE_BOARD"}, "description": "WRITE_BOARD"})
            edges.append({"from": eid, "to": "END", "type": "DECISION", "condition": {"type": "STATUS"}, "description": "STATUS"})
        anchor_executor = random.choice(executor_ids)
        return {"nodes": nodes, "edges": edges, "anchor_node": anchor_executor}

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
