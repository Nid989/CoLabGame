"""
Star topology implementation with advisor as central coordinator.
"""

from typing import Dict

from .base import BaseTopology, TopologyConfig, TopologyType
from src.message import MessagePermissions, MessageType


class StarTopology(BaseTopology):
    """Star topology implementation with advisor as central coordinator."""

    def __init__(self):
        self.config = TopologyConfig(
            topology_type=TopologyType.STAR,
            anchor_selection="fixed",  # Advisor is always anchor
            transition_strategy="conditional",  # Based on message types
            message_permissions={
                "advisor": MessagePermissions(
                    send=[MessageType.REQUEST, MessageType.RESPONSE, MessageType.STATUS], receive=[MessageType.REQUEST, MessageType.RESPONSE]
                ),
                "executor": MessagePermissions(
                    send=[MessageType.EXECUTE, MessageType.REQUEST, MessageType.RESPONSE], receive=[MessageType.REQUEST, MessageType.RESPONSE]
                ),
            },
        )

    def generate_graph(self, participants: Dict) -> Dict:
        """Generate star topology graph.

        Args:
            participants: Dictionary with participant configuration

        Returns:
            Dict containing graph configuration with nodes, edges, and anchor_node
        """
        self.validate_participants(participants)

        executor_count = participants["executor"]["count"]

        # Create nodes
        nodes = [{"id": "START", "type": "START"}, {"id": "advisor", "type": "PLAYER", "role_index": 0}]

        # Add executor nodes
        for i in range(executor_count):
            executor_id = f"executor_{i + 1}" if executor_count > 1 else "executor"
            nodes.append({"id": executor_id, "type": "PLAYER", "role_index": 1})

        nodes.append({"id": "END", "type": "END"})

        # Create star topology edges
        edges = [{"from": "START", "to": "advisor", "type": "STANDARD", "description": ""}]

        # Add bidirectional communication between advisor and each executor
        for i in range(executor_count):
            executor_id = f"executor_{i + 1}" if executor_count > 1 else "executor"

            # Advisor ↔ Executor communication
            edges.extend(
                [
                    {"from": "advisor", "to": executor_id, "type": "DECISION", "condition": {"type": "REQUEST"}, "description": "REQUEST"},
                    {"from": "advisor", "to": executor_id, "type": "DECISION", "condition": {"type": "RESPONSE"}, "description": "RESPONSE"},
                    {"from": executor_id, "to": "advisor", "type": "DECISION", "condition": {"type": "REQUEST"}, "description": "REQUEST"},
                    {"from": executor_id, "to": "advisor", "type": "DECISION", "condition": {"type": "RESPONSE"}, "description": "RESPONSE"},
                ]
            )

            # Executor self-loop for EXECUTE
            edges.append({"from": executor_id, "to": executor_id, "type": "DECISION", "condition": {"type": "EXECUTE"}, "description": "EXECUTE"})

        # Advisor to END for goal completion
        edges.append({"from": "advisor", "to": "END", "type": "DECISION", "condition": {"type": "STATUS"}, "description": "STATUS"})

        return {"nodes": nodes, "edges": edges, "anchor_node": "advisor"}

    def validate_participants(self, participants: Dict) -> None:
        """Validate star topology participants.

        Args:
            participants: Dictionary with participant configuration

        Raises:
            ValueError: If participant configuration is invalid for star topology
        """
        if "advisor" not in participants:
            raise ValueError("Star topology requires an 'advisor' role")

        if participants["advisor"]["count"] != 1:
            raise ValueError("Star topology requires exactly 1 advisor")

        if "executor" not in participants:
            raise ValueError("Star topology requires at least one 'executor' role")

    def get_config(self) -> TopologyConfig:
        """Return star topology configuration.

        Returns:
            TopologyConfig instance for star topology
        """
        return self.config
