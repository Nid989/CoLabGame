"""
Single agent topology implementation.
"""

from typing import Dict

from .base import BaseTopology, TopologyConfig, TopologyType
from src.message import MessagePermissions, MessageType


class SingleTopology(BaseTopology):
    """Single agent topology implementation."""

    def __init__(self):
        self.config = TopologyConfig(
            topology_type=TopologyType.SINGLE,
            anchor_selection="fixed",  # Executor is always anchor
            transition_strategy="conditional",  # Based on message types
            message_permissions={"executor": MessagePermissions(send=[MessageType.EXECUTE, MessageType.STATUS], receive=[])},
        )

    def generate_graph(self, participants: Dict) -> Dict:
        """Generate single agent graph.

        Args:
            participants: Dictionary with participant configuration

        Returns:
            Dict containing graph configuration with nodes, edges, and anchor_node
        """
        self.validate_participants(participants)

        # Create nodes
        nodes = [{"id": "START", "type": "START"}, {"id": "executor", "type": "PLAYER", "role_index": 0}, {"id": "END", "type": "END"}]

        # Create edges
        edges = [
            {"from": "START", "to": "executor", "type": "STANDARD", "description": ""},
            {"from": "executor", "to": "executor", "type": "DECISION", "condition": {"type": "EXECUTE"}, "description": "EXECUTE"},
            {"from": "executor", "to": "END", "type": "STANDARD", "description": ""},
        ]

        return {"nodes": nodes, "edges": edges, "anchor_node": "executor"}

    def validate_participants(self, participants: Dict) -> None:
        """Validate single agent participants.

        Args:
            participants: Dictionary with participant configuration

        Raises:
            ValueError: If participant configuration is invalid for single agent topology
        """
        if len(participants) != 1:
            raise ValueError("Single agent topology requires exactly one participant")

        if "executor" not in participants:
            raise ValueError("Single agent topology requires an 'executor' role")

        if participants["executor"]["count"] != 1:
            raise ValueError("Single agent topology requires exactly 1 executor")

    def get_config(self) -> TopologyConfig:
        """Return single agent topology configuration.

        Returns:
            TopologyConfig instance for single agent topology
        """
        return self.config
