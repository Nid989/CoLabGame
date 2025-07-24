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
        """Generate star topology graph with advisor as central coordinator."""
        self.validate_participants(participants)
        executor_count = participants["executor"]["count"]
        nodes = [{"id": "START", "type": "START"}, {"id": "advisor", "type": "PLAYER", "role_index": 0}]
        executor_ids = [f"executor_{i + 1}" if executor_count > 1 else "executor" for i in range(executor_count)]
        nodes.extend({"id": eid, "type": "PLAYER", "role_index": 1} for eid in executor_ids)
        nodes.append({"id": "END", "type": "END"})
        edges = [{"from": "START", "to": "advisor", "type": "STANDARD", "description": ""}]
        for eid in executor_ids:
            edges.extend(
                [
                    {"from": "advisor", "to": eid, "type": "DECISION", "condition": {"type": "REQUEST"}, "description": "REQUEST"},
                    {"from": "advisor", "to": eid, "type": "DECISION", "condition": {"type": "RESPONSE"}, "description": "RESPONSE"},
                    {"from": eid, "to": "advisor", "type": "DECISION", "condition": {"type": "REQUEST"}, "description": "REQUEST"},
                    {"from": eid, "to": "advisor", "type": "DECISION", "condition": {"type": "RESPONSE"}, "description": "RESPONSE"},
                    {"from": eid, "to": eid, "type": "DECISION", "condition": {"type": "EXECUTE"}, "description": "EXECUTE"},
                ]
            )
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
