"""
Star topology implementation with advisor as central coordinator.
"""

import logging
from typing import Dict, Any, List

from .base import BaseTopology, TopologyConfig, TopologyType
from src.message import MessagePermissions, MessageType

logger = logging.getLogger(__name__)


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

    def process_message(self, data: Dict, message_type: Any, player: Any, game_context: Dict) -> Dict:
        """Process star topology message transitions.

        Star topology uses the existing edge conditions and graph structure for transitions.
        No additional processing is needed as the hub-and-spoke communication pattern
        is handled by the graph edges.

        Args:
            data: Parsed JSON response data
            message_type: Type of message being processed
            player: Current player instance
            game_context: Dictionary containing game state context

        Returns:
            Dict: Data unchanged (star uses graph-based transitions)
        """
        # Star topology doesn't need special message processing
        # All communication flows through the advisor hub via graph edges
        return data

    def get_template_name(self, role_name: str) -> str:
        """Get template name for star topology roles.

        Args:
            role_name: Name of the role (e.g., 'advisor', 'executor_1')

        Returns:
            str: Template filename to use for this role
        """
        base_role = role_name.split("_")[0] if "_" in role_name else role_name

        if base_role == "advisor":
            return "star_topology_advisor_prompt.j2"
        elif base_role == "executor":
            return "star_topology_executor_prompt.j2"
        else:
            # Fallback to default implementation
            return super().get_template_name(role_name)

    def validate_experiment_config(self, experiment_config: Dict) -> List[str]:
        """Validate experiment configuration for star topology.

        Args:
            experiment_config: Dictionary containing experiment configuration

        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        participants = experiment_config.get("participants", {})

        # Check for advisor participant
        if "advisor" not in participants:
            errors.append("Star topology requires an 'advisor' participant")
        else:
            advisor_count = participants["advisor"].get("count", 0)
            if advisor_count != 1:
                errors.append(f"Star topology requires exactly 1 advisor, got {advisor_count}")

        # Check for executor participant
        if "executor" not in participants:
            errors.append("Star topology requires at least one 'executor' participant")
        else:
            executor_count = participants["executor"].get("count", 0)
            if executor_count < 1:
                errors.append(f"Star topology requires at least 1 executor, got {executor_count}")

        return errors
