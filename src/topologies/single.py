"""
Single agent topology implementation.
"""

import logging
from typing import Dict, Any, List

from .base import BaseTopology, TopologyConfig, TopologyType
from src.message import MessagePermissions, MessageType

logger = logging.getLogger(__name__)


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

    def process_message(self, data: Dict, message_type: Any, player: Any, game_context: Dict) -> Dict:
        """Process single agent topology message transitions.

        Single agent topology uses simple self-transitions for EXECUTE messages
        and standard transitions for STATUS. No additional processing is needed
        as there's only one agent.

        Args:
            data: Parsed JSON response data
            message_type: Type of message being processed
            player: Current player instance
            game_context: Dictionary containing game state context

        Returns:
            Dict: Data unchanged (single agent uses simple transitions)
        """
        # Single agent topology doesn't need special message processing
        # Only EXECUTE (self-loop) and STATUS (to END) messages are used
        return data

    def get_template_name(self, role_name: str) -> str:
        """Get template name for single topology roles.

        Args:
            role_name: Name of the role (e.g., 'executor')

        Returns:
            str: Template filename to use for this role
        """
        base_role = role_name.split("_")[0] if "_" in role_name else role_name

        if base_role == "executor":
            return "single_topology_executor_prompt.j2"
        else:
            # Fallback to default implementation
            return super().get_template_name(role_name)

    def validate_experiment_config(self, experiment_config: Dict) -> List[str]:
        """Validate experiment configuration for single topology.

        Args:
            experiment_config: Dictionary containing experiment configuration

        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        participants = experiment_config.get("participants", {})

        # Check for exactly one participant type
        if len(participants) != 1:
            errors.append(f"Single topology requires exactly one participant type, got {len(participants)}")

        # Check for executor participant
        if "executor" not in participants:
            errors.append("Single topology requires an 'executor' participant")
        else:
            executor_count = participants["executor"].get("count", 0)
            if executor_count != 1:
                errors.append(f"Single topology requires exactly 1 executor, got {executor_count}")

        return errors
