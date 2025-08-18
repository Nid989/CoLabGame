"""
Single agent topology implementation.
"""

import logging
from typing import Dict, Any, List

from .base import BaseTopology, TopologyConfig, TopologyType

logger = logging.getLogger(__name__)


class SingleTopology(BaseTopology):
    """Single agent topology implementation."""

    def __init__(self):
        self.config = TopologyConfig(
            topology_type=TopologyType.SINGLE,
            anchor_selection="fixed",  # Single agent is always anchor
            transition_strategy="conditional",  # Based on message types
            message_permissions={},  # Will be populated from topology config
        )

    def generate_graph(self, participants: Dict) -> Dict:
        """Generate single agent graph.

        Args:
            participants: Dictionary with participant configuration

        Returns:
            Dict containing graph configuration with nodes, edges, and anchor_node
        """
        self.validate_participants(participants)

        # Get the single role name from participants (should be only one)
        role_name = list(participants.keys())[0]

        # Create nodes
        nodes = [{"id": "START", "type": "START"}, {"id": role_name, "type": "PLAYER", "role_index": 0}, {"id": "END", "type": "END"}]

        # Create edges
        edges = [
            {"from": "START", "to": role_name, "type": "STANDARD", "description": ""},
            {"from": role_name, "to": role_name, "type": "DECISION", "condition": {"type": "EXECUTE"}, "description": "EXECUTE"},
            {"from": role_name, "to": "END", "type": "STANDARD", "description": ""},
        ]

        return {
            "nodes": nodes,
            "edges": edges,
            "anchor_node": role_name,
            "domain_definitions": getattr(self, "topology_config", {}).get("domain_definitions", {}),  # For template manager
        }

    def validate_participants(self, participants: Dict) -> None:
        """Validate single agent participants.

        Args:
            participants: Dictionary with participant configuration

        Raises:
            ValueError: If participant configuration is invalid for single agent topology
        """
        if len(participants) != 1:
            raise ValueError("Single agent topology requires exactly one participant")

        # Get the single role name and validate its count
        role_name = list(participants.keys())[0]
        role_config = participants[role_name]

        if role_config.get("count", 0) != 1:
            raise ValueError(f"Single agent topology requires exactly 1 {role_name}, got {role_config.get('count', 0)}")

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
            role_name: Name of the role (e.g., 'participant_w_execute')

        Returns:
            str: Template filename to use for this role
        """
        # For single topology, always use the single executor prompt regardless of role name
        return "single_topology_executor_prompt.j2"

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
            return errors

        # Get the single role and validate its count
        role_name = list(participants.keys())[0]
        role_config = participants[role_name]
        role_count = role_config.get("count", 0)

        if role_count != 1:
            errors.append(f"Single topology requires exactly 1 {role_name}, got {role_count}")

        return errors
