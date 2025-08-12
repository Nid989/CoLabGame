"""
Mesh topology implementation with peer-to-peer communication and negotiated transitions.
"""

import random
import logging
from typing import Dict, Any, List, Tuple

from .base import BaseTopology, TopologyConfig, TopologyType
from src.message import MessagePermissions, MessageType

logger = logging.getLogger(__name__)


class MeshTopology(BaseTopology):
    """Mesh topology implementation with dynamic participant configuration and full peer-to-peer communication."""

    def __init__(self):
        # Initialize with minimal config, will be populated by load_game_instance_config
        self.config = TopologyConfig(
            topology_type=TopologyType.MESH,
            anchor_selection="random",  # Random anchor selection
            transition_strategy="negotiated",  # Peer-decided transitions
            message_permissions={},  # Will be populated dynamically
        )
        self.topology_config = None

    def generate_graph(self, participants: Dict) -> Dict:
        """Generate mesh topology graph using dynamic configuration and algorithmic generation."""
        if not self.topology_config:
            raise ValueError("Topology configuration not loaded. Call load_game_instance_config first.")

        # Map legacy participants to topology roles
        participant_assignments = self._map_participants_to_roles(participants)

        # Validate the mapped participants
        self.validate_participants(participant_assignments)

        # Create node assignments with role indices and domains
        node_assignments = self._create_node_assignments(participant_assignments)

        # Generate graph structure algorithmically
        nodes, edges = self._generate_mesh_structure(node_assignments)

        # Random anchor selection from all participants
        all_participant_nodes = []
        for role_nodes in node_assignments.values():
            all_participant_nodes.extend([node["node_id"] for node in role_nodes])

        anchor_node = random.choice(all_participant_nodes) if all_participant_nodes else None

        return {
            "nodes": nodes,
            "edges": edges,
            "anchor_node": anchor_node,
            "node_assignments": node_assignments,  # For role creation in master.py
        }

    def _map_participants_to_roles(self, participants: Dict) -> Dict:
        """Map legacy participant format to topology roles using legacy mapping."""
        if not self.topology_config:
            raise ValueError("Topology configuration not loaded")

        legacy_mapping = self.topology_config.get("legacy_mapping", {})
        default_assignments = self.topology_config.get("default_participant_assignments", {})

        mapped_assignments = {}

        # Map legacy participants to topology roles
        for legacy_role, participant_config in participants.items():
            if legacy_role in legacy_mapping:
                topology_role = legacy_mapping[legacy_role]
                mapped_assignments[topology_role] = participant_config
            else:
                mapped_assignments[legacy_role] = participant_config

        # If no mappings were found, use default assignments
        if not mapped_assignments and default_assignments:
            logger.info("No participant mappings found, using default assignments")
            mapped_assignments = default_assignments.copy()

        return mapped_assignments

    def _create_node_assignments(self, participant_assignments: Dict) -> Dict:
        """Create node assignments with role indices and domains."""
        node_assignments = {}
        role_index = 0

        for role_name, assignment in participant_assignments.items():
            count = assignment["count"]
            domains = assignment.get("domains", [])

            role_nodes = []
            for i in range(count):
                node_id = f"{role_name}_{i + 1}" if count > 1 else role_name
                domain = domains[i] if i < len(domains) else (domains[0] if domains else f"general_{role_name}")

                role_nodes.append(
                    {
                        "node_id": node_id,
                        "role_index": role_index,
                        "domain": domain,
                        "topology_role": role_name,
                    }
                )

            node_assignments[role_name] = role_nodes
            role_index += 1

        return node_assignments

    def _generate_mesh_structure(self, node_assignments: Dict) -> Tuple[List, List]:
        """Generate mesh topology structure algorithmically with full peer-to-peer connectivity."""
        nodes = [{"id": "START", "type": "START"}]
        edges = []

        # Get all participant nodes
        participant_w_execute_nodes = node_assignments.get("participant_w_execute", [])
        participant_wo_execute_nodes = node_assignments.get("participant_wo_execute", [])

        all_participant_nodes = participant_w_execute_nodes + participant_wo_execute_nodes

        # Add all nodes to graph
        for role_name, role_nodes in node_assignments.items():
            for node in role_nodes:
                nodes.append(
                    {
                        "id": node["node_id"],
                        "type": "PLAYER",
                        "role_index": node["role_index"],
                        "domain": node["domain"],
                        "topology_role": node["topology_role"],
                    }
                )

        nodes.append({"id": "END", "type": "END"})

        # START edges to all participants
        for node in all_participant_nodes:
            edges.append(
                {
                    "from": "START",
                    "to": node["node_id"],
                    "type": "STANDARD",
                    "description": "",
                }
            )

        # MESH Algorithm: Full peer-to-peer connectivity
        for i, node in enumerate(all_participant_nodes):
            # Peer-to-peer REQUEST/RESPONSE edges to all other participants
            for j, other_node in enumerate(all_participant_nodes):
                if i != j:  # Don't create REQUEST/RESPONSE self-loops
                    edges.extend(
                        [
                            {
                                "from": node["node_id"],
                                "to": other_node["node_id"],
                                "type": "DECISION",
                                "condition": {"type": "REQUEST"},
                                "description": "REQUEST",
                            },
                            {
                                "from": node["node_id"],
                                "to": other_node["node_id"],
                                "type": "DECISION",
                                "condition": {"type": "RESPONSE"},
                                "description": "RESPONSE",
                            },
                        ]
                    )

            # STATUS transition to END
            edges.append(
                {
                    "from": node["node_id"],
                    "to": "END",
                    "type": "DECISION",
                    "condition": {"type": "STATUS"},
                    "description": "STATUS",
                }
            )

        # EXECUTE self-loops only for participants with execute permissions
        for node in participant_w_execute_nodes:
            edges.append(
                {
                    "from": node["node_id"],
                    "to": node["node_id"],
                    "type": "DECISION",
                    "condition": {"type": "EXECUTE"},
                    "description": "EXECUTE",
                }
            )

        return nodes, edges

    def validate_participants(self, participant_assignments: Dict) -> None:
        """Validate mesh topology participant assignments.

        Args:
            participant_assignments: Dictionary with topology role assignments

        Raises:
            ValueError: If participant configuration is invalid for mesh topology
        """
        if len(participant_assignments) < 1:
            raise ValueError("Mesh topology requires at least 1 participant role")

        # Count total participants across all roles
        total_participants = sum(assignment["count"] for assignment in participant_assignments.values())
        if total_participants < 2:
            raise ValueError("Mesh topology requires at least 2 participants total for meaningful peer-to-peer communication")

    def get_config(self) -> TopologyConfig:
        """Return mesh topology configuration.

        Returns:
            TopologyConfig instance for mesh topology
        """
        # Build message permissions dynamically from loaded topology config
        if self.topology_config:
            self._build_dynamic_permissions()
        return self.config

    def _build_dynamic_permissions(self) -> None:
        """Build message permissions dynamically from topology configuration."""
        if not self.topology_config:
            return

        role_definitions = self.topology_config.get("role_definitions", {})
        message_permissions = {}

        for role_name, role_config in role_definitions.items():
            permissions = role_config.get("message_permissions", {})
            send_types = [MessageType.from_string(mt) for mt in permissions.get("send", [])]
            receive_types = [MessageType.from_string(mt) for mt in permissions.get("receive", [])]

            message_permissions[role_name] = MessagePermissions(
                send=send_types,
                receive=receive_types,
            )

        # Update the config with dynamic permissions
        self.config.message_permissions = message_permissions

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

        if base_role == "participant":
            # Both participant_w_execute and participant_wo_execute use participant template
            return "mesh_topology_participant_prompt.j2"
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
