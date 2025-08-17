"""
Star topology implementation with dynamic role configuration.
"""

import logging
import os
import yaml
from typing import Dict, Any, List, Tuple

from .base import BaseTopology, TopologyConfig, TopologyType
from src.message import MessagePermissions, MessageType

logger = logging.getLogger(__name__)


class StarTopology(BaseTopology):
    """Star topology implementation with dynamic hub-and-spoke configuration."""

    def __init__(self):
        # Initialize with minimal config, will be populated by load_game_instance_config
        self.config = TopologyConfig(
            topology_type=TopologyType.STAR,
            anchor_selection="fixed",  # Hub is always anchor
            transition_strategy="conditional",  # Based on message types
            message_permissions={},  # Will be populated dynamically
        )
        self.topology_config = None

    def generate_graph(self, participants: Dict) -> Dict:
        """Generate star topology graph using dynamic configuration and algorithmic generation."""
        if not self.topology_config:
            raise ValueError("Topology configuration not loaded. Call load_game_instance_config first.")

        # Map legacy participants to topology roles
        participant_assignments = self._map_participants_to_roles(participants)

        # Validate the mapped participants
        self.validate_participants(participant_assignments)

        # Create node assignments with role indices and domains
        node_assignments = self._create_node_assignments(participant_assignments)

        # Generate graph structure algorithmically
        nodes, edges = self._generate_star_structure(node_assignments)

        # Determine anchor (hub node)
        hub_nodes = node_assignments.get("hub", [])
        if not hub_nodes:
            raise ValueError("Star topology requires exactly one hub node")
        hub_node = hub_nodes[0]["node_id"]

        return {
            "nodes": nodes,
            "edges": edges,
            "anchor_node": hub_node,
            "node_assignments": node_assignments,  # For role creation in master.py
            "domain_definitions": self.topology_config.get("domain_definitions", {}),  # For template manager
        }

    def _map_participants_to_roles(self, participants: Dict) -> Dict:
        """Map legacy participant format to topology roles using legacy mapping."""
        if not self.topology_config:
            raise ValueError("Topology configuration not loaded")

        legacy_mapping = self.topology_config.get("legacy_mapping", {})

        # Use default participant assignments if available
        default_assignments = self.topology_config.get("default_participant_assignments", {})

        mapped_assignments = {}

        # Map legacy participants to topology roles
        for legacy_role, participant_config in participants.items():
            if legacy_role in legacy_mapping:
                topology_role = legacy_mapping[legacy_role]
                mapped_assignments[topology_role] = participant_config
            else:
                # If no mapping found, use the role as-is (for new topology roles)
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

    def _generate_star_structure(self, node_assignments: Dict) -> Tuple[List, List]:
        """Generate star topology structure algorithmically."""
        nodes = [{"id": "START", "type": "START"}]
        edges = []

        # Get all role nodes
        hub_nodes = node_assignments.get("hub", [])
        spoke_w_execute_nodes = node_assignments.get("spoke_w_execute", [])
        spoke_wo_execute_nodes = node_assignments.get("spoke_wo_execute", [])

        all_spoke_nodes = spoke_w_execute_nodes + spoke_wo_execute_nodes

        if not hub_nodes:
            raise ValueError("Star topology requires exactly one hub node")
        if len(hub_nodes) > 1:
            raise ValueError("Star topology can have only one hub node")

        hub_node = hub_nodes[0]

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

        # START -> hub connection
        edges.append(
            {
                "from": "START",
                "to": hub_node["node_id"],
                "type": "STANDARD",
                "description": "",
            }
        )

        # STAR Algorithm: Hub bidirectional communication with all spokes
        for spoke_node in all_spoke_nodes:
            edges.extend(
                [
                    {
                        "from": hub_node["node_id"],
                        "to": spoke_node["node_id"],
                        "type": "DECISION",
                        "condition": {"type": "REQUEST"},
                        "description": "REQUEST",
                    },
                    {
                        "from": hub_node["node_id"],
                        "to": spoke_node["node_id"],
                        "type": "DECISION",
                        "condition": {"type": "RESPONSE"},
                        "description": "RESPONSE",
                    },
                    {
                        "from": spoke_node["node_id"],
                        "to": hub_node["node_id"],
                        "type": "DECISION",
                        "condition": {"type": "REQUEST"},
                        "description": "REQUEST",
                    },
                    {
                        "from": spoke_node["node_id"],
                        "to": hub_node["node_id"],
                        "type": "DECISION",
                        "condition": {"type": "RESPONSE"},
                        "description": "RESPONSE",
                    },
                ]
            )

        # EXECUTE self-loops only for spokes with execute permissions
        for spoke_node in spoke_w_execute_nodes:
            edges.append(
                {
                    "from": spoke_node["node_id"],
                    "to": spoke_node["node_id"],
                    "type": "DECISION",
                    "condition": {"type": "EXECUTE"},
                    "description": "EXECUTE",
                }
            )

        # Hub -> END (STATUS)
        edges.append(
            {
                "from": hub_node["node_id"],
                "to": "END",
                "type": "DECISION",
                "condition": {"type": "STATUS"},
                "description": "STATUS",
            }
        )

        return nodes, edges

    def validate_participants(self, participant_assignments: Dict) -> None:
        """Validate star topology participant assignments.

        Args:
            participant_assignments: Dictionary with topology role assignments

        Raises:
            ValueError: If participant configuration is invalid for star topology
        """
        # Must have exactly one hub
        if "hub" not in participant_assignments:
            raise ValueError("Star topology requires a 'hub' role")

        hub_count = participant_assignments["hub"]["count"]
        if hub_count != 1:
            raise ValueError(f"Star topology requires exactly 1 hub, got {hub_count}")

        # Must have at least one spoke
        spoke_w_count = participant_assignments.get("spoke_w_execute", {}).get("count", 0)
        spoke_wo_count = participant_assignments.get("spoke_wo_execute", {}).get("count", 0)
        total_spokes = spoke_w_count + spoke_wo_count

        if total_spokes < 1:
            raise ValueError("Star topology requires at least 1 spoke participant")

    def get_config(self) -> TopologyConfig:
        """Return star topology configuration.

        Returns:
            TopologyConfig instance for star topology
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
        # All communication flows through the hub via graph edges
        return data

    def get_template_name(self, role_name: str) -> str:
        """Get template name for star topology roles.

        Args:
            role_name: Name of the role (e.g., 'hub', 'spoke_w_execute_1')

        Returns:
            str: Template filename to use for this role
        """
        from src.message import MessageType

        base_role = role_name.split("_")[0] if "_" in role_name else role_name

        if base_role == "hub":
            return "star_topology_hub_prompt.j2"
        elif base_role in ["spoke"]:
            # Get role config to check for EXECUTE permissions
            role_config = self._get_role_config_for_name(role_name)

            # Check if role has EXECUTE permissions
            has_execute = role_config and hasattr(role_config, "message_permissions") and MessageType.EXECUTE in role_config.message_permissions.send

            execute_suffix = "w_execute" if has_execute else "wo_execute"
            return f"star_topology_spoke_{execute_suffix}_prompt.j2"
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

        try:
            # Load topology config for validation if not already loaded
            if not self.topology_config:
                # Try to load config temporarily for validation
                topology_name = self.get_config().topology_type.value
                config_path = f"configs/topologies/{topology_name}_topology.yaml"
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        self.topology_config = yaml.safe_load(f)

            # Map participants to topology roles
            participant_assignments = self._map_participants_to_roles(participants)

            # Validate using the mapped assignments
            self.validate_participants(participant_assignments)

        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return errors
