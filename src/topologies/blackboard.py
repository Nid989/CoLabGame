"""
Blackboard topology implementation with shared memory and round-robin turn-taking.
"""

import logging
import os
import yaml
from typing import Dict, Any, List, Tuple

from .base import BaseTopology, TopologyConfig, TopologyType
from src.message import MessagePermissions, MessageType

logger = logging.getLogger(__name__)


class BlackboardTopology(BaseTopology):
    """Blackboard topology implementation with dynamic participant configuration."""

    def __init__(self):
        # Initialize with minimal config, will be populated by load_game_instance_config
        self.config = TopologyConfig(
            topology_type=TopologyType.BLACKBOARD,
            transition_strategy="round_robin",  # Round-robin after blackboard updates
            message_permissions={},  # Will be populated dynamically
        )
        self.topology_config = None

    def generate_graph(self, participants: Dict) -> Dict:
        """Generate blackboard topology graph using dynamic configuration and algorithmic generation."""
        if not self.topology_config:
            raise ValueError("Topology configuration not loaded. Call load_game_instance_config first.")

        # Use participants directly, assuming they are already in the correct format
        participant_assignments = participants

        # Validate the mapped participants
        self.validate_participants(participant_assignments)

        # Create node assignments with lookup index for efficient anchor finding
        node_assignments, role_domain_lookup = self._create_node_assignments_with_lookup(participant_assignments)

        # Generate graph structure algorithmically
        nodes, edges, node_sequence = self._generate_blackboard_structure(node_assignments)

        # Use centralized, config-driven anchor logic
        anchor_node = self.generate_anchor_node(node_assignments, role_domain_lookup)

        return {
            "nodes": nodes,
            "edges": edges,
            "anchor_node": anchor_node,
            "node_sequence": node_sequence,  # Needed for round-robin processing
            "node_assignments": node_assignments,  # For role creation in master.py
            "domain_definitions": self.topology_config.get("domain_definitions", {}),  # For template manager
        }

    def _create_node_assignments(self, participant_assignments: Dict) -> Dict:
        """Create node assignments with role indices and domains."""
        node_assignments, _ = self._create_node_assignments_with_lookup(participant_assignments)
        return node_assignments

    def _create_node_assignments_with_lookup(self, participant_assignments: Dict) -> Tuple[Dict, Dict]:
        """Create node assignments and build lookup index for efficient anchor finding.

        Args:
            participant_assignments: Dictionary with participant configuration

        Returns:
            Tuple[Dict, Dict]: (node_assignments, role_domain_lookup)
        """
        node_assignments = {}
        # Build (role, domain) -> node_id lookup for O(1) anchor finding
        role_domain_lookup = {}
        role_index = 0

        for role_name, assignment in participant_assignments.items():
            count = assignment["count"]
            domains = assignment.get("domains", [])

            role_nodes = []
            for i in range(count):
                node_id = f"{role_name}_{i + 1}" if count > 1 else role_name
                domain = domains[i] if i < len(domains) else (domains[0] if domains else f"general_{role_name}")

                # Add to lookup index
                role_domain_lookup[(role_name, domain)] = node_id

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

        return node_assignments, role_domain_lookup

    def _generate_blackboard_structure(self, node_assignments: Dict) -> Tuple[List, List, List]:
        """Generate blackboard topology structure algorithmically."""
        nodes = [{"id": "START", "type": "START"}]
        edges = []

        # Get all participant nodes
        participant_w_execute_nodes = node_assignments.get("participant_w_execute", [])
        participant_wo_execute_nodes = node_assignments.get("participant_wo_execute", [])

        all_participant_nodes = participant_w_execute_nodes + participant_wo_execute_nodes

        # Create node sequence for round-robin (order doesn't matter, just needs to be consistent)
        node_sequence = [node["node_id"] for node in all_participant_nodes]

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
        for node_id in node_sequence:
            edges.append(
                {
                    "from": "START",
                    "to": node_id,
                    "type": "STANDARD",
                    "description": "",
                }
            )

        # BLACKBOARD Algorithm: Round-robin connectivity with self-loops and status exits
        for i, node_id in enumerate(node_sequence):
            next_node_id = node_sequence[(i + 1) % len(node_sequence)]

            # WRITE_BOARD transitions to next participant (round-robin)
            edges.append(
                {
                    "from": node_id,
                    "to": next_node_id,
                    "type": "DECISION",
                    "condition": {"type": "WRITE_BOARD"},
                    "description": "WRITE_BOARD",
                }
            )

            # STATUS transition to END
            edges.append(
                {
                    "from": node_id,
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

        return nodes, edges, node_sequence

    def validate_participants(self, participant_assignments: Dict) -> None:
        """Validate blackboard topology participant assignments.

        Args:
            participant_assignments: Dictionary with topology role assignments

        Raises:
            ValueError: If participant configuration is invalid for blackboard topology
        """
        if len(participant_assignments) < 1:
            raise ValueError("Blackboard topology requires at least 1 participant role")

        # Count total participants across all roles
        total_participants = sum(assignment["count"] for assignment in participant_assignments.values())
        if total_participants < 2:
            raise ValueError("Blackboard topology requires at least 2 participants total for meaningful collaboration")

    def get_config(self) -> TopologyConfig:
        """Return blackboard topology configuration.

        Returns:
            TopologyConfig instance for blackboard topology
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
        """Process blackboard topology message transitions.

        For WRITE_BOARD, sets the 'to' field to the next node in round-robin and logs the transition.
        For other message types, returns data unchanged.

        Args:
            data: Parsed JSON response data
            message_type: Type of message being processed
            player: Current player instance
            game_context: Dictionary containing game state context

        Returns:
            Dict: Data with topology-specific modifications
        """
        if message_type.name == "WRITE_BOARD":
            next_node_function = game_context.get("next_node_function")
            current_node = game_context.get("current_node")

            if next_node_function:
                next_node = next_node_function()
                if next_node:
                    data["to"] = next_node
                    logger.info(f"Blackboard: {player.name} at node {current_node} wrote to board, transitioning to {next_node}")
        return data

    def initialize_game_components(self, game_instance: Dict, game_config: Dict) -> Dict:
        """Initialize blackboard-specific components.

        Sets up the blackboard manager, node sequence for round-robin, and writes
        the initial goal to the blackboard.

        Args:
            game_instance: Dictionary containing game instance configuration
            game_config: Dictionary containing game configuration

        Returns:
            Dict: Dictionary of component names to component instances
        """
        from src.utils.blackboard_manager import BlackboardManager

        # Create blackboard manager
        blackboard_manager = BlackboardManager()

        # Get node_sequence from graph metadata
        graph_config = game_instance.get("graph", {})
        node_sequence = graph_config.get("node_sequence", [])

        # Write the goal to the blackboard
        goal = game_instance.get("task_config", {}).get("instruction")
        if goal:
            blackboard_manager.write_content(role_id="Goal", content=goal)

        return {"blackboard_manager": blackboard_manager, "node_sequence": node_sequence}

    def get_template_name(self, role_name: str) -> str:
        """Get template name for blackboard topology roles.

        Args:
            role_name: Name of the role (e.g., 'participant_w_execute_1')

        Returns:
            str: Template filename to use for this role
        """
        # Extract base role name correctly (e.g., 'participant_w_execute_1' -> 'participant_w_execute')
        base_role = "_".join(role_name.split("_")[:-1]) if role_name.split("_")[-1].isdigit() else role_name

        if base_role.startswith("participant"):
            # Determine template based on role name pattern, not permissions
            if "_w_execute" in base_role:
                return "blackboard_topology_participant_w_execute_prompt.j2"
            elif "_wo_execute" in base_role:
                return "blackboard_topology_participant_wo_execute_prompt.j2"
            else:
                # Fallback for legacy role names
                return "blackboard_topology_participant_w_execute_prompt.j2"
        else:
            # Fallback to default implementation
            return super().get_template_name(role_name)

    def validate_experiment_config(self, experiment_config: Dict) -> List[str]:
        """Validate experiment configuration for blackboard topology.

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
                config_path = f"resources/topologies/{topology_name}_topology.yaml"
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        self.topology_config = yaml.safe_load(f)

            # Use participants directly for validation
            self.validate_participants(participants)

        except ValueError as e:
            errors.append(str(e))
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return errors
