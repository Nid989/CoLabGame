"""
Base classes for topology management system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
import logging
import yaml
import os

from src.utils.domain_manager import DomainManager, DomainResolutionError

logger = logging.getLogger(__name__)


class TopologyType(Enum):
    """Enum for different topology types."""

    SINGLE = "single"
    STAR = "star"
    BLACKBOARD = "blackboard"
    MESH = "mesh"


class TopologyConfig:
    """Configuration for a specific topology."""

    def __init__(self, topology_type: TopologyType, **kwargs):
        self.topology_type = topology_type
        self.anchor_selection = kwargs.get("anchor_selection", "fixed")  # fixed/random
        self.transition_strategy = kwargs.get("transition_strategy", "conditional")  # conditional/round_robin
        self.message_permissions = kwargs.get("message_permissions", {})
        self.role_configs = kwargs.get("role_configs", {})


class BaseTopology(ABC):
    """Abstract base class for all topologies."""

    @abstractmethod
    def generate_graph(self, participants: Dict) -> Dict:
        """Generate graph configuration for the topology.

        Args:
            participants: Dictionary with participant configuration

        Returns:
            Dict containing graph configuration with nodes, edges, and anchor_node
        """
        pass

    @abstractmethod
    def get_config(self) -> TopologyConfig:
        """Return topology-specific configuration.

        Returns:
            TopologyConfig instance for this topology
        """
        pass

    @abstractmethod
    def validate_participants(self, participants: Dict) -> None:
        """Validate participant configuration for this topology.

        Args:
            participants: Dictionary with participant configuration

        Raises:
            ValueError: If participant configuration is invalid for this topology
        """
        pass

    def load_game_instance_config(self, game_instance: Dict) -> None:
        """Load topology configuration for a specific game instance.

        This method loads the topology-specific configuration from YAML files
        and sets up the participant assignments based on the game instance data.

        Args:
            game_instance: Dictionary containing game instance configuration
                          with fields like game_id, category, task_type, participants
        """
        # Extract and store category and task_type from game instance
        self.category = game_instance.get("category", None)
        self.task_type = game_instance.get("task_type", None)

        # Get topology type name for config file
        topology_name = self.get_config().topology_type.value
        config_path = f"resources/topologies/{topology_name}_topology.yaml"

        # Load topology configuration from YAML file
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                topology_config = yaml.safe_load(f)
                self.topology_config = topology_config

                # Validate domain definitions are present and properly configured
                self._validate_domain_definitions()

                # Validate anchor configuration
                self._validate_anchor_configuration()

                logger.info(f"Loaded topology config from {config_path} for category: {self.category}, task_type: {self.task_type}")
        else:
            logger.warning(f"Topology config file not found: {config_path}")
            self.topology_config = None

    def _validate_domain_definitions(self) -> None:
        """
        Validate that domain definitions are properly configured.

        Ensures:
        1. domain_definitions section exists
        2. All domains have descriptions
        3. All referenced domains in participant assignments are defined

        Raises:
            DomainResolutionError: If domain configuration is invalid
        """
        if not self.topology_config:
            raise DomainResolutionError("Topology configuration is required but not loaded")

        # Check if domain_definitions section exists
        if "domain_definitions" not in self.topology_config:
            raise DomainResolutionError(
                "Domain definitions section is missing from topology configuration. All domains must have descriptions defined."
            )

        domain_definitions = self.topology_config["domain_definitions"]
        if not domain_definitions:
            raise DomainResolutionError("Domain definitions section is empty. All domains must have descriptions.")

        # Create domain manager to validate domain definitions structure
        try:
            domain_manager = DomainManager(domain_definitions)
        except DomainResolutionError as e:
            raise DomainResolutionError(f"Invalid domain definitions: {str(e)}")

        # Validate that USED domains in participant assignments are properly defined
        used_domains = set()

        # Collect domains from default participant assignments
        if "default_participant_assignments" in self.topology_config:
            participant_assignments = self.topology_config["default_participant_assignments"]
            for role_name, role_config in participant_assignments.items():
                if "domains" in role_config:
                    used_domains.update(role_config["domains"])

        # Collect domains from category-specific participant assignments
        if "category_participant_assignments" in self.topology_config:
            category_assignments = self.topology_config["category_participant_assignments"]
            for category, category_config in category_assignments.items():
                # Handle new task_types structure
                if "task_types" in category_config:
                    task_type_assignments = category_config["task_types"]
                    for task_type, participant_assignments in task_type_assignments.items():
                        for role_name, role_config in participant_assignments.items():
                            if "domains" in role_config:
                                used_domains.update(role_config["domains"])

                # Handle direct participant assignments (backward compatibility)
                for role_name, role_config in category_config.items():
                    if role_name != "task_types" and isinstance(role_config, dict) and "domains" in role_config:
                        used_domains.update(role_config["domains"])

        # Only validate domains that are actually used in participant assignments
        # This allows topology configs to define many domains but only use a subset
        if used_domains:
            try:
                domain_manager.validate_domain_references(list(used_domains))
                logger.info(f"Validated {len(used_domains)} used domains out of {len(domain_definitions)} total defined domains")
            except DomainResolutionError as e:
                raise DomainResolutionError(f"Used domain validation failed in participant assignments: {str(e)}")

        logger.info(f"Domain definitions validation passed for {len(domain_definitions)} domains")

    def get_default_participants(self) -> Dict:
        """
        Get participant assignments from topology configuration, prioritizing task_type-specific assignments.

        Returns:
            Dictionary containing participant assignments for this topology.
            Lookup priority:
            1. category_participant_assignments[category][task_types][task_type] (most specific)
            2. category_participant_assignments[category] (category-level fallback)
            3. default_participant_assignments (global fallback)

        Raises:
            ValueError: If topology config is not loaded or no assignments are found
        """
        if not self.topology_config:
            raise ValueError("Topology configuration not loaded - call load_game_instance_config() first")

        # Check for category-specific assignments first
        if self.category and "category_participant_assignments" in self.topology_config:
            category_assignments = self.topology_config["category_participant_assignments"]
            if self.category in category_assignments:
                category_config = category_assignments[self.category]

                # Level 1: Check for task_type-specific assignments (most specific)
                if self.task_type and "task_types" in category_config:
                    task_type_assignments = category_config["task_types"]
                    if self.task_type in task_type_assignments:
                        logger.info(f"Using task_type-specific participant assignments for category: {self.category}, task_type: {self.task_type}")
                        return task_type_assignments[self.task_type]
                    elif "default" in task_type_assignments:
                        logger.info(f"Task_type '{self.task_type}' not found, using default task_type for category: {self.category}")
                        return task_type_assignments["default"]
                    else:
                        logger.info(
                            f"No task_type assignments found for '{self.task_type}' in category: {self.category}, falling back to category level"
                        )

                # Level 2: Check if category has direct participant assignments (backward compatibility)
                # This handles categories that don't use task_types structure
                category_direct_assignments = {k: v for k, v in category_config.items() if k != "task_types"}
                if category_direct_assignments:
                    logger.info(f"Using category-level participant assignments for category: {self.category}")
                    return category_direct_assignments
            else:
                logger.info(f"Category '{self.category}' not found in category_participant_assignments, falling back to default")

        # Level 3: Fallback to default participant assignments
        if "default_participant_assignments" not in self.topology_config:
            raise ValueError("No default participant assignments found in topology configuration")

        logger.info("Using default participant assignments")
        return self.topology_config["default_participant_assignments"]

    def process_message(self, data: Dict, message_type: Any, player: Any, game_context: Dict) -> Dict:
        """Process topology-specific message logic.

        This method allows topologies to modify message data based on their specific
        communication patterns and transition rules.

        Args:
            data: Parsed JSON response data
            message_type: Type of message being processed
            player: Current player instance
            game_context: Dictionary containing game state context

        Returns:
            Dict: Modified data with topology-specific changes (e.g., added 'to' field)
        """
        # Default implementation: no processing
        return data

    def initialize_game_components(self, game_instance: Dict, game_config: Dict) -> Dict:
        """Initialize topology-specific components.

        This method allows topologies to set up any special components they need
        for their operation (e.g., blackboard manager, coordination state).

        Args:
            game_instance: Dictionary containing game instance configuration
            game_config: Dictionary containing game configuration

        Returns:
            Dict: Dictionary of component names to component instances
        """
        # Default implementation: no components
        return {}

    def _get_role_config_for_name(self, role_name: str) -> Optional[Any]:
        """Get role configuration for a specific role name.

        Args:
            role_name: Name of the role (e.g., 'spoke_w_execute_1', 'participant_wo_execute_2')

        Returns:
            RoleConfig instance if found, None otherwise
        """
        if not self.topology_config:
            return None

        # Extract base role from role name (e.g., 'spoke_w_execute_1' -> 'spoke_w_execute')
        base_role = "_".join(role_name.split("_")[:-1]) if role_name.split("_")[-1].isdigit() else role_name

        # Look for role in topology configuration
        role_definitions = self.topology_config.get("role_definitions", {})
        if base_role in role_definitions:
            from src.message import MessagePermissions, MessageType, RoleConfig

            role_config = role_definitions[base_role]
            permissions = role_config.get("message_permissions", {})
            send_types = [MessageType.from_string(mt) for mt in permissions.get("send", [])]
            receive_types = [MessageType.from_string(mt) for mt in permissions.get("receive", [])]

            return RoleConfig(
                name=base_role,
                handler_type=role_config.get("handler_type", "standard"),
                message_permissions=MessagePermissions(send=send_types, receive=receive_types),
                allowed_components=role_config.get("allowed_components", []),
            )

        return None

    def get_template_name(self, role_name: str) -> str:
        """Get template name for a specific role.

        This method allows topologies to specify which template files should be used
        for different roles in their topology.

        Args:
            role_name: Name of the role (e.g., 'advisor', 'executor_1')

        Returns:
            str: Template filename to use for this role
        """
        # Default implementation: construct template name from topology type and base role
        base_role = role_name.split("_")[0] if "_" in role_name else role_name
        return f"{self.get_config().topology_type.value}_topology_{base_role}_prompt.j2"

    def get_anchor_selection_config(self) -> Dict:
        """Get anchor selection configuration from topology config.

        Returns:
            Dict: Configuration with 'mode' and 'config' keys
        """
        if not self.topology_config:
            raise ValueError("Topology configuration not loaded")

        # Check category-specific config first
        if self.category and "category_participant_assignments" in self.topology_config:
            category_assignments = self.topology_config["category_participant_assignments"]
            if self.category in category_assignments:
                category_config = category_assignments[self.category]

                # Task-type specific config (if exists)
                if self.task_type and "task_types" in category_config:
                    task_type_config = category_config["task_types"].get(self.task_type, {})
                    if "anchor_selection_mode" in task_type_config:
                        return {"mode": task_type_config["anchor_selection_mode"], "config": task_type_config.get("anchor_node_config")}

                # Category-level config
                if "anchor_selection_mode" in category_config:
                    return {"mode": category_config["anchor_selection_mode"], "config": category_config.get("anchor_node_config")}

        # Global default config
        default_mode = self.topology_config.get("anchor_selection_mode", "random")  # backward compatibility
        default_config = self.topology_config.get("default_anchor_node_config")

        return {"mode": default_mode, "config": default_config}

    def generate_anchor_node(self, node_assignments: Dict, role_domain_lookup: Dict = None) -> str:
        """Generate anchor node based strictly on configuration - no fallback priorities.

        Args:
            node_assignments: Dictionary of role assignments
            role_domain_lookup: Optional lookup dict for O(1) access

        Returns:
            str: Node ID of the selected anchor node
        """
        anchor_config = self.get_anchor_selection_config()
        mode = anchor_config["mode"]
        config = anchor_config["config"]

        if mode == "fixed":
            if not config:
                raise ValueError("anchor_selection_mode is 'fixed' but no anchor_node_config provided")

            return self._get_fixed_anchor_node(config, node_assignments, role_domain_lookup)

        elif mode == "random":
            return self._get_random_anchor_node(node_assignments)

        else:
            raise ValueError(f"Invalid anchor_selection_mode: '{mode}'. Must be 'fixed' or 'random'")

    def _get_fixed_anchor_node(self, config: Dict, node_assignments: Dict, role_domain_lookup: Dict = None) -> str:
        """Get the exactly specified anchor node with strict validation.

        Args:
            config: Anchor node configuration with role and domain
            node_assignments: Dictionary of role assignments
            role_domain_lookup: Optional lookup dict for O(1) access

        Returns:
            str: Node ID of the fixed anchor node
        """
        required_role = config.get("role")
        required_domain = config.get("domain")

        if not required_role or not required_domain:
            raise ValueError("Fixed anchor_node_config must specify both 'role' and 'domain'")

        # Use O(1) lookup if available
        if role_domain_lookup and (required_role, required_domain) in role_domain_lookup:
            anchor_node = role_domain_lookup[(required_role, required_domain)]
            logger.info(f"Using fixed anchor node: {anchor_node} (role='{required_role}', domain='{required_domain}')")
            return anchor_node

        # O(n) fallback lookup
        if required_role in node_assignments:
            for node in node_assignments[required_role]:
                if node.get("domain") == required_domain:
                    logger.info(f"Using fixed anchor node: {node['node_id']} (role='{required_role}', domain='{required_domain}')")
                    return node["node_id"]

        # Strict validation - fail if not found
        available_combinations = []
        for role, nodes in node_assignments.items():
            for node in nodes:
                available_combinations.append(f"role='{role}', domain='{node.get('domain')}'")

        raise ValueError(
            f"Could not find participant with role='{required_role}' and domain='{required_domain}' "
            f"for fixed anchor node. Available combinations: {available_combinations}"
        )

    def _get_random_anchor_node(self, node_assignments: Dict) -> str:
        """Get random anchor node for backward compatibility.

        Args:
            node_assignments: Dictionary of role assignments

        Returns:
            str: Node ID of the randomly selected anchor node
        """
        import random

        all_participant_nodes = []
        for role_nodes in node_assignments.values():
            all_participant_nodes.extend([node["node_id"] for node in role_nodes])

        if not all_participant_nodes:
            raise ValueError("No participant nodes available for random anchor selection")

        anchor_node = random.choice(all_participant_nodes)
        logger.info(f"Using random anchor node: {anchor_node}")
        return anchor_node

    def _validate_anchor_configuration(self) -> None:
        """Validate anchor configuration during config loading."""
        try:
            anchor_config = self.get_anchor_selection_config()
            mode = anchor_config["mode"]
            config = anchor_config["config"]

            if mode == "fixed" and not config:
                raise ValueError("anchor_selection_mode is 'fixed' but no anchor_node_config provided")

            if mode == "fixed":
                required_role = config.get("role")
                required_domain = config.get("domain")
                if not required_role or not required_domain:
                    raise ValueError("Fixed anchor_node_config must specify both 'role' and 'domain'")

            logger.info(f"Anchor configuration validated: mode='{mode}'")

        except Exception as e:
            logger.error(f"Anchor configuration validation failed: {e}")
            raise

    def validate_experiment_config(self, experiment_config: Dict) -> List[str]:
        """Validate experiment configuration for this topology.

        This method allows topologies to validate their specific configuration
        requirements and return a list of validation errors.

        Args:
            experiment_config: Dictionary containing experiment configuration

        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        # Default implementation: no additional validation
        return []
