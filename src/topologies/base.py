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
        # Extract and store category from game instance
        self.category = game_instance.get("category", None)

        # Get topology type name for config file
        topology_name = self.get_config().topology_type.value
        config_path = f"configs/topologies/{topology_name}_topology.yaml"

        # Load topology configuration from YAML file
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                topology_config = yaml.safe_load(f)
                self.topology_config = topology_config

                # Validate domain definitions are present and properly configured
                self._validate_domain_definitions()

                logger.info(f"Loaded topology config from {config_path} for category: {self.category}")
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
            for category, participant_assignments in category_assignments.items():
                for role_name, role_config in participant_assignments.items():
                    if "domains" in role_config:
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
        Get participant assignments from topology configuration, prioritizing category-specific assignments.

        Returns:
            Dictionary containing participant assignments for this topology.
            First checks for category-specific assignments, then falls back to default assignments.

        Raises:
            ValueError: If topology config is not loaded or no assignments are found
        """
        if not self.topology_config:
            raise ValueError("Topology configuration not loaded - call load_game_instance_config() first")

        # Check for category-specific assignments first
        if self.category and "category_participant_assignments" in self.topology_config:
            category_assignments = self.topology_config["category_participant_assignments"]
            if self.category in category_assignments:
                logger.info(f"Using category-specific participant assignments for category: {self.category}")
                return category_assignments[self.category]
            else:
                logger.info(f"Category '{self.category}' not found in category_participant_assignments, falling back to default")

        # Fallback to default participant assignments
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
