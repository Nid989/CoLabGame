"""
Base classes for topology management system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List
import logging
import yaml
import os

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
        # Get topology type name for config file
        topology_name = self.get_config().topology_type.value
        config_path = f"configs/topologies/{topology_name}_topology.yaml"

        # Load topology configuration from YAML file
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                topology_config = yaml.safe_load(f)
                self.topology_config = topology_config

                # Future: Can add category/task-specific config loading here
                # Example: star_topology_code_ops.yaml for category-specific configs

                logger.info(f"Loaded topology config from {config_path}")
        else:
            logger.warning(f"Topology config file not found: {config_path}")
            self.topology_config = None

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
