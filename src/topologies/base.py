"""
Base classes for topology management system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict


class TopologyType(Enum):
    """Enum for different topology types."""

    SINGLE = "single"
    STAR = "star"
    BLACKBOARD = "blackboard"


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
