"""
Factory for creating topology instances.
"""

from typing import Dict, Type

from .base import BaseTopology, TopologyType
from .single import SingleTopology
from .star import StarTopology
from .blackboard import BlackboardTopology
from .mesh import MeshTopology


class TopologyFactory:
    """Factory for creating topology instances."""

    _topologies: Dict[TopologyType, Type[BaseTopology]] = {
        TopologyType.SINGLE: SingleTopology,
        TopologyType.STAR: StarTopology,
        TopologyType.BLACKBOARD: BlackboardTopology,
        TopologyType.MESH: MeshTopology,
    }

    @classmethod
    def create_topology(cls, topology_type: TopologyType) -> BaseTopology:
        """Create topology instance by type.

        Args:
            topology_type: The type of topology to create

        Returns:
            BaseTopology instance

        Raises:
            ValueError: If topology type is not supported
        """
        if topology_type not in cls._topologies:
            raise ValueError(f"Unknown topology type: {topology_type}")

        topology_class = cls._topologies[topology_type]
        return topology_class()

    @classmethod
    def register_topology(cls, topology_type: TopologyType, topology_class: Type[BaseTopology]) -> None:
        """Register a new topology type.

        Args:
            topology_type: The topology type to register
            topology_class: The topology class to register
        """
        cls._topologies[topology_type] = topology_class

    @classmethod
    def get_available_topologies(cls) -> list:
        """Get list of available topology types.

        Returns:
            List of available TopologyType values
        """
        return list(cls._topologies.keys())
