"""
Factory for creating topology instances.
"""

from typing import Dict, Type, Union

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
    def create_topology(cls, topology_type: Union[TopologyType, str]) -> BaseTopology:
        """Create topology instance by type.

        Args:
            topology_type: The type of topology to create (enum or string)

        Returns:
            BaseTopology instance

        Raises:
            ValueError: If topology type is not supported
        """
        # Convert string to enum if needed
        if isinstance(topology_type, str):
            topology_type = cls._string_to_enum(topology_type)

        if topology_type not in cls._topologies:
            raise ValueError(f"Unknown topology type: {topology_type}")

        topology_class = cls._topologies[topology_type]
        return topology_class()

    @classmethod
    def _string_to_enum(cls, topology_string: str) -> TopologyType:
        """Convert string topology type to enum.

        Args:
            topology_string: String representation of topology type

        Returns:
            TopologyType enum value

        Raises:
            ValueError: If string doesn't match any topology type
        """
        string_to_enum = {
            "single": TopologyType.SINGLE,
            "star": TopologyType.STAR,
            "blackboard": TopologyType.BLACKBOARD,
            "mesh": TopologyType.MESH,
        }

        if topology_string not in string_to_enum:
            available = list(string_to_enum.keys())
            raise ValueError(f"Unknown topology string: {topology_string}. Available: {available}")

        return string_to_enum[topology_string]

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
