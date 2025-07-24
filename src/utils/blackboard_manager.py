"""
Blackboard manager for shared memory in blackboard topology.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class BlackboardEntry:
    """Represents a single entry in the blackboard."""

    role_id: str
    content: str
    timestamp: float
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return asdict(self)


class BlackboardManager:
    """Manages shared blackboard memory for blackboard topology."""

    def __init__(self):
        """Initialize empty blackboard."""
        self.entries: List[BlackboardEntry] = []

    def write_content(self, role_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Write content to the blackboard.

        Args:
            role_id: ID of the role writing the content
            content: Content to write to the blackboard
            metadata: Optional metadata for the entry
        """
        entry = BlackboardEntry(role_id=role_id, content=content, timestamp=time.time(), metadata=metadata or {})
        self.entries.append(entry)

    def get_history(self) -> List[Dict]:
        """Get blackboard history as list of dictionaries.

        Returns:
            List of blackboard entries as dictionaries
        """
        return [entry.to_dict() for entry in self.entries]

    def clear(self) -> None:
        """Clear all blackboard entries."""
        self.entries.clear()

    def get_entry_count(self) -> int:
        """Get number of entries in blackboard.

        Returns:
            Number of entries
        """
        return len(self.entries)

    def get_latest_entry(self) -> Optional[BlackboardEntry]:
        """Get the most recent blackboard entry.

        Returns:
            Latest entry or None if empty
        """
        return self.entries[-1] if self.entries else None
