"""
Domain management system for dynamic prompt generation with mandatory descriptions.

This module provides functionality to resolve domain names to their full definitions
including descriptions, ensuring all domains have proper context for template rendering.
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DomainResolutionError(Exception):
    """Raised when domain resolution fails due to missing definitions or descriptions."""

    pass


class DomainManager:
    """
    Manages domain definitions and resolves domain names to full context information.

    Enforces mandatory domain descriptions - all referenced domains MUST have
    descriptions defined in the topology configuration.
    """

    def __init__(self, domain_definitions: Dict):
        """
        Initialize domain manager with domain definitions.

        Args:
            domain_definitions: Dictionary mapping domain names to their definitions
                                Expected format: {domain_name: {"self_description": "...", "team_description": "..."}}

        Raises:
            DomainResolutionError: If domain_definitions is empty or malformed
        """
        if not domain_definitions:
            raise DomainResolutionError("Domain definitions are required but not provided")

        self.domain_definitions = domain_definitions
        self._validate_domain_definitions()

    def _validate_domain_definitions(self) -> None:
        """
        Validate that all domain definitions have required self_description and team_description fields.

        Raises:
            DomainResolutionError: If any domain lacks required descriptions
        """
        for domain_name, domain_def in self.domain_definitions.items():
            if not isinstance(domain_def, dict):
                raise DomainResolutionError(f"Domain '{domain_name}' definition must be a dictionary")

            required_fields = ["self_description", "team_description"]
            for field in required_fields:
                if field not in domain_def:
                    raise DomainResolutionError(f"Domain '{domain_name}' is missing required '{field}' field")

                if not domain_def[field] or not isinstance(domain_def[field], str):
                    raise DomainResolutionError(f"Domain '{domain_name}' has empty or invalid {field}")

    def resolve_domain(self, domain_name: str, context: str = "self") -> Dict:
        """
        Resolve domain name to full definition with context-appropriate description.

        Args:
            domain_name: Name of the domain to resolve
            context: Context for description ('self' or 'team')

        Returns:
            Dictionary containing:
                - name: The domain name
                - description: The appropriate description based on context
                - has_description: Always True (since descriptions are mandatory)

        Raises:
            DomainResolutionError: If domain is not defined or context is invalid
        """
        if domain_name not in self.domain_definitions:
            available_domains = list(self.domain_definitions.keys())
            raise DomainResolutionError(f"Domain '{domain_name}' is not defined. Available domains: {available_domains}")

        if context not in ["self", "team"]:
            raise DomainResolutionError(f"Invalid context '{context}'. Must be 'self' or 'team'")

        domain_def = self.domain_definitions[domain_name]

        # Select appropriate description based on context
        description_key = f"{context}_description"
        description = domain_def[description_key]

        return {
            "name": domain_name,
            "description": description,
            "has_description": True,  # Always True since descriptions are mandatory
        }

    def resolve_domains(self, domain_names: List[str], context: str = "self") -> List[Dict]:
        """
        Resolve multiple domain names to their full definitions.

        Args:
            domain_names: List of domain names to resolve
            context: Context for description ('self' or 'team')

        Returns:
            List of resolved domain dictionaries

        Raises:
            DomainResolutionError: If any domain is not defined or lacks description
        """
        return [self.resolve_domain(name, context) for name in domain_names]

    def get_available_domains(self) -> List[str]:
        """
        Get list of all available domain names.

        Returns:
            List of domain names that are defined and have descriptions
        """
        return list(self.domain_definitions.keys())

    def validate_domain_references(self, referenced_domains: List[str]) -> None:
        """
        Validate that all referenced domains exist and have descriptions.

        Args:
            referenced_domains: List of domain names that are referenced

        Raises:
            DomainResolutionError: If any referenced domain is not properly defined
        """
        for domain_name in referenced_domains:
            self.resolve_domain(domain_name)  # Will raise error if invalid

    def get_domain_description(self, domain_name: str, context: str = "self") -> str:
        """
        Get the description for a specific domain in the specified context.

        Args:
            domain_name: Name of the domain
            context: Context for description ('self' or 'team')

        Returns:
            The domain description

        Raises:
            DomainResolutionError: If domain is not defined
        """
        domain_info = self.resolve_domain(domain_name, context)
        return domain_info["description"]
