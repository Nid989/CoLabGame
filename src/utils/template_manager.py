"""
Template management system for dynamic prompt generation based on role permissions.
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
import jinja2

from src.message import MessagePermissions, MessageType, RoleConfig
from src.topologies.base import TopologyType

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """Manages dynamic prompt generation based on role configurations."""

    def __init__(self, template_dir: Optional[str] = None):
        """Initialize the template manager.

        Args:
            template_dir: Directory containing template files. Defaults to in/prompts/
        """
        if template_dir is None:
            template_dir = Path(__file__).parent.parent.parent / "resources" / "prompts"

        self.template_dir = Path(template_dir)
        self.env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(self.template_dir)), trim_blocks=True, lstrip_blocks=True)

        # Add custom filters
        self.env.filters["join_with_or"] = self._join_with_or
        self.env.filters["message_type_schema"] = self._message_type_schema
        self.env.filters["select_message_type"] = self._select_message_type
        self.env.filters["generate_json_schema"] = self._generate_json_schema
        self.env.filters["tojson"] = lambda obj, **kwargs: json.dumps(obj, **kwargs)

    def _join_with_or(self, items: List[str]) -> str:
        """Join list items with 'or' for the last item."""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} or {items[1]}"
        return f"{', '.join(items[:-1])}, or {items[-1]}"

    def _message_type_schema(self, message_types: List[str]) -> str:
        """Generate JSON schema comment for message types."""
        if len(message_types) == 1:
            return f'// Must be "{message_types[0]}"'
        quoted_types = [f'"{mt}"' for mt in message_types]
        return f"// Must be {self._join_with_or(quoted_types)}"

    def _select_message_type(self, message_types: List[str], preferred_type: str) -> str:
        """Select a message type from the list, preferring the specified type."""
        if preferred_type in message_types:
            return preferred_type
        return message_types[0] if message_types else ""

    def _generate_json_schema(
        self, permissions: MessagePermissions, role_name: str, participants: Optional[Dict] = None, graph_config: Optional[Dict] = None
    ) -> str:
        """Generate a standard JSON schema for message format."""
        send_types = permissions.get_send_types_str()
        requires_to = [mt for mt in permissions.send if mt in MessageType.requires_to()]

        # Build available target roles dynamically from graph configuration
        available_targets = []

        if graph_config and "nodes" in graph_config:
            # Extract all player node IDs from the graph
            for node in graph_config["nodes"]:
                if node.get("type") == "PLAYER":
                    node_id = node.get("id")
                    if node_id and node_id != role_name:  # Exclude self
                        available_targets.append(node_id)
        elif participants:
            # Fallback to participant-based logic for backward compatibility
            if "advisor" in participants:
                available_targets.append("advisor")
            if "executor" in participants:
                executor_count = participants["executor"].get("count", 1)
                if executor_count == 1:
                    available_targets.append("executor")
                else:
                    for i in range(1, executor_count + 1):
                        available_targets.append(f"executor_{i}")
            # Remove the current agent's ID from available targets
            if role_name in available_targets:
                available_targets.remove(role_name)
        else:
            # Default targets as last resort
            available_targets = ["advisor", "executor"]

        # Build properties in logical order: type, from, to, content
        properties = {
            "type": {"type": "string", "enum": send_types, "description": f"Message type, must be one of: {', '.join(send_types)}"},
            "from": {"type": "string", "const": role_name, "description": f"Sender role identifier, must be '{role_name}'"},
        }

        # Add 'to' field for message types that require it (inserted between 'from' and 'content')
        if requires_to:
            properties["to"] = {
                "type": "string",
                "enum": available_targets,
                "description": f"Target role identifier, required for {', '.join([mt.name for mt in requires_to])} messages",
            }

        # Add content field last to maintain logical order
        properties["content"] = {"type": "string", "minLength": 1, "description": "Message content"}

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": properties,
            "required": ["type", "from", "content"],
            "additionalProperties": False,
        }

        # Make 'to' conditionally required
        if requires_to:
            schema["anyOf"] = [
                {"properties": {"type": {"enum": [mt.name for mt in requires_to]}}, "required": ["type", "from", "to", "content"]},
                {
                    "properties": {"type": {"enum": [mt.name for mt in permissions.send if mt not in requires_to]}},
                    "required": ["type", "from", "content"],
                },
            ]

        return schema

    def generate_prompt(
        self,
        role_config: RoleConfig,
        observation_type: Optional[str] = None,
        participants: Optional[Dict] = None,
        node_id: Optional[str] = None,
        goal: Optional[str] = None,
        topology_type: Optional[TopologyType] = None,
        graph_config: Optional[Dict] = None,
    ) -> str:
        """Generate a dynamic prompt based on role configuration.

        Args:
            role_config: Configuration for the role
            observation_type: The type of observation for environment-specific prompts.
            participants: Multi-agent participant configuration for dynamic context
            node_id: The specific node ID (e.g., 'executor_1', 'executor_2') for context
            goal: Optional goal string to be included in the prompt
            topology_type: The topology type enum (e.g., TopologyType.BLACKBOARD, TopologyType.STAR)

        Returns:
            Generated prompt string
        """
        # Determine template file based on role and topology type
        template_name = self._get_template_name(role_config, topology_type)

        try:
            template = self.env.get_template(template_name)
        except jinja2.TemplateNotFound:
            logger.warning(f"Template {template_name} not found, falling back to base template")
            template = self._get_base_template(role_config)
            # For base template, we need to add the JSON schema to the context
            context = self._prepare_template_context(role_config, observation_type, participants, node_id, goal, graph_config=graph_config)
            return template.render(**context)

        # Prepare template context
        # For blackboard topology executor, do not include goal in context
        if template_name == "blackboard_topology_executor_prompt.j2":
            context = self._prepare_template_context(role_config, observation_type, participants, node_id, goal=None, graph_config=graph_config)
        else:
            context = self._prepare_template_context(role_config, observation_type, participants, node_id, goal, graph_config=graph_config)

        # Render the template
        return template.render(**context)

    def _get_template_name(self, role_config: RoleConfig, topology_type: Optional[TopologyType] = None) -> str:
        """Determine the appropriate template file name based on role and topology type."""
        role_name = role_config.name

        # Delegate to topology class for template selection
        if topology_type:
            from src.topologies.factory import TopologyFactory

            topology = TopologyFactory.create_topology(topology_type)
            return topology.get_template_name(role_name)

        # Fallback to default naming if no topology specified
        base_role = role_name.split("_")[0] if "_" in role_name else role_name
        return f"default_{base_role}_prompt.j2"

    def _get_base_template(self, role_config: RoleConfig) -> jinja2.Template:
        """Create a basic template if specific template not found."""
        base_template = """
**{{ role_name|title }} Prompt**

You are the **{{ role_name|title }}**, operating in the system. Your role is to {{ role_description }}.

---

### Message Format (JSON Schema)

You must respond using structured JSON messages. Each reply must contain **exactly one JSON object** enclosed in a markdown code block using the `json` language identifier. The schema is:

```json
{{ json_schema | tojson(indent=2) }}
```

---

{% for msg_type in send_types %}
### Message Type: `{{ msg_type }}`

{{ message_descriptions[msg_type] }}

**Rules for `{{ msg_type }}`:**
- **`from`** must be `"{{ role_name }}"`.
{% if msg_type in requires_to_types -%}
- **`to`** must be present and set to a valid target role.
{% else -%}
- **Do not include** the `to` field.
{% endif -%}
- **`content`** must be a non-empty string.

---

{% endfor %}

### General Guidelines

- Use **only** the allowed message types: {{ send_types|join_with_or }}.
- All messages must appear inside a `json` code block and follow the defined schema precisely.
- You may add explanatory text outside the JSON block, but not inside.

---

Proceed with your assigned responsibilities.
        """.strip()

        return self.env.from_string(base_template)

    def _prepare_template_context(
        self,
        role_config: RoleConfig,
        observation_type: Optional[str] = None,
        participants: Optional[Dict] = None,
        node_id: Optional[str] = None,
        goal: Optional[str] = None,
        graph_config: Optional[Dict] = None,
    ) -> Dict:
        """Prepare context variables for template rendering."""
        permissions = role_config.message_permissions
        send_types = permissions.get_send_types_str()
        receive_types = permissions.get_receive_types_str()

        # Determine which message types require 'to' field
        requires_to_types = [mt.name for mt in permissions.send if mt in MessageType.requires_to()]

        # Get addressable types for documentation
        addressable_types = [mt for mt in send_types if mt in requires_to_types]

        # Role-specific descriptions
        role_descriptions = {
            "advisor": "coordinate and manage tasks by communicating with executors",
            "executor": "perform assigned tasks and communicate with the advisor",
        }

        # Extract base role for description lookup
        base_role = role_config.name.split("_")[0] if "_" in role_config.name else role_config.name

        # Message type descriptions
        message_descriptions = {
            "EXECUTE": "Use this type to perform actions in the environment.",
            "REQUEST": "Use this type to communicate with other roles when you need clarification, want to report status, or need to provide updates.",
            "RESPONSE": "Use this type to assign tasks, respond to requests, or provide clarification and feedback.",
            "STATUS": "Use this type to indicate completion status of the overall goal.",
            "TASK": "Use this type to define or describe tasks.",
        }

        context = {
            "role_name": node_id or role_config.name,  # Use node_id for specific instance identification
            "role_description": role_descriptions.get(base_role, "perform your assigned role"),
            "handler_type": role_config.handler_type,
            "send_types": send_types,
            "receive_types": receive_types,
            "requires_to_types": requires_to_types,
            "addressable_types": addressable_types,
            "has_addressable_types": len(addressable_types) > 0,
            "allowed_components": role_config.allowed_components,
            "message_descriptions": message_descriptions,
            "observation_type": observation_type,
            "json_schema": self._generate_json_schema(permissions, node_id or role_config.name, participants, graph_config),
            "goal": goal,
        }

        # Add dynamic multi-agent context if participants provided
        if participants:
            # Use dynamic topology-aware context building
            self._add_dynamic_topology_context(context, base_role, node_id, participants, graph_config)

        return context

    def _add_dynamic_topology_context(
        self, context: Dict, base_role: str, node_id: str, participants: Dict, graph_config: Optional[Dict] = None
    ) -> None:
        """Add dynamic topology-aware context variables."""
        # Get the topology-specific role information
        from src.player import RoleBasedMeta

        # Extract actual base role using the same logic as RoleBasedPlayer
        actual_base_role = RoleBasedMeta._extract_base_role(node_id or context["role_name"])

        # Handle different topology types dynamically
        if actual_base_role in ["hub", "advisor"]:
            # Star topology hub/advisor context
            self._add_hub_context(context, participants, graph_config)
        elif actual_base_role in ["spoke_w_execute", "spoke_wo_execute", "executor"]:
            # Star topology spoke/executor context
            self._add_spoke_context(context, actual_base_role, node_id, participants, graph_config)
        elif actual_base_role in ["participant_w_execute", "participant_wo_execute"]:
            # Blackboard/Mesh topology participant context
            self._add_participant_context(context, actual_base_role, node_id, participants, graph_config)

    def _add_hub_context(self, context: Dict, participants: Dict, graph_config: Optional[Dict] = None) -> None:
        """Add context for hub/advisor roles in star topology."""
        executor_domains = []

        # Try to get domain information from graph configuration first
        if graph_config and "node_assignments" in graph_config:
            node_assignments = graph_config["node_assignments"]
            # Collect node_id and domain pairs from all spoke types
            for role_type in ["spoke_w_execute", "spoke_wo_execute"]:
                if role_type in node_assignments:
                    for node_info in node_assignments[role_type]:
                        node_id = node_info.get("node_id")
                        domain = node_info.get("domain")
                        if node_id and domain and domain != f"general_{role_type}":
                            executor_domains.append({"node_id": node_id, "domain": domain})

        # Fallback to participant information if graph config doesn't have domains
        if not executor_domains:
            # Check for new topology participants first
            spoke_domains = []
            for role_type in ["spoke_w_execute", "spoke_wo_execute"]:
                if role_type in participants:
                    spoke_domains.extend(participants[role_type].get("domains", []))

            # Fallback to legacy executor participants
            if not spoke_domains and "executor" in participants:
                spoke_domains = participants["executor"].get("domains", [])

            # Convert domain list to node_id/domain pairs with generic names
            for i, domain in enumerate(spoke_domains):
                executor_domains.append(
                    {
                        "node_id": f"spoke_{i + 1}",  # Generic fallback names
                        "domain": domain,
                    }
                )

        context.update(
            {
                "include_executor_domains": len(executor_domains) > 0,
                "executor_domains": executor_domains,
            }
        )

    def _add_spoke_context(self, context: Dict, base_role: str, node_id: str, participants: Dict, graph_config: Optional[Dict] = None) -> None:
        """Add context for spoke/executor roles in star topology."""
        own_domain = None
        total_participants = 0

        # Try to get domain from graph configuration first
        if graph_config and "node_assignments" in graph_config and node_id:
            node_assignments = graph_config["node_assignments"]
            # Search for this specific node's domain
            for role_type, nodes in node_assignments.items():
                for node_info in nodes:
                    if node_info.get("node_id") == node_id:
                        domain = node_info.get("domain")
                        if domain and domain != f"general_{role_type}":
                            own_domain = domain
                        break
                if own_domain:
                    break

            # Count total participants
            for role_type in ["spoke_w_execute", "spoke_wo_execute"]:
                if role_type in node_assignments:
                    total_participants += len(node_assignments[role_type])

        # Fallback to participant information
        if not own_domain or total_participants == 0:
            for participant_type, participant_info in participants.items():
                if participant_type in ["executor", "spoke_w_execute", "spoke_wo_execute"]:
                    domains = participant_info.get("domains", [])
                    count = participant_info.get("count", 0)
                    total_participants += count

                    # Try to match node_id to domain
                    if node_id and "_" in node_id and not own_domain:
                        try:
                            # Extract number from node_id like "spoke_w_execute_1" -> 1
                            parts = node_id.split("_")
                            if parts[-1].isdigit():
                                idx = int(parts[-1]) - 1  # Convert to 0-based index
                                if 0 <= idx < len(domains):
                                    own_domain = domains[idx]
                        except (ValueError, IndexError):
                            pass

                    # If no domain found yet, try first domain as fallback
                    if not own_domain and domains:
                        own_domain = domains[0]

        context.update(
            {
                "include_own_domain": own_domain is not None,
                "own_domain": own_domain,
                "include_other_executors": total_participants > 1,
                "total_executors": total_participants,
            }
        )

    def _add_participant_context(self, context: Dict, base_role: str, node_id: str, participants: Dict, graph_config: Optional[Dict] = None) -> None:
        """Add context for participant roles in blackboard/mesh topologies."""
        own_domain = None
        total_participants = 0
        peer_domains = []

        # Collect participant information from all participant types
        for participant_type, participant_info in participants.items():
            if participant_type in ["executor", "participant_w_execute", "participant_wo_execute"]:
                domains = participant_info.get("domains", [])
                count = participant_info.get("count", 0)
                total_participants += count
                peer_domains.extend(domains)

                # Try to match node_id to domain
                if node_id and "_" in node_id:
                    try:
                        # Extract number from node_id like "participant_w_execute_2" -> 2
                        parts = node_id.split("_")
                        if parts[-1].isdigit():
                            idx = int(parts[-1]) - 1  # Convert to 0-based index
                            if 0 <= idx < len(domains):
                                own_domain = domains[idx]
                    except (ValueError, IndexError):
                        pass

        context.update(
            {
                "include_own_domain": own_domain is not None,
                "own_domain": own_domain,
                "include_other_executors": total_participants > 1,
                "total_executors": total_participants,
                "include_peer_domains": len(peer_domains) > 1,
                "peer_domains": peer_domains,
            }
        )

    def create_message_schema(self, permissions: MessagePermissions) -> Dict:
        """Create a JSON schema for the message format based on permissions.

        Args:
            permissions: MessagePermissions instance

        Returns:
            Dictionary representing the JSON schema
        """
        send_types = permissions.get_send_types_str()

        schema = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": send_types, "description": f"Message type, must be one of: {', '.join(send_types)}"},
                "from": {"type": "string", "description": "Sender role identifier"},
                "content": {"type": "string", "description": "Message content"},
            },
            "required": ["type", "from", "content"],
        }

        # Add 'to' field for message types that require it
        requires_to = [mt for mt in permissions.send if mt in MessageType.requires_to()]

        if requires_to:
            schema["properties"]["to"] = {"type": "string", "description": "Target role identifier (required for REQUEST and RESPONSE messages)"}

            # Make 'to' conditionally required
            schema["anyOf"] = [
                {"properties": {"type": {"enum": [mt.name for mt in requires_to]}}, "required": ["type", "from", "to", "content"]},
                {
                    "properties": {"type": {"enum": [mt.name for mt in permissions.send if mt not in requires_to]}},
                    "required": ["type", "from", "content"],
                },
            ]

        return schema
