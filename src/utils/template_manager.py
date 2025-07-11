"""
Template management system for dynamic prompt generation based on role permissions.
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import jinja2

from src.message import MessagePermissions, MessageType, RoleConfig

logger = logging.getLogger(__name__)


class PromptTemplateManager:
    """Manages dynamic prompt generation based on role configurations."""

    def __init__(self, template_dir: Optional[str] = None):
        """Initialize the template manager.

        Args:
            template_dir: Directory containing template files. Defaults to in/prompts/
        """
        if template_dir is None:
            template_dir = Path(__file__).parent.parent.parent / "in" / "prompts"

        self.template_dir = Path(template_dir)
        self.env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(self.template_dir)), trim_blocks=True, lstrip_blocks=True)

        # Add custom filters
        self.env.filters["join_with_or"] = self._join_with_or
        self.env.filters["message_type_schema"] = self._message_type_schema
        self.env.filters["select_message_type"] = self._select_message_type

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

    def generate_prompt(self, role_config: RoleConfig, observation_type: Optional[str] = None) -> str:
        """Generate a dynamic prompt based on role configuration.

        Args:
            role_config: Configuration for the role
            observation_type: The type of observation for environment-specific prompts.

        Returns:
            Generated prompt string
        """
        # Determine template file based on role and handler type
        template_name = self._get_template_name(role_config)

        try:
            template = self.env.get_template(template_name)
        except jinja2.TemplateNotFound:
            logger.warning(f"Template {template_name} not found, falling back to base template")
            template = self._get_base_template(role_config)

        # Prepare template context
        context = self._prepare_template_context(role_config, observation_type)

        # Render the template
        return template.render(**context)

    def _get_template_name(self, role_config: RoleConfig) -> str:
        """Determine the appropriate template file name."""
        role_name = role_config.name
        handler_type = role_config.handler_type

        # Extract base role type (e.g., 'executor' from 'executor_1')
        base_role = role_name.split("_")[0] if "_" in role_name else role_name

        # Try role-specific template first
        if base_role == "advisor":
            return "advisor_prompt.j2"
        elif base_role == "executor":
            if handler_type == "environment":
                return "executor_prompt.j2"
            else:
                return "executor_standard_prompt.j2"
        else:
            # Generic role template
            return "generic_role_prompt.j2"

    def _get_base_template(self, role_config: RoleConfig) -> jinja2.Template:
        """Create a basic template if specific template not found."""
        base_template = """
**{{ role_name|title }} Prompt**

You are the **{{ role_name|title }}**, operating in the system. Your role is to {{ role_description }}.

---

### Message Format (JSON Schema)

You must respond using structured JSON messages. Each reply must contain **exactly one JSON object** enclosed in a markdown code block using the `json` language identifier. The schema is:

```json
{
  "type": "message type",     {{ send_types|message_type_schema }}
  "from": "{{ role_name }}",  // Always set to "{{ role_name }}"
  {% if has_addressable_types -%}
  "to": "target_role",        // Required for {{ addressable_types|join(', ') }} messages
  {% endif -%}
  "content": "string"         // Message content
}
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

        return jinja2.Template(base_template)

    def _prepare_template_context(self, role_config: RoleConfig, observation_type: Optional[str] = None) -> Dict:
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

        return {
            "role_name": role_config.name,
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
        }

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
