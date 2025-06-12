import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from dataclasses import dataclass
from PIL import Image

from src.player import RoleBasedPlayer
from src.utils.registry.processors import processors
from src.utils.constants import (
    HANDLER_TYPE,
    OBSERVATION_TYPE_values,
)

logger = logging.getLogger(__name__)


@dataclass
class MessageState:
    """Dynamic container for message components updated during gameplay.

    Fields:
        observation: Optional dictionary (e.g., {'screenshot': str, 'accessibility_tree': str})
        request: Optional request string
        response: Optional response string
        plan: Optional plan string
        task: Optional task string
        actions: Optional list of action strings (currently unused)
        tagged_content: Optional dictionary of tag-content pairs (e.g., {'note': 'text'})
    """

    observation: Optional[Dict[str, Union[str, Image.Image, Dict]]] = None
    goal: Optional[str] = None
    plan: Optional[str] = None
    task: Optional[str] = None
    request: Optional[str] = None
    response: Optional[str] = None
    actions: Optional[List[str]] = None
    tagged_content: Optional[Dict[str, str]] = None

    def reset(self, preserve: Optional[List[str]] = None):
        """Reset specified fields to None, preserving others.

        Args:
            preserve: List of field names to preserve; defaults to ['observation']

        Returns:
            None: No return value
        """
        preserve = preserve or ["observation"]
        for field in self.__dataclass_fields__:
            if field not in preserve:
                setattr(self, field, None)
        return None

    def update(self, **kwargs):
        """Update state fields with new values, validating types.

        Args:
            **kwargs: Field names and values to update (e.g., request='new request')

        Returns:
            None: No return value

        Raises:
            ValueError: If an invalid field or incorrect type is provided
        """
        valid_fields = self.__dataclass_fields__
        for field, value in kwargs.items():
            if field not in valid_fields:
                raise ValueError(
                    f"Invalid field '{field}', must be one of {set(valid_fields)}"
                )
            if value is not None:
                if field == "tagged_content" and not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in value.items()
                ):
                    raise ValueError("Tagged content must be Dict[str, str]")
                elif field == "actions" and not all(isinstance(a, str) for a in value):
                    raise ValueError("Actions must be List[str]")
                elif field == "observation" and not isinstance(value, dict):
                    raise ValueError("Observation must be a dictionary")
                elif field in {
                    "goal",
                    "plan",
                    "task",
                    "request",
                    "response",
                } and not isinstance(value, str):
                    raise ValueError(f"{field} must be a string")
            setattr(self, field, value)
        return None

    def is_empty(self) -> bool:
        """Check if all fields are None.

        Returns:
            bool: True if all fields are None, False otherwise
        """
        return all(getattr(self, field) is None for field in self.__dataclass_fields__)

    def preview(self) -> str:
        """Generate a concise preview of MessageState fields and their values.

        Returns:
            str: Formatted string showing field names and summarized values
        """
        previews = []
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is None:
                previews.append(f"{field}: None")
            elif field == "observation" and isinstance(value, dict):
                keys = list(value.keys())
                previews.append(f"{field}: Dict with keys {keys}")
            elif field == "tagged_content" and isinstance(value, dict):
                tags = list(value.keys())
                previews.append(f"{field}: {len(tags)} tags - {tags}")
            elif field == "actions" and isinstance(value, list):
                previews.append(f"{field}: {len(value)} actions")
            elif isinstance(value, str):
                preview_text = value[:50] + "..." if len(value) > 50 else value
                preview_text = preview_text.replace("\n", " ")
                previews.append(f"{field}: {preview_text}")
            else:
                previews.append(f"{field}: {type(value).__name__}")

        return "\n".join(previews)


class PlayerContextFormatter:
    """Formats message contexts for players based on message state and player-specific requirements."""

    def __init__(self, game_config: Dict = None):
        """Initialize the player context formatter.

        Args:
            game_config: Game specific configuration, contains meta-data enatailing to environment and game.
        """
        self.format_handlers = {}
        self.game_config = game_config
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up the default format handlers."""
        self.add_handler("observation", self._format_observation)
        self.add_handler("goal", self._format_goal)
        self.add_handler("plan", self._format_plan)
        self.add_handler("task", self._format_task)
        self.add_handler("request", self._format_request)
        self.add_handler("response", self._format_response)
        self.add_handler("tagged_content", self._format_tagged_content)

    def add_handler(self, component_name: str, handler_function):
        """Register a handler function for a specific component type.

        Args:
            component_name: Name of the component (e.g., 'observation', 'request')
            handler_function: Function to handle formatting of that component
        """
        self.format_handlers[component_name] = handler_function

    def create_context_for(
        self, message_state: MessageState, player: RoleBasedPlayer
    ) -> Optional[Dict]:
        """Create a formatted context for a specific player from the current message state.

        Args:
            message_state: Current message state (instance of MessageState)
            player: Player instance to build context for (instance of RoleBasedPlayer)

        Returns:
            Dict: Dictionary containing formatted context with 'role', 'content', and optional 'image' keys
        """
        handler_type = player.handler_type
        allowed_components = (
            player.allowed_components if player.allowed_components else set()
        )
        footer_prompt = player._footer_prompt if player._footer_prompt else None
        filtered_state = self._filter_components(
            message_state, handler_type, allowed_components
        )
        if filtered_state.is_empty():
            return None
        processed_state = self._process_components(filtered_state)
        formatted_context = self.assemble(processed_state)
        if footer_prompt and "content" in formatted_context:
            formatted_context["content"] += f"\n\n{footer_prompt}"
        return formatted_context

    def _filter_components(
        self,
        message_state: MessageState,
        handler_type: HANDLER_TYPE,
        allowed_components: Set[str],
    ) -> MessageState:
        """Filter message state components based on handler type and allowed components.

        Args:
            message_state: Instance of MessageState
            handler_type: Type of handler ('standard' or 'environment')
            allowed_components: Set of permitted component types

        Returns:
            MessageState: Filtered MessageState instance

        Raises:
            ValueError: If allowed_components contains invalid components or no valid components remain
        """
        handler_rules = {
            "standard": {
                "goal",
                "plan",
                "task",
                "request",
                "response",
                "tagged_content",
            },
            "environment": {
                "observation",
                "goal",
                "plan",
                "task",
                "request",
                "response",
                "tagged_content",
            },
        }
        valid_components = set(MessageState.__dataclass_fields__)
        allowed_components = set(allowed_components)
        if invalid_components := allowed_components - valid_components:
            raise ValueError(
                f"Invalid components in allowed_components: {invalid_components}. Must be one of: {valid_components}"
            )
        permitted_components = (
            handler_rules.get(handler_type, set()) & allowed_components
        )
        filtered_components = {
            k: v
            for k, v in message_state.__dict__.items()
            if k in permitted_components and v is not None
        }
        return MessageState(**filtered_components)

    def _process_components(self, message_state: MessageState) -> MessageState:
        """Process each component using registered processors from the external 'processors' registry.

        Args:
            message_state: Instance of MessageState

        Returns:
            MessageState: New MessageState instance with processed component values

        Raises:
            ValueError: If processing a component fails
        """
        processed = {}
        for component_name, component_value in message_state.__dict__.items():
            if component_value is None:
                continue
            if component_name in processors:
                try:
                    processor = processors[component_name]
                    processed_value = processor(component_value, self.game_config)
                    if processed_value is not None:
                        processed[component_name] = processed_value
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Failed to process component '{component_name}': {str(e)}"
                    )
            else:
                processed[component_name] = component_value
        return MessageState(**processed)

    def assemble(self, message_state: MessageState) -> Dict:
        """Assemble a message context using registered handlers.

        Args:
            message_state: Instance of MessageState

        Returns:
            Dict: Dictionary with 'role', 'content', and optional 'image' keys
        """
        parts = []
        image_paths = []
        for component_name, component_value in message_state.__dict__.items():
            if component_value is None:
                continue
            if component_name in self.format_handlers:
                handler = self.format_handlers[component_name]
                formatted_component = handler(component_value)
                parts.append(formatted_component["content"])
                image_paths.extend(formatted_component.get("image", []))
            else:
                parts.append(f"{component_name.capitalize()}: {str(component_value)}")
        return {"content": "\n".join(parts), "image": image_paths or None}

    def _format_observation(self, observation: Dict) -> Dict:
        """Format an observation component.

        Args:
            observation: Dictionary containing observation data

        Returns:
            Dict: Dictionary with 'content' (formatted text) and 'image' (list of image paths)

        Raises:
            ValueError: If observation_type is invalid
        """
        formatters = {
            "screenshot": lambda obs: (
                "### Screenshot",
                [obs["screenshot"]]
                if "screenshot" in obs and isinstance(obs["screenshot"], str)
                else [],
            ),
            "a11y_tree": lambda obs: (
                f"### Accessibility Tree\n```\n{obs.get('accessibility_tree', '')}\n```",
                [],
            ),
            "screenshot_a11y_tree": lambda obs: (
                f"### Screenshot\n### Accessibility Tree\n```\n{obs.get('accessibility_tree', '')}\n```",
                [obs["screenshot"]]
                if "screenshot" in obs and isinstance(obs["screenshot"], str)
                else [],
            ),
            "som": lambda obs: (
                f"### Tagged Screenshot\n### Accessibility Tree\n```\n{obs.get('accessibility_tree', '')}\n```",
                [obs["screenshot"]]
                if "screenshot" in obs and isinstance(obs["screenshot"], str)
                else [],
            ),
        }
        observation_type = self.game_config.get("observation_type")
        if observation_type not in formatters:
            raise ValueError(
                f"Invalid observation_type: {observation_type}. Expected one of [{OBSERVATION_TYPE_values}]"
            )
        content, images = formatters[observation_type](observation)
        return {"content": f"## Observation\n{content}", "image": images}

    def _format_goal(self, goal: str) -> Dict:
        """Format a goal component.

        Args:
            goal: Goal string

        Returns:
            Dict: Dictionary with 'content' and 'image' keys
        """
        return {"content": f"## Goal\n{goal}", "image": []}

    def _format_plan(self, plan: str) -> Dict:
        """Format a plan component.

        Args:
            plan: Plan string

        Returns:
            Dict: Dictionary with 'content' and 'image' keys
        """
        return {"content": f"## Plan\n{plan}", "image": []}

    def _format_task(self, task: str) -> Dict:
        """Format a task component.

        Args:
            task: Task string

        Returns:
            Dict: Dictionary with 'content' and 'image' keys
        """
        return {"content": f"## Task\n{task}", "image": []}

    def _format_request(self, request: str) -> Dict:
        """Format a request component.

        Args:
            request: Request string

        Returns:
            Dict: Dictionary with 'content' and 'image' keys
        """
        return {"content": f"## REQUEST\n{request}", "image": []}

    def _format_response(self, response: str) -> Dict:
        """Format a response component.

        Args:
            response: Response string

        Returns:
            Dict: Dictionary with 'content' and 'image' keys
        """
        return {"content": f"## Response\n{response}", "image": []}

    def _format_tagged_content(self, tagged_content: Dict[str, str]) -> Dict:
        """Format tagged content.

        Args:
            tagged_content: Dictionary of tag-content pairs

        Returns:
            Dict: Dictionary with 'content' and 'image' keys
        """
        formatted_parts = [
            f"## {tag}\n{content}" for tag, content in tagged_content.items()
        ]
        return {"content": "\n\n".join(formatted_parts), "image": []}


class PipeStage:
    """Defines an individual processing stage within a parser pipeline."""

    def __init__(self, processor_func, output_field=None, description=""):
        self.processor_func = processor_func
        self.output_field = output_field
        self.description = description

    def execute(self, content, message_state):
        """Execute the processing step.

        Args:
            content: The input content to process
            message_state: MessageState instance to update

        Returns:
            Any: The result of the processor function execution

        Raises:
            Exception: If the processor function fails
        """
        is_bound_method = (
            hasattr(self.processor_func, "__self__")
            and self.processor_func.__self__ is not None
        )
        try:
            result = self.processor_func(content)
        except Exception as e:
            logger.error(
                f"{'Bound' if is_bound_method else 'Unbound'} processor function failed: {str(e)}"
            )
            raise
        if self.output_field and hasattr(message_state, self.output_field):
            message_state.update(**{self.output_field: result})
        return result


class PipeManager:
    """Manages and executes parser-specific processing pipelines."""

    def __init__(self):
        self.parser_pipelines = {}

    def register_pipeline(self, parser_id: str, steps: List[PipeStage]):
        """Register a processing pipeline for a parser.

        Args:
            parser_id: Identifier for the parser
            steps: List of PipeStage instances
        """
        self.parser_pipelines[parser_id] = steps

    def get_pipeline(self, parser_id: str) -> List[PipeStage]:
        """Get processing pipeline for a parser.

        Args:
            parser_id: Identifier for the parser

        Returns:
            List[PipeStage]: List of pipeline stages
        """
        return self.parser_pipelines.get(parser_id, [])

    def execute_pipeline(
        self, parser_id: str, content: Any, message_state
    ) -> Tuple[bool, Any]:
        """Execute the entire processing pipeline for a parser.

        Args:
            parser_id: Identifier for the parser
            content: Input content to process
            message_state: MessageState instance to update

        Returns:
            Tuple[bool, Any]: Success flag and result of pipeline execution
        """
        if parser_id not in self.parser_pipelines:
            return False, content
        current_content = content
        result = None
        for step in self.parser_pipelines[parser_id]:
            result = step.execute(current_content, message_state)
            current_content = result
        return True, result
