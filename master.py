import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from dataclasses import dataclass
from PIL import Image

from clemcore import backends
from clemcore.clemgame import Player, GameMaster, GameBenchmark
from src.game_master import (
    NetworkDialogueGameMaster,
    EdgeCondition,
    EdgeType,
    ConditionType,
)
from src.game import ComputerGame, RoleBasedPlayer
from src.utils.registry.parsers import parsers, get_parser_metadata
from src.utils.registry.processors import processors
from src.utils.constants import (
    HANDLER_TYPE,
    DEFAULT_ENV_CONFIG,
    DEFAULT_HANDLER_TYPE,
    OBSERVATION_TYPE_values,
)

logger = logging.getLogger(__name__)


@dataclass
class MessageState:
    """Dynamic container for message components updated during gameplay.

    Fields:
        observation: Optional dictionary (e.g., {'screenshot': str, 'accessibility_tree': str})
        query: Optional query string
        response: Optional response string
        plan: Optional plan string
        task: Optional task string
        actions: Optional list of action strings (currently unused)
        tagged_content: Optional dictionary of tag-content pairs (e.g., {'note': 'text'})
    """

    observation: Optional[Dict[str, Union[str, Image.Image, Dict]]] = None
    query: Optional[str] = None
    response: Optional[str] = None
    plan: Optional[str] = None
    task: Optional[str] = None
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
            **kwargs: Field names and values to update (e.g., query='new query')

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
                elif field in {"query", "response", "plan", "task"} and not isinstance(
                    value, str
                ):
                    raise ValueError(f"{field} must be a string")
            setattr(self, field, value)
        return None

    def is_empty(self) -> bool:
        """Check if all fields are None.

        Returns:
            bool: True if all fields are None, False otherwise
        """
        return all(getattr(self, field) is None for field in self.__dataclass_fields__)


class PlayerContextFormatter:
    """Formats message contexts for players based on message state and player-specific requirements."""

    def __init__(self, game_config: Dict = None):
        """Initialize the player context formatter.

        Args:
            game_config: Optional environment configuration dictionary; defaults to DEFAULT_ENV_CONFIG if None
        """
        self.format_handlers = {}
        self.game_config = game_config or DEFAULT_ENV_CONFIG
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up the default format handlers."""
        self.add_handler("observation", self._format_observation)
        self.add_handler("query", self._format_query)
        self.add_handler("response", self._format_response)
        self.add_handler("plan", self._format_plan)
        self.add_handler("task", self._format_task)
        self.add_handler("tagged_content", self._format_tagged_content)

    def add_handler(self, component_name: str, handler_function):
        """Register a handler function for a specific component type.

        Args:
            component_name: Name of the component (e.g., 'observation', 'query')
            handler_function: Function to handle formatting of that component
        """
        self.format_handlers[component_name] = handler_function

    def create_context_for(
        self, message_state: MessageState, player: "RoleBasedPlayer"
    ) -> Dict:
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
            "standard": {"query", "response", "plan", "task", "tagged_content"},
            "environment": {
                "observation",
                "query",
                "response",
                "plan",
                "task",
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
        if not filtered_components:
            raise ValueError(
                f"No permitted components found for {handler_type} handler with allowed components {allowed_components}"
            )
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
                    processed_value = processor(component_value, self)
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

    def _format_query(self, query: str) -> Dict:
        """Format a query component.

        Args:
            query: Query string

        Returns:
            Dict: Dictionary with 'content' and 'image' keys
        """
        return {"content": f"## Query\n{query}", "image": []}

    def _format_response(self, response: str) -> Dict:
        """Format a response component.

        Args:
            response: Response string

        Returns:
            Dict: Dictionary with 'content' and 'image' keys
        """
        return {"content": f"## Response\n{response}", "image": []}

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


class ComputerGameMaster(NetworkDialogueGameMaster):
    def __init__(
        self,
        name: str,
        path: str,
        experiment: Dict,
        player_models: List[backends.Model],
    ):
        super().__init__(name, path, experiment, player_models)
        self.experiment: str = experiment["name"]
        self.game: ComputerGame = None
        self.game_instance: Dict = None
        self.terminated: bool = False
        self.message_state = MessageState()
        self.pipe_manager = PipeManager()
        self.player_context_formatter = None

    def _on_setup(self, **game_instance) -> None:
        """Method executed at the start of the default setup method.

        Key Actions:
            - Sets up environment and loads initial observation + starts gameplay recording.
            - Constructs player interaction graph/network.
            - Sets up trigger pipeline (specific) 'parse func. -> after parse steps'

        Args:
            game_instance: Keyword arguments of the game_instance

        Returns:
            None: No return value
        """
        self.game_instance = game_instance
        self._initialize_environment()
        # TODO have to standardize the method to either say game or env including src/game.py
        game_config = DEFAULT_ENV_CONFIG.copy()
        use_images = (
            True
            if game_config["observation_type"]
            in ["screenshot", "screenshot_a11y_tree", "som"]
            else False
        )
        if use_images:
            from src.game import TemporaryImageManager

            game_config["temporary_image_manager"] = TemporaryImageManager()
        self.player_context_formatter = PlayerContextFormatter(game_config=game_config)
        self._build_graph()
        self._setup_after_parse_pipelines()

    def _initialize_environment(self) -> None:
        """Initializes game environment with recording capabilities and retrieves the initial state observation.

        Returns:
            None: No return value

        Raises:
            RuntimeError: If environment or recording initialization fails
        """
        try:
            env_config = DEFAULT_ENV_CONFIG.copy()
            self.game = ComputerGame(
                **env_config, game_instance=self.game_instance["task_config"]
            )
            observation = self.game.env.reset(
                task_config=self.game_instance["task_config"]
            )
            self.message_state.update(observation=observation)
            if not self.game.env.start_recording():
                raise RuntimeError("Failed to start environment recording")
        except Exception as e:
            error_message = (
                f"Environment initialization failed: {str(e)}"
                if "recording" not in str(e).lower()
                else f"Recording initialization failed: {str(e)}"
            )
            raise RuntimeError(error_message) from e

    def _build_graph(self) -> None:
        """Builds a dialogic-player network graph from the game instance configuration.

        Returns:
            None: No return value

        Raises:
            RuntimeError: If graph building fails
            ValueError: If player model is not available for role index
            KeyError: If condition type is invalid
        """
        try:
            graph_config = self.game_instance.get("graph")
            for node in graph_config.get("nodes", []):
                node_id = node.get("id")
                node_type = node.get("type")
                if not node_id or not node_type:
                    continue
                if node_type == "PLAYER":
                    role_index = node.get("role_index", 0)
                    if role_index >= len(self.player_models):
                        raise ValueError(
                            f"Player model not available for role index {role_index}"
                        )
                    roles = self.game_instance.get("roles", [])
                    role_config = roles[role_index]
                    role_name = role_config.get("name")
                    handler_type = role_config.get("handler_type", DEFAULT_HANDLER_TYPE)
                    allowed_components = role_config.get("allowed_components", [])
                    initial_prompt = role_config.get("initial_prompt")
                    player = RoleBasedPlayer(
                        self.player_models[role_index],
                        role=role_name,
                        handler_type=handler_type,
                        allowed_components=allowed_components,
                    )
                    self.add_player(
                        player=player, initial_prompt=initial_prompt, node_id=node_id
                    )
            for edge in graph_config.get("edges", []):
                from_node = edge.get("from")
                to_node = edge.get("to")
                edge_type = edge.get("type")
                description = edge.get("description", "")
                if not from_node or not to_node or not edge_type:
                    continue
                if edge_type == "STANDARD":
                    self.add_standard_edge(from_node, to_node, description)
                elif edge_type == "DECISION":
                    condition_config = edge.get("condition", {})
                    condition_type = condition_config.get("condition_type")
                    if (
                        not condition_type
                        or condition_type not in ConditionType._value2member_map_
                    ):
                        raise KeyError(
                            f"Invalid condition-type field: {condition_type}"
                        )
                    condition = EdgeCondition(
                        condition_type=condition_type, description=description
                    )
                    self.add_decision_edge(from_node, to_node, condition, description)
            anchor_node = graph_config.get("anchor_node")
            if anchor_node:
                self.set_anchor_node(anchor_node)
            logger.info("Graph building complete")
        except Exception as e:
            raise RuntimeError(f"Failed to build interaction graph: {str(e)}") from e

    def _setup_after_parse_pipelines(self) -> None:
        """Initialize processing pipelines for different parsers.

        Returns:
            None: No return value
        """
        self.pipe_manager.register_pipeline(
            "pyautogui_actions",
            [
                PipeStage(
                    self._execute_actions,
                    output_field="observation",
                    description="applies player actions to environment, producing state-based observations.",
                )
            ],
        )
        self.pipe_manager.register_pipeline(
            "done_or_fail",
            [
                PipeStage(
                    self._execute_actions,
                    output_field="observation",
                    description="applies player actions (either done or fail) to environment, producing final-state-based observation",
                )
            ],
        )

    def _does_game_proceed(self) -> bool:
        """Determine if the game should continue to the next turn.

        Returns:
            bool: False if game is completed or max steps reached, True otherwise
        """
        return self.current_node != "END"

    def _on_before_game(self):
        """Executed once at the start, before entering the play loop.

        Key Actions:
            - Adds the initial game-context to the anchor player
        """
        super()._on_before_game()
        assert self.current_node == self.anchor_node, (
            "Current node must be the anchor node at game start"
        )
        current_player = self.get_player_from_node(self.current_node)
        formatted_context = self.player_context_formatter.create_context_for(
            self.message_state, current_player
        )
        if "image" in formatted_context and formatted_context["image"]:
            self.set_context_for(
                current_player,
                formatted_context["content"],
                image=formatted_context["image"],
            )
        else:
            self.set_context_for(current_player, formatted_context["content"])
        logger.info(f"Set initial context for player at node {self.current_node}")

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        """Method executed after a player response has been parsed and validated.

        Key Actions:
            - Updates the message state with the parsed response
            - Sets context for the next node's player using the current message state

        Args:
            player: The Player instance that produced the response
            parsed_response: The parsed and valid response of the current player
        """
        if self.transition.next_node:
            next_player = self.get_player_from_node(self.transition.next_node)
            if next_player:
                formatted_context = self.player_context_formatter.create_context_for(
                    self.message_state, next_player
                )
                if "image" in formatted_context and formatted_context["image"]:
                    self.set_context_for(
                        next_player,
                        formatted_context["content"],
                        image=formatted_context["image"],
                    )
                else:
                    self.set_context_for(next_player, formatted_context["content"])
                logger.info(
                    f"Set context for next player at node {self.transition.next_node}"
                )

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """Decide if an utterance should be added to the conversation history.

        Args:
            player: The Player instance for which the response is added
            utterance: The text content of the message to be added

        Returns:
            bool: True if the utterance is valid, False otherwise
        """
        print("::VALIDAION UTTERANCE::", utterance)
        if utterance is None:
            return False
        player_node = self.get_node_from_player(player)
        decision_edges = [
            (to_node, edge_data["condition"])
            for _, to_node, edge_data in self.graph.out_edges(player_node, data=True)
            if edge_data.get("type") == EdgeType.DECISION and edge_data.get("condition")
        ]
        # If no decision edges, check for exactly one standard edge
        if not decision_edges:
            standard_edges = [
                (to_node, edge_data)
                for _, to_node, edge_data in self.graph.out_edges(
                    player_node, data=True
                )
                if edge_data.get("type") == EdgeType.STANDARD
            ]
            if len(standard_edges) != 1:
                logger.warning(
                    f"Node {player_node} has {len(standard_edges)} standard edges, expected exactly 1"
                )
                return False
            return True
        # Check each decision edge's condition
        for to_node, condition in decision_edges:
            validation_result = condition.validate(utterance)
            # Case 1: Valid and intended format - successful validation
            if validation_result.is_valid and validation_result.intended_format:
                return True
            # Case 2: Invalid but intended format - validation failed for intended format
            elif not validation_result.is_valid and validation_result.intended_format:
                if validation_result.error:
                    logger.warning(
                        f"Validation failed for edge to {to_node}: {validation_result.error.message}"
                    )
                return False
            # Case 3: Not intended format - continue to next edge
            elif not validation_result.intended_format:
                continue
        # Finally, resort to a fallback procedure (TODO)
        logger.warning(
            "Validation failed, utterance did not comply with available conditions for connecting edges"
        )
        return False

    def _parse_response_for_decision_routing(
        self, player: Player, utterance: str
    ) -> Tuple[str, bool, Optional[str], Optional[Any]]:
        """Parse player response and evaluate decision edge conditions.

        Key Actions:
            1. Parse the player's utterance for relevant content
            2. Evaluate decision edge conditions based on the parsed content
            3. Determine which decision edge (if any) should be taken

        Args:
            player: The Player instance that produced the response
            utterance: The text content of the response

        Returns:
            Tuple[str, bool, Optional[str], Optional[Any]]: Modified utterance, logging flag, next node ID, extracted content
        """
        player_node = self.get_node_from_player(player)
        decision_edges = [
            (to_node, edge_data["condition"])
            for _, to_node, edge_data in self.graph.out_edges(player_node, data=True)
            if edge_data.get("type") == EdgeType.DECISION and edge_data.get("condition")
        ]
        print("::DECISION_EDGES::", decision_edges)
        if not decision_edges:
            return utterance, False, None, None
        # Evaluate each decision edge condition
        for to_node, condition in decision_edges:
            print("::TO_NODE::", to_node, "::CONDITION::", condition)
            try:
                parse_result = condition.parse(utterance)
                print(
                    "::IS_SUCCESSFUL, EXTRACTED_CONTENT::",
                    parse_result.is_successful,
                    parse_result.content,
                )
                if parse_result.is_successful and parse_result.content:
                    parser_id = None
                    parse_func_name = condition.function_pair.parse_func.__name__
                    for pid in parsers:
                        if (
                            parse_func_name.startswith(f"parse_{pid}")
                            or pid in parse_func_name
                        ):
                            parser_id = pid
                            break
                    if parser_id:
                        metadata = get_parser_metadata(parser_id)
                        target_field = metadata.get("target_field")
                        if target_field and hasattr(self.message_state, target_field):
                            self.message_state.update(
                                **{target_field: parse_result.content}
                            )
                        success, result = self.pipe_manager.execute_pipeline(
                            parser_id=parser_id,
                            content=parse_result.content,
                            message_state=self.message_state,
                        )
                        if success:
                            logger.info(
                                f"Processing pipeline executed for parser {parser_id}"
                            )
                    logger.info(
                        f"Decision edge condition met: {self.current_node} → {to_node}"
                    )
                    return (
                        utterance,
                        True,
                        to_node,
                        (
                            parse_result.content,
                            result if "result" in locals() else None,
                        ),
                    )
            except Exception as e:
                logger.error(
                    f"Error evaluating condition for edge to {to_node}: {str(e)}"
                )
        return utterance, False, None, None

    def _execute_actions(
        self, content: str
    ) -> Optional[Dict[str, Union[str, Image.Image, Dict]]]:
        """Execute either pyautogui or computer13 actions and record observations (upon environment-state change).

        Args:
            content: Parser extracted actions typically either pyautogui python-code (as str.), or computer13 actions in JSON (as str.)

        Returns:
            Optional[Dict]: Observation dictionary or None if execution fails
        """
        print("::EXECUTE ACTION::", content)
        try:
            if not content:
                logger.error("No actions to execute")
                return None
            observation = None
            for action in content:
                try:
                    observation, reward, done, info = self.game.env.step(
                        action, self.game.sleep_after_execution
                    )
                    print(reward, done, info)
                    if observation is None:
                        logger.error("Recieved None observation after game execution")
                        return None
                    self.current_observation = observation
                    if done:
                        logger.info("Game termination signal recieved (done=True)")
                        break
                except Exception as e:
                    logger.error(f"Failed to execute action {str(action)}: {str(e)}")
            return observation
        except Exception as e:
            logger.error(f"Action execution failed: {str(e)}")
            return None


class ComputerGameBenchmark(GameBenchmark):
    def create_game_master(
        self, experiment: Dict, player_models: List[backends.Model]
    ) -> GameMaster:
        return ComputerGameMaster(
            self.game_name, self.game_path, experiment, player_models
        )


if __name__ == "__main__":

    def load_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    experiments = load_json("./in/instances.json")
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"][0]
    master = ComputerGameMaster("computergame", None, experiment_1, ["mock", "mock"])
    master.setup(**game_1)
