import os
import sys
import yaml
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from clemcore.clemgame.instances import GameInstanceGenerator
import copy


class ComputerGameInstanceGenerator(GameInstanceGenerator):
    """
    Generates instances for the computer game with different observation types.
    Creates experiments for single executor scenarios with various observation configurations.
    """

    def __init__(self, path: str):
        """
        Initialize the ComputerGameInstanceGenerator.

        Args:
            path: The path to the game directory
        """
        super().__init__(path)
        self._cached_sessions = {}  # Cache entire GenerationSession objects
        self._load_unified_config()

    def _load_unified_config(self):
        """Load unified configuration from yaml file."""
        load_dotenv()

        # Set default S3 bucket name if not provided
        if not os.getenv("S3_BUCKET_NAME"):
            os.environ["S3_BUCKET_NAME"] = "thesis-bhavsar"

        task_generator_path = os.getenv("TASK_GENERATOR_PROJECT_PATH")
        if task_generator_path and task_generator_path not in sys.path:
            sys.path.append(task_generator_path)

        with open("unified_config.yaml", "r") as f:
            self.unified_config = yaml.safe_load(f)

        self._validate_unified_config(self.unified_config)

    def _validate_unified_config(self, config: dict) -> None:
        """Validate the unified configuration format."""

        # Required top-level keys
        required_keys = ["task_generation", "experiments", "system"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")

        # Validate task generation configuration
        task_gen = config["task_generation"]
        if "experiments" not in task_gen:
            raise ValueError("task_generation must contain 'experiments' section")

        # Validate experiments configuration
        experiments = config["experiments"]
        for exp_name, exp_config in experiments.items():
            required_exp_keys = ["environment_type", "observation_type", "participants"]
            for key in required_exp_keys:
                if key not in exp_config:
                    raise ValueError(f"Experiment '{exp_name}' missing required key: {key}")

        # Validate system configuration
        system = config["system"]
        required_system_keys = [
            "vm_path",
            "screen_width",
            "screen_height",
            "max_rounds",
            "max_transitions_per_round",
            "player_consecutive_violation_limit",
            "player_total_violation_limit",
        ]
        for key in required_system_keys:
            if key not in system:
                raise ValueError(f"System config missing required key: {key}")

    def _create_output_directory(self, experiment_name: str) -> str:
        """Create and return the output directory path for an experiment."""
        # Create base outputs directory
        base_dir = Path("outputs")
        base_dir.mkdir(exist_ok=True)

        # Create runs directory
        runs_dir = base_dir / "runs"
        runs_dir.mkdir(exist_ok=True)

        # Create run-specific directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)

        # Create experiment-specific directory
        experiment_dir = run_dir / experiment_name
        experiment_dir.mkdir(exist_ok=True)

        return str(experiment_dir)

    def _generate_tasks(self, config: dict, run_name: str):
        """
        Generate tasks using the TaskSamplingInterface from a given configuration.
        The 'run_name' is used to create a unique output directory.
        """
        from core.task_sampling import TaskSamplingInterface

        try:
            # Deep copy config to avoid modifying the original dict during the run
            run_config = copy.deepcopy(config)

            # Update output directory to use our structured approach
            output_dir = self._create_output_directory(run_name)
            run_config["output"]["output_dir"] = output_dir

            # Initialize the interface and generate tasks
            sampler = TaskSamplingInterface()
            session = sampler.sample_from_dict(run_config)

            # Check for failed tasks and log them
            if session.failed_tasks:
                print(f"Warning: {len(session.failed_tasks)} tasks failed to generate for run '{run_name}'")
                for failure in session.failed_tasks:
                    print(f"  - Failed task: {failure.get('spec', 'Unknown Spec')} -> {failure.get('error', 'Unknown Error')}")

            return session

        except Exception as e:
            print(f"FATAL: Error generating tasks for run '{run_name}': {str(e)}")
            raise

    def _extract_framework_configs(self, session) -> list:
        """Extract framework configs from GenerationSession."""
        framework_configs = []

        if not session.successful_tasks:
            print("Warning: No successful tasks generated in session")
            return framework_configs

        for task_package in session.successful_tasks:
            framework_configs.append(task_package.framework_config)

        print(f"Extracted {len(framework_configs)} framework configs from session")
        return framework_configs

    def _create_participants_config(self, experiment_config: dict) -> dict:
        """Create participants configuration from experiment config."""
        return experiment_config.get("participants", {})

    def _create_graph_config(self, experiment_name: str, participants: dict) -> dict:
        """Create graph configuration based on experiment type and participants."""
        # For now, return empty graph as mentioned that graph generation is handled elsewhere
        return {}

    def _merge_task_config_with_game_config(self, task_config: dict, game_config: dict) -> dict:
        """Merge task configuration with game-specific configuration."""
        # Create a merged task config that includes game-specific settings
        merged_config = task_config.copy()

        # Add any game-specific configurations here if needed
        # For now, we'll use the task_config as is since it should already be complete

        return merged_config

    def _create_experiment_config(self, experiment_name: str, observation_type: str) -> dict:
        """Create experiment configuration with system settings."""
        system_config = self.unified_config["system"]

        return {
            "headless": False,
            "observation_type": observation_type,
            "action_space": "pyautogui",
            "screen_width": system_config["screen_width"],
            "screen_height": system_config["screen_height"],
            "path_to_vm": system_config["vm_path"],
            "sleep_after_execution": 0,
            "max_retries": 2,
            "max_rounds": system_config["max_rounds"],
            "max_transitions_per_round": system_config["max_transitions_per_round"],
            "player_consecutive_violation_limit": system_config["player_consecutive_violation_limit"],
            "player_total_violation_limit": system_config["player_total_violation_limit"],
        }

    def _create_single_agent_templates(self):
        """Create templates for single-agent scenarios."""
        return {
            "roles": self._create_single_agent_roles(),
            "graph": {},  # Empty - will be generated dynamically based on participants
        }

    def _create_multi_agent_templates(self):
        """Create templates for multi-agent star topology scenarios."""
        return {
            "roles": self._create_multi_agent_roles(),
            "graph": {},  # Empty - will be generated dynamically based on participants
        }

    def _create_single_agent_roles(self):
        """Create executor role configuration for single-agent scenarios."""
        return [
            {
                "name": "executor",
                "handler_type": "environment",
                "message_permissions": {"send": ["EXECUTE", "STATUS"], "receive": []},
                "allowed_components": ["goal", "observation"],
            }
        ]

    def _create_multi_agent_roles(self):
        """Create advisor and executor role configurations for multi-agent scenarios."""
        return [
            {
                "name": "advisor",
                "handler_type": "standard",
                "message_permissions": {"send": ["REQUEST", "RESPONSE", "STATUS"], "receive": ["REQUEST", "RESPONSE"]},
                "allowed_components": ["request", "response", "goal"],
            },
            {
                "name": "executor",
                "handler_type": "environment",
                "message_permissions": {"send": ["EXECUTE", "REQUEST", "RESPONSE"], "receive": ["REQUEST", "RESPONSE"]},
                "allowed_components": ["observation", "request", "response"],
            },
        ]

    def on_generate(self, **kwargs):
        """
        Generate experiments with different observation types.
        This method optimizes task generation by sampling shared configurations only once
        and reusing the generated tasks across all experiments that reference them.
        """
        # Step 1: Pre-generate and cache all shared task configurations.
        task_gen_configs = self.unified_config.get("task_generation", {}).get("experiments", {})
        shared_configs = task_gen_configs.get("shared", {})
        cached_shared_task_configs = {}

        print("INFO: Generating shared task sets...")
        for shared_name, config in shared_configs.items():
            session_name = config.get("output", {}).get("session_name", shared_name)
            session = self._generate_tasks(config, f"shared__{session_name}")
            cached_shared_task_configs[shared_name] = self._extract_framework_configs(session)
        print("INFO: Finished generating shared task sets.")

        # Step 2: Generate all experiments, reusing shared tasks where possible.
        all_experiments = self.unified_config.get("experiments", {})
        individual_task_configs = task_gen_configs.get("individual", {})

        print("INFO: Generating experiments...")
        for experiment_name, experiment_config in all_experiments.items():
            experiment = self.add_experiment(experiment_name)
            experiment["environment_type"] = experiment_config["environment_type"]

            # Determine templates (single or multi-agent)
            participants = experiment_config["participants"]
            if len(participants) == 1 and "executor" in participants:
                experiment["templates"] = self._create_single_agent_templates()
            else:
                experiment["templates"] = self._create_multi_agent_templates()

            # Get the list of task configurations for this experiment
            task_gen_instruction = individual_task_configs.get(experiment_name)
            if not task_gen_instruction:
                raise ValueError(f"Task generation configuration not found for experiment: {experiment_name}")

            framework_configs = []
            if "use_shared" in task_gen_instruction:
                shared_name = task_gen_instruction["use_shared"]
                if shared_name not in cached_shared_task_configs:
                    raise ValueError(f"Experiment '{experiment_name}' references a non-existent or failed shared task set: '{shared_name}'")
                print(f"INFO: Reusing shared task set '{shared_name}' for experiment '{experiment_name}'.")
                framework_configs = cached_shared_task_configs[shared_name]
            else:
                # This is an individual, non-shared task configuration
                print(f"INFO: Generating individual tasks for experiment '{experiment_name}'.")
                session = self._generate_tasks(task_gen_instruction, experiment_name)
                framework_configs = self._extract_framework_configs(session)

            # Create final game instances for the experiment
            for task_config in framework_configs:
                game_id = str(uuid.uuid4())
                game_instance = self.add_game_instance(experiment, game_id)

                merged_task_config = self._merge_task_config_with_game_config(task_config, experiment_config)

                game_instance["task_config"] = merged_task_config
                game_instance["participants"] = participants

            experiment["config"] = self._create_experiment_config(experiment_name, experiment_config["observation_type"])
        print("INFO: Finished generating all experiments.")

    def validate_participants(self, participants: dict):
        """
        Validate participants configuration ensuring count == len(domains).

        Args:
            participants: Dictionary with participant configuration
                         e.g., {"advisor": {"count": 1, "domains": ["task coordination"]},
                               "executor": {"count": 2, "domains": ["web automation", "file management"]}}

        Raises:
            ValueError: If validation fails
        """
        for role_name, config in participants.items():
            count = config.get("count", 0)
            domains = config.get("domains", [])

            if count != len(domains):
                raise ValueError(f"Role '{role_name}': count ({count}) must equal number of domains ({len(domains)})")

            if count < 1:
                raise ValueError(f"Role '{role_name}': count must be at least 1")

        # Star topology specific validation
        if "advisor" not in participants:
            raise ValueError("Star topology requires an 'advisor' role")

        if participants["advisor"]["count"] != 1:
            raise ValueError("Star topology requires exactly 1 advisor")

        if "executor" not in participants:
            raise ValueError("Star topology requires at least one 'executor' role")

    def generate_star_topology_graph(self, participants: dict):
        """
        Generate a star topology graph based on participant configuration.

        Args:
            participants: Dictionary with participant configuration
                         e.g., {"advisor": {"count": 1, "domains": ["task coordination"]},
                               "executor": {"count": 2, "domains": ["web automation", "file management"]}}

        Returns:
            Dict: Complete graph configuration for star topology
        """
        self.validate_participants(participants)
        # NOTE: assumption is that there is only one advisor and multiple executors
        executor_count = participants["executor"]["count"]

        # Create nodes
        nodes = [{"id": "START", "type": "START"}, {"id": "advisor", "type": "PLAYER", "role_index": 0}]

        # Add executor nodes
        for i in range(executor_count):
            executor_id = f"executor_{i + 1}" if executor_count > 1 else "executor"
            nodes.append(
                {
                    "id": executor_id,
                    "type": "PLAYER",
                    "role_index": 1,  # All executors use the same role template
                }
            )

        nodes.append({"id": "END", "type": "END"})

        # Create edges for star topology with task flow: advisor -> executor -> advisor -> goal completion
        edges = [
            # Start with advisor (advisor initiates the task)
            {"from": "START", "to": "advisor", "type": "STANDARD", "description": ""}
        ]

        # Add bidirectional communication between advisor and each executor
        for i in range(executor_count):
            executor_id = f"executor_{i + 1}" if executor_count > 1 else "executor"

            # Advisor to executor (sends tasks/requests/responses)
            edges.extend(
                [
                    {"from": "advisor", "to": executor_id, "type": "DECISION", "condition": {"type": "REQUEST"}, "description": "REQUEST"},
                    {"from": "advisor", "to": executor_id, "type": "DECISION", "condition": {"type": "RESPONSE"}, "description": "RESPONSE"},
                ]
            )

            # Executor to advisor (reports completion, asks questions)
            edges.extend(
                [
                    {"from": executor_id, "to": "advisor", "type": "DECISION", "condition": {"type": "REQUEST"}, "description": "REQUEST"},
                    {"from": executor_id, "to": "advisor", "type": "DECISION", "condition": {"type": "RESPONSE"}, "description": "RESPONSE"},
                ]
            )

            # Executor self-loop for EXECUTE actions (performing the actual tasks)
            edges.append({"from": executor_id, "to": executor_id, "type": "DECISION", "condition": {"type": "EXECUTE"}, "description": "EXECUTE"})

        # Advisor completes the goal (STATUS to END)
        edges.append({"from": "advisor", "to": "END", "type": "DECISION", "condition": {"type": "STATUS"}, "description": "STATUS"})

        return {"nodes": nodes, "edges": edges, "anchor_node": "advisor"}

    def should_include_dynamic_domain_info(self, participants: dict):
        """
        Determine if domain information should be dynamically included in prompts.

        Args:
            participants: Dictionary with participant configuration

        Returns:
            Dict: Information about what dynamic info to include for each role type
        """
        # NOTE: assumption is that there is only one advisor and multiple executors
        executor_count = participants["executor"]["count"]

        return {
            "advisor": {
                "include_executor_domains": executor_count > 1,  # Only if multiple executors
                "executor_domains": participants["executor"]["domains"] if executor_count > 1 else [],
            },
            "executor": {
                "include_own_domain": executor_count > 1,  # Only if multiple executors need domain distinction
                "include_other_executors": executor_count > 1,  # Only if there are other executors
                "total_executors": executor_count,
            },
        }


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate computer game instances.")
    parser.add_argument("--test", action="store_true", help="Run test/demo code after generating instances")
    args = parser.parse_args()

    generator = ComputerGameInstanceGenerator("./")
    generator.generate(filename="instances.json")
    print("Generated instances file: instances.json")

    if args.test:
        print("Created experiments from unified configuration:")
        experiments = generator.unified_config["experiments"]
        for exp_name in experiments.keys():
            print(f"  - {exp_name}")

        print("\n" + "=" * 50)
        print("EXAMPLE 1: Single agent setup")
        example_participants_1 = {
            "executor": {"count": 1, "domains": []},
        }
        print(f"Participants: {example_participants_1}")

        print("\n" + "=" * 50)
        print("EXAMPLE 2: Multi-agent setup")
        example_participants_2 = {
            "advisor": {"count": 1, "domains": ["task coordination"]},
            "executor": {"count": 2, "domains": ["web automation", "file management"]},
        }
        graph_2 = generator.generate_star_topology_graph(example_participants_2)
        dynamic_info_2 = generator.should_include_dynamic_domain_info(example_participants_2)
        print(f"Participants: {example_participants_2}")
        print(f"Graph nodes: {len(graph_2['nodes'])}")
        print(f"Graph edges: {len(graph_2['edges'])}")
        print(f"Anchor: {graph_2['anchor_node']}")
        print(f"Dynamic info for advisor: {dynamic_info_2['advisor']}")
        print(f"Dynamic info for executor: {dynamic_info_2['executor']}")
