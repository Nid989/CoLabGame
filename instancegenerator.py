import os
import sys
import yaml
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from clemcore.clemgame.instances import GameInstanceGenerator
from src.topologies.factory import TopologyFactory
from src.topologies.base import TopologyType
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

        # Validate all top-level sections exist
        required_sections = ["task_generation", "experiments", "system", "roles"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")

        # Validate each section with dedicated methods
        self._validate_task_generation(config["task_generation"])
        self._validate_experiments(config["experiments"])
        self._validate_system(config["system"])
        self._validate_roles(config["roles"], config["experiments"])

    def _validate_required_keys(self, data: dict, required_keys: list, context: str) -> None:
        """Generic validation for required keys."""
        for key in required_keys:
            if key not in data:
                raise ValueError(f"{context} missing required key: {key}")

    def _validate_data_type(self, data: any, expected_type: type, context: str) -> None:
        """Generic validation for data types."""
        if not isinstance(data, expected_type):
            raise ValueError(f"{context} must be {expected_type.__name__}, got {type(data).__name__}")

    def _validate_task_generation(self, task_gen: dict) -> None:
        """Validate task generation configuration."""
        self._validate_required_keys(task_gen, ["experiments"], "task_generation")

    def _validate_experiments(self, experiments: dict) -> None:
        """Validate experiments configuration."""
        for exp_name, exp_config in experiments.items():
            self._validate_required_keys(exp_config, ["environment_type", "observation_type", "participants"], f"Experiment '{exp_name}'")

            # Validate topology_type if present
            if "topology_type" in exp_config:
                self._validate_data_type(exp_config["topology_type"], str, f"Experiment '{exp_name}' topology_type")

    def _validate_system(self, system: dict) -> None:
        """Validate system configuration."""
        required_system_keys = [
            "vm_path",
            "screen_width",
            "screen_height",
            "max_rounds",
            "max_transitions_per_round",
            "player_consecutive_violation_limit",
            "player_total_violation_limit",
        ]
        self._validate_required_keys(system, required_system_keys, "System config")

    def _validate_roles(self, roles: dict, experiments: dict) -> None:
        """Validate roles section with topology consistency."""
        # Get used topologies
        used_topologies = self._extract_used_topologies(experiments)

        # Validate topology implementations exist
        self._validate_topology_types(used_topologies)

        # Validate role configurations for each used topology
        for topology in used_topologies:
            self._validate_topology_roles(roles, topology)

        # Validate consistency between roles and experiments
        self._validate_topology_consistency(roles, experiments)

    def _extract_used_topologies(self, experiments: dict) -> set:
        """Extract topology types used in experiments."""
        used_topologies = set()
        for exp_config in experiments.values():
            if "topology_type" in exp_config:
                used_topologies.add(exp_config["topology_type"])
        return used_topologies

    def _validate_topology_types(self, used_topologies: set) -> None:
        """Validate that all used topology types have implementations."""
        from src.topologies.factory import TopologyFactory

        # Get available topologies from factory
        available_topologies = TopologyFactory.get_available_topologies()
        available_topology_names = [t.value for t in available_topologies]

        # Check each used topology has an implementation
        for topology in used_topologies:
            if topology not in available_topology_names:
                raise ValueError(
                    f"Topology '{topology}' is used in experiments but has no implementation. Available topologies: {available_topology_names}"
                )

    def _validate_topology_roles(self, roles: dict, topology: str) -> None:
        """Validate roles for a specific topology."""
        if topology not in roles:
            raise ValueError(f"Missing roles configuration for topology: {topology}")

        topology_roles = roles[topology]
        self._validate_data_type(topology_roles, dict, f"Roles for {topology} topology")

        for role_name, role_config in topology_roles.items():
            self._validate_role_config(role_config, role_name, topology)

    def _validate_role_config(self, role_config: dict, role_name: str, topology: str) -> None:
        """Validate individual role configuration."""
        required_fields = ["name", "handler_type", "message_permissions", "allowed_components", "receives_goal"]
        self._validate_required_keys(role_config, required_fields, f"Role {role_name} in {topology} topology")

        # Validate message_permissions structure
        self._validate_message_permissions(role_config["message_permissions"], role_name, topology)

    def _validate_message_permissions(self, message_permissions: dict, role_name: str, topology: str) -> None:
        """Validate message permissions structure."""
        self._validate_data_type(message_permissions, dict, f"Role {role_name} in {topology} topology message_permissions")

        for perm_type in ["send", "receive"]:
            if perm_type not in message_permissions:
                raise ValueError(f"Role {role_name} in {topology} topology: message_permissions missing '{perm_type}' key")
            self._validate_data_type(
                message_permissions[perm_type], list, f"Role {role_name} in {topology} topology: message_permissions['{perm_type}']"
            )

    def _validate_topology_consistency(self, roles: dict, experiments: dict) -> None:
        """Validate consistency between roles and experiments."""
        for exp_name, exp_config in experiments.items():
            if "topology_type" in exp_config:
                topology = exp_config["topology_type"]
                participants = exp_config.get("participants", {})

                # Check that participants match role definitions
                self._validate_participants_roles_consistency(participants, roles.get(topology, {}), exp_name)

    def _validate_participants_roles_consistency(self, participants: dict, topology_roles: dict, exp_name: str) -> None:
        """Validate that participant roles exist in topology definition."""
        for participant_role in participants.keys():
            if participant_role not in topology_roles:
                raise ValueError(
                    f"Experiment '{exp_name}' uses participant role '{participant_role}' "
                    f"but this role is not defined in the topology roles configuration"
                )

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
        """Create graph configuration using topology factory."""
        experiment_config = self.unified_config["experiments"][experiment_name]
        topology_type = experiment_config.get("topology_type", "star")

        # Determine topology type
        if len(participants) == 1 and "executor" in participants:
            topology_type = "single"

        # Create topology instance
        topology = TopologyFactory.create_topology(TopologyType(topology_type))

        # Generate graph using topology
        return topology.generate_graph(participants)

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
        experiment_config = self.unified_config["experiments"][experiment_name]

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
            "sliding_window_size": system_config.get("sliding_window_size"),
            "topology_type": experiment_config.get("topology_type", "star"),
        }

    def _create_agent_templates(self, topology_type: str):
        """Create templates for agent scenarios using configuration."""
        roles_config = self.unified_config.get("roles", {})

        if topology_type not in roles_config:
            raise ValueError(f"Unknown topology type: {topology_type}")

        topology_roles = roles_config[topology_type]
        return {"roles": list(topology_roles.values()), "graph": {}}

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
            experiment["templates"] = self._create_agent_templates(experiment_config["topology_type"])

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
