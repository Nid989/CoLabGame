from clemcore.clemgame.instances import GameInstanceGenerator


class ComputerGameInstanceGenerator(GameInstanceGenerator):
    """
    Generates instances for the computer game with different observation types.
    Creates experiments for single executor scenarios with various observation configurations.
    """

    # Constants for observation types
    OBSERVATION_TYPES = ["a11y_tree", "screenshot_a11y_tree", "screenshot"]
    MULTI_AGENT_OBSERVATION_TYPES = ["screenshot", "screenshot_a11y_tree"]

    def __init__(self, path: str):
        """
        Initialize the ComputerGameInstanceGenerator.

        Args:
            path: The path to the game directory
        """
        super().__init__(path)

    def on_generate(self, **kwargs):
        """
        Generate experiments with different observation types.
        Creates both single-agent and multi-agent experiments.

        Args:
            **kwargs: Additional keyword arguments (unused for now)
        """
        # Generate single-agent experiments
        for obs_type in self.OBSERVATION_TYPES:
            experiment_name = f"single_agent_{obs_type}"
            experiment = self.add_experiment(experiment_name)

            # Configure experiment
            experiment["environment_type"] = "osworld"
            experiment["templates"] = self._create_single_agent_templates()
            experiment["game_instances"] = []  # Empty for now
            experiment["config"] = self._create_config(obs_type)

        # Generate multi-agent star topology experiments
        for obs_type in self.MULTI_AGENT_OBSERVATION_TYPES:
            experiment_name = f"multi_agent_star_{obs_type}"
            experiment = self.add_experiment(experiment_name)

            # Configure experiment
            experiment["environment_type"] = "osworld"
            experiment["templates"] = self._create_multi_agent_templates()
            experiment["game_instances"] = []  # Empty for now
            experiment["config"] = self._create_config(obs_type)

    def _create_single_agent_templates(self):
        """
        Create templates for single-agent scenarios (graph generated dynamically).

        Returns:
            Dict: Templates configuration with roles and empty graph
        """
        return {
            "roles": self._create_single_agent_roles(),
            "graph": {},  # Empty - will be generated dynamically based on participants
        }

    def _create_multi_agent_templates(self):
        """
        Create templates for multi-agent star topology scenarios (graph generated dynamically).

        Returns:
            Dict: Templates configuration with roles and empty graph
        """
        return {
            "roles": self._create_multi_agent_roles(),
            "graph": {},  # Empty - will be generated dynamically based on participants
        }

    def _create_single_agent_roles(self):
        """
        Create executor role configuration for single-agent scenarios.

        Returns:
            List[Dict]: List containing the executor role configuration
        """
        return [
            {
                "name": "executor",
                "handler_type": "environment",
                "message_permissions": {"send": ["EXECUTE", "STATUS"], "receive": []},
                "allowed_components": ["goal", "observation"],
            }
        ]

    def _create_multi_agent_roles(self):
        """
        Create advisor and executor role configurations for multi-agent scenarios.
        These are base templates that will be dynamically customized based on participants.

        Returns:
            List[Dict]: List containing advisor and executor role configurations
        """
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

    def _create_config(self, observation_type: str):
        """
        Create config with specific observation type.

        Args:
            observation_type: The observation type for this experiment

        Returns:
            Dict: Configuration dictionary with the specified observation type
        """
        return {
            "headless": False,
            "observation_type": observation_type,
            "action_space": "pyautogui",
            "screen_width": 1920,
            "screen_height": 1080,
            "path_to_vm": "/Users/nidhirbhavsar/Desktop/WORK/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx",
            "sleep_after_execution": 0,
            "max_retries": 2,
            "max_rounds": 30,
            "max_transitions_per_round": 30,
            "player_consecutive_violation_limit": 3,
            "player_total_violation_limit": 5,
        }

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
        print("Created single-agent experiments:")
        for obs_type in ComputerGameInstanceGenerator.OBSERVATION_TYPES:
            print(f"  - single_agent_{obs_type}")

        print("Created multi-agent experiments:")
        for obs_type in ComputerGameInstanceGenerator.MULTI_AGENT_OBSERVATION_TYPES:
            print(f"  - multi_agent_star_{obs_type}")

        print("\n" + "=" * 50)
        print("EXAMPLE 1: Single executor setup")
        example_participants_1 = {
            "advisor": {"count": 1, "domains": ["task coordination"]},
            "executor": {"count": 1, "domains": ["general automation"]},
        }
        graph_1 = generator.generate_star_topology_graph(example_participants_1)
        dynamic_info_1 = generator.should_include_dynamic_domain_info(example_participants_1)
        print(f"Participants: {example_participants_1}")
        print(f"Graph nodes: {len(graph_1['nodes'])}")
        print(f"Graph edges: {len(graph_1['edges'])}")
        print(f"Anchor: {graph_1['anchor_node']}")
        print(f"Dynamic info for advisor: {dynamic_info_1['advisor']}")
        print(f"Dynamic info for executor: {dynamic_info_1['executor']}")

        print("\n" + "=" * 50)
        print("EXAMPLE 2: Multi-executor setup")
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
