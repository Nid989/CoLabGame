from clemcore.clemgame.instances import GameInstanceGenerator


class ComputerGameInstanceGenerator(GameInstanceGenerator):
    """
    Generates instances for the computer game with different observation types.
    Creates experiments for single executor scenarios with various observation configurations.
    """

    # Constants for observation types
    OBSERVATION_TYPES = ["a11y_tree", "screenshot_a11y_tree", "screenshot"]

    def __init__(self, path: str):
        """
        Initialize the ComputerGameInstanceGenerator.

        Args:
            path: The path to the game directory
        """
        super().__init__(path)

    def on_generate(self, **kwargs):
        """
        Generate three experiments with different observation types.

        Args:
            **kwargs: Additional keyword arguments (unused for now)
        """
        for obs_type in self.OBSERVATION_TYPES:
            experiment_name = f"single_executor_{obs_type}"
            experiment = self.add_experiment(experiment_name)

            # Configure experiment
            experiment["environment_type"] = "osworld"
            experiment["templates"] = self._create_templates()
            experiment["game_instances"] = []  # Empty for now
            experiment["config"] = self._create_config(obs_type)

    def _create_templates(self):
        """
        Create templates with roles and empty graph.

        Returns:
            Dict: Templates configuration with roles and graph
        """
        return {
            "roles": self._create_roles(),
            "graph": {},  # Empty for now, will be adapted to be dynamic in future
        }

    def _create_roles(self):
        """
        Create executor role configuration.

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


# Example usage
if __name__ == "__main__":
    # Create generator instance
    generator = ComputerGameInstanceGenerator("./")

    # Generate instances
    generator.generate(filename="instances.json")

    print("Generated instances file: instances.json")
    print("Created experiments:")
    for obs_type in ComputerGameInstanceGenerator.OBSERVATION_TYPES:
        print(f"  - single_executor_{obs_type}")
