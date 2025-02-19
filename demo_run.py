import yaml
import logging
from pathlib import Path
from orchestrator import ExperimentOrchestrator
from master import DesktopGameMaster

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,  # Force configuration even if already configured elsewhere
    handlers=[
        logging.StreamHandler(),  # Add explicit console handler
        logging.FileHandler('game.log')  # Also log to a file for debugging
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Explicitly set logger level

def main():
    try:
        # Load environment config
        env_config_path = Path("resources/env_config.yaml")
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
            logger.info("Loaded environment configuration")

        # Initialize orchestrator and get specific game instance
        orchestrator = ExperimentOrchestrator()
        game_instance = orchestrator.get_game_by_id("0bf05a7d-b28b-44d2-955a-50b41e24012a")
        
        if not game_instance:
            raise ValueError("Game instance not found")
        logger.info(f"Loaded game instance: {game_instance['game_id']}")

        # Create game master instance
        game_master = DesktopGameMaster(
            name="desktop_demo",
            path="./",
            experiment={"name": "desktop_automation"},
            player_models=["gpt-4o"]
        )
        logger.info("Created DesktopGameMaster instance")

        # Setup the game with configurations
        game_master._on_setup(
            env_kwargs=env_config,
            game_kwargs=game_instance
        )
        logger.info("Game setup completed successfully")

        # Here you could add additional game loop logic if needed
        # For example:
        # while not game_master.game_completed and game_master.current_step < game_master.game.max_steps:
        #     game_master.step()

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during game setup: {e}")

if __name__ == "__main__":
    main() 