import json 
from pathlib import Path
from typing import Dict, List, Union, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

@dataclass(frozen=True)
class GameInstance:
    """Immutable container for automation game instance data."""
    game_id: str
    snapshot: str
    instruction: str
    source: str
    config: List[Dict[str, Any]]
    trajectory: str
    related_apps: List[str]
    evaluator: Union[Dict[str, Any], str]

class ExperimentOrchestrator:
    """
    Manages experiment configuration processing for desktop automation tasks.
    Maintains a persistent cache of experiments that can be loaded, updated and analyzed.
    """
    def __init__(self, max_workers: int = 4, cache_path: Path = Path("in/instances.json")):
        self.base_path = Path("./in/legacy")
        self.max_workers = max_workers
        self.cache_path = cache_path
        self._validate_directory()
        self.cached_data = self._load_or_create_cache()

    def _validate_directory(self) -> None:
        """Validates the legacy experiments directory structure."""
        if not self.base_path.exists():
            raise FileNotFoundError(f"Legacy directory not found: {self.base_path}")
        if not self.base_path.is_dir():
            raise NotADirectoryError(f"Invalid legacy path: {self.base_path}")

    @lru_cache(maxsize=128)
    def _load_json_file(self, file_path: Path) -> Dict:
        """
        Loads and parses a JSON file with caching.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON content as dictionary
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON file {file_path}: {str(e)}")
            return {}
    
    def _process_domain_instances(self, domain_path: Path) -> Dict:
        """
        Processes all game instances for a specific domain.
        
        Args:
            domain_path: Path to domain directory
            
        Returns:
            Structured domain data with game instances
        """
        game_instances = []
        
        for json_file in domain_path.glob("*.json"):
            try:
                data = self._load_json_file(json_file)
                if not data:
                    continue
                    
                instance = GameInstance(
                    game_id=data.get('id', ''),
                    snapshot=data.get('snapshot', ''),
                    instruction=data.get('instruction', ''),
                    source=data.get('source', ''),
                    config=data.get('config', []),
                    trajectory=data.get('trajectory', ''),
                    related_apps=data.get('related_apps', []),
                    evaluator=data.get('evaluator', {})
                )
                
                game_instances.append(instance.__dict__)
                
            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")
                print(f"Data content: {data}")
                continue
        
        return {
            "name": domain_path.name,
            "game_instances": game_instances
        }

    def _load_or_create_cache(self) -> Dict:
        """
        Loads existing cache if available, creates new one if needed,
        and updates with any missing data.
        """
        existing_data = {"experiments": []}
        
        # Try to load existing cache
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    existing_data = json.load(f)
                print(f"Loaded existing cache from {self.cache_path}")
            except Exception as e:
                print(f"Failed to load cache: {str(e)}, creating new cache")

        # Process directories and update cache with any missing data
        existing_domains = {exp["name"] for exp in existing_data["experiments"]}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_domain = {
                executor.submit(self._process_domain_instances, domain_path): domain_path
                for domain_path in self.base_path.iterdir()
                if domain_path.is_dir() and domain_path.name not in existing_domains
            }
            
            for future in future_to_domain:
                try:
                    domain_data = future.result()
                    if domain_data["game_instances"]:
                        existing_data["experiments"].append(domain_data)
                except Exception as e:
                    domain_path = future_to_domain[future]
                    print(f"Failed to process domain {domain_path}: {str(e)}")

        # Save updated cache
        with open(self.cache_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        return existing_data

    def get_all_experiments(self) -> Dict:
        """Returns all cached experiments."""
        return self.cached_data

    def get_domain_instances(self, domain_name: str) -> Union[Dict, None]:
        """
        Retrieves all game instances for a specific domain from cache.
        
        Args:
            domain_name: Name of the domain (e.g., 'chrome', 'gimp')
            
        Returns:
            Domain configuration with game instances or None if not found
        """
        for exp in self.cached_data["experiments"]:
            if exp["name"] == domain_name:
                return exp
        return None

    def get_game_by_id(self, game_id: str) -> Union[Dict, None]:
        """
        Retrieves a specific game instance by ID from cache.
        
        Args:
            game_id: The ID of the game to find
            
        Returns:
            Game instance data or None if not found
        """
        for exp in self.cached_data["experiments"]:
            for game in exp["game_instances"]:
                if game["game_id"] == game_id:
                    return game
        return None

def main():
    orchestrator = ExperimentOrchestrator(max_workers=4)
    all_data = orchestrator.get_all_experiments()

if __name__ == "__main__":
    main()

"""
ExperimentOrchestrator Usage Examples:

1. Initialize the orchestrator:
   orchestrator = ExperimentOrchestrator(max_workers=4)

2. Get all experiments data:
   all_data = orchestrator.get_all_experiments()
   # Returns dictionary with all experiments and their game instances

3. Get games for specific domain:
   chrome_games = orchestrator.get_domain_instances("chrome")
   # Returns dictionary with domain name and list of game instances
   # Returns None if domain not found

4. Get specific game by ID:
   game = orchestrator.get_game_by_id("game-id-here")
   # Returns dictionary with game instance data
   # Returns None if game ID not found

Game Instance Structure:
{
    "id": str,                      # Unique identifier
    "snapshot": str,                # Game snapshot data
    "instruction": str,             # Game instructions
    "source": str,                  # Source information
    "config": List[Dict],           # Configuration parameters
    "trajectory": str,              # Trajectory data
    "related_apps": List[str],      # Related applications
    "evaluator": Union[Dict, str]   # Evaluation criteria
}
"""