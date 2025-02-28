import os
import logging
from enum import Enum
from typing import Dict, List, Tuple
import json

from clemcore.clemgame import Player, DialogueGameMaster, GameMaster, GameBenchmark
from clemcore.backends import Model
from clemcore.utils import file_utils

from game import DesktopGame, InteractiveAssistant
from utils import load_json, extract_actions

logger = logging.getLogger(__name__)

class LogType(Enum):
    """Log types for internal game master logging"""
    ACTION_INFO = "action_info"     # For successful action extractions and general action information
    ACTION_FAIL = "action_fail"     # For failed actions or errors during action execution
    ACTION_EXEC = "action_exec"     # For successful action execution results
    TURN_PLAN = "turn_plan"         # For logging turn planning and thought processes
    # TURN_SKIP = "turn_skip"       # For when no actions are available to execute
    TURN_FAIL = "turn_fail"         # For failures at the turn level
    VALIDATION = "validation"       # For validation related messages
    GAME_STATE = "game_state"       # For tracking game state transitions and termination conditions
    SETUP_ERROR = "setup_error"     # For initialization and setup related errors

class DesktopGameMaster(DialogueGameMaster):
    def __init__(self, name: str, path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(name, path, experiment, player_models)

        self.experiment: str = experiment["name"]

        self.game: DesktopGame = None
        self.game_instance: Dict = None

        self.terminated: bool = False # indicates when the game ends due to completion, failure, or wait state

    # NOTE: log_to_self + self.terminated = True; does not seem ideal for setup related issues, good idea will be to replace it with logger.error (if encountered)
    # however, we need to set self.terminated = True since game must not proceed further. 
    def _on_setup(self, **game_instance) -> None: 
        """Initializes game environment with provided configuration and registers player agents."""
        self.game_instance = game_instance
        
        # Environment configuration (hardcoded)
        # FIXME: need to change the way `path_to_vm` value is provided
        env_config = {
            "headless": False,
            "observation_type": "a11y_tree",
            "action_space": "pyautogui",
            "screen_width": 1920,
            "screen_height": 1080,
            "max_steps": 5,
            "max_trajectory_length": 3,
            "path_to_vm": "/Users/nidhirbhavsar/Desktop/WORK/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx",
            "sleep_after_execution": 0.0,
        }
        
        self.game = DesktopGame(
            **env_config,
            game_instance=self.game_instance
        )

        try:
            self.current_observation = (
                self.game.env.reset(task_config={
                    'id' if k == 'game_id' else k: v
                    for k, v in self.game_instance.items()
                })
            )
        except Exception as e:
            self.terminated = True
            self.log_to_self(LogType.SETUP_ERROR.value, f"Environment initialization failed: {str(e)}")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: failed to initialize game environment")

        self._initialize_recording()
        self._register_players()

    def _initialize_recording(self) -> None:
        try:
            # Use the standardized interface method instead of directly accessing controller
            if not self.game.env.start_recording():
                self.terminated = True
                self.log_to_self(LogType.SETUP_ERROR.value, "Failed to start environment recording")
                self.log_to_self(LogType.GAME_STATE.value, "Game terminated: failed to start environment recording")
        except Exception as e:
            self.terminated = True
            self.log_to_self(LogType.SETUP_ERROR.value, f"Recording initialization error: {str(e)}")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: failed to start environment recording")

    def _register_players(self) -> None:
        if not self.player_models:
            self.terminated = True
            self.log_to_self(LogType.SETUP_ERROR.value, "No player models available for registration")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: no players to register")
        
        try:
            # FIXME: make it more dynamic
            self.assistant = InteractiveAssistant(self.player_models[0])
            self.add_player(self.assistant)
        except Exception as e:
            self.terminated = True
            self.log_to_self(LogType.SETUP_ERROR.value, f"Failed to register players: {str(e)}")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: player registration failed")
    
    def _does_game_proceed(self) -> bool:
        """Determine if the game should continue to the next turn.
        Returns:
            bool: False if game is completed or max steps reached, True otherwise
        """
        return not self.terminated and self.current_turn < self.game.max_steps    

    def _on_before_game(self) -> None:
        """Initializes game instruction and adds system message to all players' dialogue history."""
        self.instruction = self.game_instance.get('instruction')
        if not self.instruction:
            self.terminated = True
            self.log_to_self(LogType.SETUP_ERROR.value, "Game instance missing required 'instruction' field")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: missing instruction")
            return

        for player in self.get_players(): 
            initial_context = self.game.prompt_handler._get_turn_context(
                self.current_observation,
                turn=self.current_turn,
                instruction=self.instruction
            )
            self.add_user_message(player, initial_context["content"])
            logger.info("Initial instruction prompt added for %s", player.descriptor)

    def _on_before_turn(self, turn_idx: int):
        """Updates the game's turn counter and player context before each turn.
        
        Args:
            turn_idx: The current turn index.
        """
        self.game._current_turn = turn_idx
        
        # Skip player context update for turn 0 as it's handled in _on_before_game
        if turn_idx > 0:
            for player in self.get_players():
                self._update_player_context(player)

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """Basic format validation for player responses.
        Args:
            utterance (str): Response/utterance to validate
        Returns:
            bool: Validation result
        """
        if not utterance or not isinstance(utterance, str):
            self.terminated = True
            self.log_to_self(LogType.VALIDATION.value, "Invalid response: empty or non-string message")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: received invalid/empty response")
            return False

        return True

    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        """Extracts executable actions from player response.
        Args:
            utterance (str): Response text to parse
        Returns:
            Tuple[str, bool]: (original utterance, parsing success)
        Note: Extracted actions stored in self._temp_extracted_actions
        """
        try:
            masks = self.current_observation.get('masks') if self.game.observation_type == 'som' else None
            extracted_actions = extract_actions(
                utterance,
                self.game.observation_type,
                self.game.action_space,
                masks
            )
            self._temp_extracted_actions = extracted_actions
            self.log_to_self(LogType.ACTION_INFO.value, f"Extracted actions: {', '.join(str(a) for a in extracted_actions)}")
            return utterance, True
            
        except ValueError as e:
            self._temp_extracted_actions = []
            self.terminated = True
            self.log_to_self(LogType.ACTION_FAIL.value, f"Error: {str(e)}, Response preview: {utterance[:100]}")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: failed to parse actions")
            return utterance, False

    def _execute_action(self, action) -> bool:
        """Execute a single action and process its results.
        
        Args:
            action: The action to execute
            
        Returns:
            bool: True if execution was successful, False if game should terminate
        """
        try:
            self.current_observation, reward, done, info = (
                self.game.env.step(action, self.game.sleep_after_execution)
            )

            if self.current_observation is None:
                print("none current observation")
                self.terminated = True
                self.log_to_self(LogType.ACTION_FAIL.value, "Received None observation after action execution")
                self.log_to_self(LogType.GAME_STATE.value, "Game terminated: invalid observation state")
                return False

            action_result = f"Action: {str(action)}, Reward: {reward}, Done: {done}"
            print(action_result)
            if info:
                action_result += f", Additional info: {str(info)}"
            self.log_to_self(LogType.ACTION_EXEC.value, action_result)

            if done:
                self.terminated = True
                self.log_to_self(LogType.GAME_STATE.value, "Game termination signal received (done=True)")
                return False

            return True

        except Exception as e:
            print(e)
            self.terminated = True
            self.log_to_self(LogType.ACTION_FAIL.value, f"Failed to execute action {str(action)}: {str(e)}")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: action execution failed")
            return False

    def _execute_all_actions(self, extracted_actions):
        """Execute all extracted actions in sequence.
        
        Args:
            extracted_actions: List of actions to execute
        """
        if not extracted_actions:
            self.terminated = True
            self.log_to_self(LogType.ACTION_FAIL.value, "No actions extracted from response")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: no actions to execute")
            return

        for action in extracted_actions:
            print(action)
            if not self._execute_action(action):
                break

    def _update_player_context(self, player: Player) -> None:
        """Updates player's context with the current observation.
        
        Args:
            player (Player): The player to update context for
        """
        try:
            turn_context = self.game.prompt_handler._get_turn_context(
                self.current_observation,
                turn=self.current_turn
            )
            message = turn_context["content"]
            self.add_user_message(player, message)
        except Exception as e:
            self.terminated = True
            self.log_to_self(LogType.TURN_FAIL.value, f"Failed to update player context: {str(e)}")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: failed to update player context")

    def _after_add_player_response(self, player: Player, utterance: str):
        """Updates interaction history with response and extracted actions, then executes actions.
        
        Args:
            utterance (str): Validated response text
        Note: Game terminates on history update failure or action execution failure
        """
        if self.terminated:
            return 
        
        extracted_actions = getattr(self, '_temp_extracted_actions', [])
        
        print(extracted_actions)

        # First try block: Update interaction history
        try:
            self.game.prompt_handler.update_interaction_history(
                thought=utterance,
                action=extracted_actions,
                obs=self.current_observation
            )
            
            self.log_to_self(LogType.TURN_PLAN.value, 
                f"Turn {self.current_turn}: Thought preview: {utterance[:100]}, Actions: {', '.join(str(a) for a in extracted_actions)}")
            
        except Exception as e:
            self.terminated = True
            self.log_to_self(LogType.ACTION_FAIL.value, f"Failed to update interaction history: {str(e)}")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: failed to update interaction history")
            return
        
        # Check termination before executing actions
        if self.terminated:
            return
        
        # Second try block: Execute actions only if history update was successful
        try:
            self._execute_all_actions(extracted_actions)
        except Exception as e:
            self.terminated = True
            self.log_to_self(LogType.TURN_FAIL.value, f"Turn {self.current_turn} failed: {str(e)}")
            self.log_to_self(LogType.GAME_STATE.value, "Game terminated: turn execution failed")
        finally:
            if hasattr(self, '_temp_extracted_actions'):
                del self._temp_extracted_actions

class DesktopGameBenchmakr(GameBenchmark):
    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return DesktopGameMaster(self.game_name, self.game_path, experiment, player_models)

if __name__ == "__main__":
    game_path = os.path.dirname(os.path.abspath(__file__))
    experiments = file_utils.load_json("in/instances.json", game_path)

    