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
            "max_steps": 15,
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
            self.game.env.controller.start_recording()
        except AttributeError as e:
            self.terminated = True
            self.log_to_self(LogType.SETUP_ERROR.value, f"Recording controller not properly initialized: {str(e)}")
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

        for player in self.get_players(): 
            system_message = self.game.prompt_handler._get_system_message(self.instruction)
            self.add_message(player, system_message["content"], role="user")
            logger.info("Initial instruction prompt added for %s", player.descriptor)

    def prompt(self, player: Player, is_reprompt=False):
        """Execute the core interaction loop with a player.

        Args:
            player (Player): The player to interact with
            is_reprompt (bool): Whether this is a repeated prompt (NOTE: not used)
        """
        curr_context = self.game.prompt_handler._get_curr_context(self.current_observation)
        message = curr_context["content"]
        self.add_message(player, message, role="user")
        
        action_type = 'send message' if not is_reprompt else 'send message (reprompt)'
        action = {'type': action_type, 'content': message}
        self.log_event(from_='GM', to=player.descriptor, action=action)

        full_context = self.game.prompt_handler._get_ovr_context(self.instruction, self.current_observation)
        _prompt, _response, response_message = player(full_context, self.current_turn)

        action = {'type': 'get message', 'content': response_message}
        self.log_event(from_=player.descriptor, to="GM", action=action, call=(_prompt, _response))

        self._DialogueGameMaster__validate_parse_and_add_player_response(player, response_message)

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """Basic format validation for player responses.

        Args:
            utterance (str): Response to validate

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

    def _after_add_player_response(self, player: Player, utterance: str):
        """Updates interaction history with response and extracted actions.

        Args:
            utterance (str): Validated response text

        Note: Game terminates on history update failure
        """
        if not self.terminated:
            return 
        
        try:
            extracted_actions = getattr(self, '_temp_extracted_actions', [])
            
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

    def _on_after_turn(self, turn_idx: int):
        """Executes pending actions and updates game state.

        Args:
            turn_idx (int): Current turn index

        Note: Handles action execution, state updates, and cleanup
        """
        if not self.terminated:
            return 
        
        try:
            extracted_actions = getattr(self, '_temp_extracted_actions', [])
            
            if not extracted_actions:
                self.terminated = True
                self.log_to_self(LogType.ACTION_FAIL.value, "No actions extracted from response")
                self.log_to_self(LogType.GAME_STATE.value, "Game terminated: no actions to execute")
                return

            for action in extracted_actions:
                try:
                    self.current_observation, reward, done, info = (
                        self.game.env.step(action, self.game.sleep_after_execution)
                    )
                    
                    action_result = f"Action: {str(action)}, Reward: {reward}, Done: {done}"
                    if info:
                        action_result += f", Additional info: {str(info)}"
                    self.log_to_self(LogType.ACTION_EXEC.value, action_result)

                    if done: 
                        self.terminated = True 
                        self.log_to_self(LogType.GAME_STATE.value, "Game termination signal received (done=True)")
                    
                except Exception as e:
                    self.terminated = True
                    self.log_to_self(LogType.ACTION_FAIL.value, f"Failed to execute action {str(action)}: {str(e)}")
                    self.log_to_self(LogType.GAME_STATE.value, "Game terminated: action execution failed")
            
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

    