# README

## Clone os-world
Clone the os-world repository externally:
```bash
git clone <OS_WORLD_REPO_URL> path_to_somewhere_external/OSWorld
```

## (Optional) VSCode Setup
1. Create a `.env` file in the project root:
   ```bash
   echo "PYTHONPATH=./src:./path_to_somewhere_external/OSWorld" > .env
   ```
2. Add to `.vscode/settings.json`:
   ```json
   {
     "python.envFile": "${workspaceFolder}/.env"
   }
   ```
3. Reload VSCode (Ctrl+Shift+P → Reload Window)

## OSWORLD-UBUNTU Setup
- Clone the OSWorld repo:
  ```bash
  git clone https://github.com/xlang-ai/OSWorld path_to_somewhere_external/
  cd path_to_somewhere_external/OSxwWorld
  ```
- Install required dependencies:
  ```bash
  # Install the dependencies using 'requirements.txt' file.
  pip install requirements.txt
  # Alternatively:
  pip install desktop-env
  ```
- Install VMware Workstation Pro (for systems with Apple chips, use VMware Fusion) and configure the `vmrun` command
- See the official instructions: https://timothyxxx.github.io/OSWorld/installation/install_provider/vmware.html
- After you finish the setup for your provider, run the following minimal example to interact with the environment and verify that everything is working. The first run will install the environment (download, unzip, rename, start, snapshot named `init_state`), which may take some time:

```python
from desktop_env.desktop_env import DesktopEnv

example = {
    "id": "94d95f96-9699-4208-98ba-3c3119edf9c2",
    "instruction": "I want to install Spotify on my current system. Could you please help me?",
    "config": [
        {
            "type": "execute",
            "parameters": {
                "command": [
                    "python",
                    "-c",
                    "import pyautogui; import time; pyautogui.click(960, 540); time.sleep(0.5);"
                ]
            }
        }
    ],
    "evaluator": {
        "func": "check_include_exclude",
        "result": {
            "type": "vm_command_line",
            "command": "which spotify"
        },
        "expected": {
            "type": "rule",
            "rules": {
                "include": ["spotify"],
                "exclude": ["not found"]
            }
        }
    }
}

env = DesktopEnv(action_space="pyautogui")
obs = env.reset(task_config=example)
obs, reward, done, info = env.step("pyautogui.rightClick()")
```

In the end, you should see a successful right-click on the screen, indicating that your environment is ready.

## ComputerGame Setup
1. Install dependencies via uv:
    ```bash
    uv install
    ```

2. Run the "collab: free" experiment:
    ```bash
    uv run clem run -g computergame -m model_name --max_tokens 400
    ```

3. Update the VM path in `DEFAULT_ENV_CONFIG`:
    ```diff
    src/utils/constants.py:
    under DEFAULT_ENV_CONFIG:
    -    "path_to_vm": "<path_to_vm>"
    +    "path_to_vm": "/Users/nidhirbhavsar/Desktop/WORK/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx"
    ```

## Environment Variable
(Optional) export the `.env` file for uv:
```bash
export UV_ENV_FILE=.env
```

## Notes
> Alternatively, to use a different Python environment management tool, install dependencies via the `requirements.txt` file:
```bash
pip install -r requirements.txt
```