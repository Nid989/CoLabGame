
# Project Setup Guide

This guide provides step-by-step instructions on how to set up your Python project using `uv`, manage dependencies with `pyproject.toml`, and run your scripts within a consistent environment.

## 1. Initialize the Project with `uv`

First, initialize your project with the desired Python version:

```bash
uv init --python 3.11.11
```

This command sets up the project environment using Python 3.11.11.

## 2. Pin the Python Version

To ensure consistency across environments, pin the Python version:

```bash
uv python pin 3.11.11
```

This locks the project to use Python 3.11.11, preventing discrepancies due to version differences.

## 3. Create a Virtual Environment

Next, create a virtual environment within your project directory:

```bash
uv venv .venv --python 3.11.11
```

This command sets up a virtual environment named `.venv` using Python 3.11.11.

## 4. Prepare Dependencies

`uv` utilizes the `requirements.txt` file for dependency management. However, it requires a specific format:

- For editable (`-e`) modules, modify entries from `-e .path` to `path`. For example, change:

  ```
  -e .
  ```

  to

  ```
  .
  ```

Ensure you refer to the original `requirements.txt` for accurate modifications.

## 5. Add Dependencies with `uv`

Once the `requirements.uv.txt` is properly formatted, add the dependencies:

```bash
uv add -a requirements.uv.txt
```

This command generates a `uv.lock` file and updates the `pyproject.toml` file with the specified dependencies.

## 6. Running Scripts

To execute scripts within this setup, use the `uv run` command:

```bash
uv run python3 filename.py
```

Replace `filename.py` with the name of your script. This ensures the script runs within the `uv`-managed environment.

---

By following these steps, you can effectively set up and manage your Python project using `uv` and `pyproject.toml`.
