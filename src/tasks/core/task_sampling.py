"""
Task Sampling Interface for Dynamic Framework Task Generation

Provides YAML-based configuration for sampling tasks across multiple categories.
Supports flexible task generation with validation, error handling, and session management.
"""

import os
import json
import yaml
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .generator import FrameworkTaskGenerator


@dataclass
class TaskSpec:
    """Specification for a single task to be generated."""

    category: str
    task_type: str
    level: int
    instance: int

    def __str__(self) -> str:
        return f"{self.category}.{self.task_type}.L{self.level}.inst{self.instance}"


class TaskPackage:
    """Container for a generated task with all its components."""

    def __init__(self, spec: TaskSpec, task_id: str):
        self.spec = spec
        self.task_id = task_id
        self.framework_config = {}
        self.local_files = {}
        self.s3_urls = {}
        self.metadata = {}
        self.package_dir = ""

    def create_package_directory(self, base_path: str) -> str:
        """Create isolated package directory for this task."""
        package_name = f"task_{self.task_id[:8]}_{self.spec.category}_{self.spec.task_type}_L{self.spec.level}_inst{self.spec.instance + 1}"
        self.package_dir = os.path.join(base_path, "tasks", package_name)

        # Create package structure
        os.makedirs(self.package_dir, exist_ok=True)
        os.makedirs(os.path.join(self.package_dir, "files"), exist_ok=True)

        return self.package_dir

    def save_local_files(self):
        """Copy generated files to package directory."""
        files_dir = os.path.join(self.package_dir, "files")

        if not self.local_files:
            # Config-only task - files directory remains empty
            return

        # Copy main file
        if "main_file" in self.local_files:
            main_file_dest = os.path.join(files_dir, self.local_files["main_filename"])
            shutil.copy2(self.local_files["main_file"], main_file_dest)

        # Copy additional files
        if "additional_files" in self.local_files:
            for orig_filename, file_info in self.local_files["additional_files"].items():
                dest_path = os.path.join(files_dir, file_info["filename"])
                shutil.copy2(file_info["local_path"], dest_path)

        # Copy ground truth files
        if "gold_standard_file" in self.local_files:
            ground_truth_dir = os.path.join(files_dir, "ground_truth")
            os.makedirs(ground_truth_dir, exist_ok=True)

            # Extract filename from path
            gt_filename = os.path.basename(self.local_files["gold_standard_file"])
            gt_dest = os.path.join(ground_truth_dir, gt_filename)
            shutil.copy2(self.local_files["gold_standard_file"], gt_dest)

        # Copy new ground truth files (for debugging_and_refactoring category)
        if "ground_truth_files" in self.local_files:
            ground_truth_dir = os.path.join(files_dir, "ground_truth")
            os.makedirs(ground_truth_dir, exist_ok=True)

            ground_truth_files = self.local_files["ground_truth_files"]
            for file_key, file_path in ground_truth_files.items():
                if file_key.endswith("_file") and os.path.exists(file_path):
                    # Extract filename from the path
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(ground_truth_dir, filename)
                    shutil.copy2(file_path, dest_path)

        # Copy JSON specification files (for image_processing category level 3)
        if "json_specification" in self.local_files:
            ground_truth_dir = os.path.join(files_dir, "ground_truth")
            os.makedirs(ground_truth_dir, exist_ok=True)

            json_spec_path = self.local_files["json_specification"]
            if os.path.exists(json_spec_path):
                # Extract filename from the path
                filename = os.path.basename(json_spec_path)
                dest_path = os.path.join(ground_truth_dir, filename)
                shutil.copy2(json_spec_path, dest_path)

    def save_metadata(self):
        """Save task metadata to package directory."""
        metadata_path = os.path.join(self.package_dir, "task_metadata.json")
        enhanced_metadata = {
            **self.metadata,
            "task_spec": {
                "category": self.spec.category,
                "task_type": self.spec.task_type,
                "level": self.spec.level,
                "instance": self.spec.instance,
            },
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_metadata, f, indent=2)

    def save_framework_config(self):
        """Save framework config to package directory."""
        config_path = os.path.join(self.package_dir, "framework_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.framework_config, f, indent=2)


class GenerationSession:
    """Manages a complete task generation session."""

    def __init__(self, output_config: Dict[str, Any], sampling_config: List[Dict[str, Any]]):
        self.session_id = self._generate_session_id(output_config.get("session_name"))
        self.timestamp = datetime.now()
        self.output_config = output_config
        self.sampling_config = sampling_config
        self.successful_tasks: List[TaskPackage] = []
        self.failed_tasks: List[Dict[str, Any]] = []
        self.output_dir = self._create_output_dir()

    def _generate_session_id(self, custom_name: Optional[str] = None) -> str:
        """Generate session ID with timestamp and optional custom name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_name:
            return f"session_{timestamp}_{custom_name}"
        return f"session_{timestamp}"

    def _create_output_dir(self) -> str:
        """Create output directory structure."""
        base_dir = self.output_config.get("output_dir", "outputs")
        session_dir = os.path.join(base_dir, self.session_id)

        # Create directory structure
        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(os.path.join(session_dir, "tasks"), exist_ok=True)

        return session_dir

    def add_successful_task(self, task_package: TaskPackage):
        """Add a successfully generated task to the session."""
        # Files are already saved by _generate_task_package
        # Just add to the successful tasks list
        self.successful_tasks.append(task_package)

    def add_failed_task(self, spec: TaskSpec, error: str):
        """Add a failed task to the session."""
        self.failed_tasks.append({"spec": spec, "error": error, "timestamp": datetime.now().isoformat()})

    def save_session_manifest(self):
        """Save comprehensive session manifest."""
        manifest = {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "sampling_config": self.sampling_config,
            "output_config": self.output_config,
            "results": {
                "successful_tasks": len(self.successful_tasks),
                "failed_tasks": len(self.failed_tasks),
                "total_requested": len(self.successful_tasks) + len(self.failed_tasks),
            },
            "successful_tasks": [
                {
                    "task_id": task.task_id,
                    "category": task.spec.category,
                    "task_type": task.spec.task_type,
                    "level": task.spec.level,
                    "instance": task.spec.instance,
                    "package_path": os.path.relpath(task.package_dir, self.output_dir),
                    "has_files": len(task.local_files) > 0,
                }
                for task in self.successful_tasks
            ],
            "failed_tasks": [
                {
                    "category": failure["spec"].category,
                    "task_type": failure["spec"].task_type,
                    "level": failure["spec"].level,
                    "instance": failure["spec"].instance,
                    "error": failure["error"],
                    "timestamp": failure["timestamp"],
                }
                for failure in self.failed_tasks
            ],
        }

        manifest_path = os.path.join(self.output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def save_generation_config(self):
        """Save the configuration used for this generation session."""
        config = {"sampling": self.sampling_config, "output": self.output_config}

        config_path = os.path.join(self.output_dir, "sampling_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def save_framework_configs(self):
        """Save flat list of framework configs for framework consumption."""
        framework_configs = [task.framework_config for task in self.successful_tasks]

        configs_path = os.path.join(self.output_dir, "framework_configs.json")
        with open(configs_path, "w", encoding="utf-8") as f:
            json.dump(framework_configs, f, indent=2)

    def upload_to_s3(self, s3_manager, bucket_name: str):
        """Upload all task files to S3 and update framework configs with URLs."""
        tasks_to_upload = [task for task in self.successful_tasks if not task.s3_urls]

        if not tasks_to_upload:
            print("📤 All tasks already uploaded to S3 during generation")
            return

        print(f"📤 Uploading {len(tasks_to_upload)} tasks to S3...")

        for task in tasks_to_upload:
            try:
                # Create temporary generator for S3 upload
                with FrameworkTaskGenerator(s3_manager, bucket_name) as generator:
                    # Upload files and get URLs
                    s3_urls = generator.file_manager.upload_task_files(task.task_id, task.local_files)
                    task.s3_urls = s3_urls

                    # Update framework config with S3 URLs
                    self._update_framework_config_with_s3_urls(task)

                print(f"  ✅ Uploaded {task.spec}")

            except Exception as e:
                print(f"  ❌ Failed to upload {task.spec}: {e}")

        # Regenerate framework configs with S3 URLs
        self.save_framework_configs()
        print("✅ S3 upload completed")

    def _update_framework_config_with_s3_urls(self, task: TaskPackage):
        """Update framework config download URLs with S3 URLs."""
        if not task.s3_urls:
            return

        # Update main file URL in download config
        for config_step in task.framework_config.get("config", []):
            if config_step.get("type") == "download":
                files = config_step.get("parameters", {}).get("files", [])
                if files and "main_file" in task.s3_urls:
                    files[0]["url"] = task.s3_urls["main_file"]

                # Update additional files
                if "additional_files" in task.s3_urls and len(files) > 1:
                    additional_idx = 1
                    for orig_filename, file_info in task.s3_urls["additional_files"].items():
                        if additional_idx < len(files):
                            files[additional_idx]["url"] = file_info["url"]
                            additional_idx += 1

        # Update evaluator expected files
        evaluator = task.framework_config.get("evaluator", {})
        if "gold_standard_url" in task.s3_urls and "expected" in evaluator:
            if isinstance(evaluator["expected"], dict) and evaluator["expected"].get("type") == "cloud_file":
                evaluator["expected"]["path"] = task.s3_urls["gold_standard_url"]


class TaskSamplingInterface:
    """Main interface for task sampling and generation."""

    def __init__(self):
        # Import and initialize category registry
        from src.tasks.core.category_registry import category_registry

        self.category_registry = category_registry
        self.category_registry.discover_categories()

    def sample_from_config(self, config_path: str) -> GenerationSession:
        """Generate tasks from YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        return self._execute_sampling(config)

    def sample_from_dict(self, config: Dict[str, Any]) -> GenerationSession:
        """Generate tasks from configuration dictionary."""
        return self._execute_sampling(config)

    def _execute_sampling(self, config: Dict[str, Any]) -> GenerationSession:
        """Execute the sampling based on configuration."""
        print("🎯 Starting task sampling session...")

        # 1. Validate configuration
        self._validate_sampling_config(config)
        print("✅ Configuration validated")

        # 2. Generate task specifications
        task_specs = self._generate_task_specs(config["sampling"])
        print(f"📋 Generated {len(task_specs)} task specifications")

        # 3. Create session structure
        session = GenerationSession(config.get("output", {}), config["sampling"])
        print(f"📁 Created session: {session.session_id}")

        # 4. Generate each task (continue on failures)
        print("⚡ Generating tasks...")
        for i, spec in enumerate(task_specs, 1):
            try:
                print(f"  Processing {i}/{len(task_specs)}: {spec}")
                task_package = self._generate_task_package(spec, session)
                session.add_successful_task(task_package)
                print(f"    ✅ Success: {task_package.task_id}")

            except Exception as e:
                session.add_failed_task(spec, str(e))
                print(f"    ❌ Failed: {str(e)}")
                continue

        # 5. Save session files
        session.save_session_manifest()
        session.save_generation_config()
        session.save_framework_configs()

        # 6. Upload to S3 if enabled
        upload_s3 = config.get("output", {}).get("upload_s3", False)
        if upload_s3:
            s3_manager, bucket_name = self._get_s3_setup()
            session.upload_to_s3(s3_manager, bucket_name)

        # Print summary
        self._print_session_summary(session)

        return session

    def _validate_sampling_config(self, config: Dict[str, Any]):
        """Validate the sampling configuration."""
        if "sampling" not in config:
            raise ValueError("Configuration must contain 'sampling' section")

        if not isinstance(config["sampling"], list):
            raise ValueError("'sampling' section must be a list")

        for i, sample_config in enumerate(config["sampling"]):
            self._validate_single_sample_config(sample_config, i)

    def _validate_single_sample_config(self, sample_config: Dict[str, Any], index: int):
        """Validate a single sampling configuration entry."""
        required_fields = ["category", "levels", "instances_per_task"]
        for field in required_fields:
            if field not in sample_config:
                raise ValueError(f"Sampling config {index}: Missing required field '{field}'")

        # Validate category exists
        category_name = sample_config["category"]
        category = self.category_registry.get_category(category_name)
        if not category:
            available = self.category_registry.get_category_names()
            raise ValueError(f"Sampling config {index}: Unknown category '{category_name}'. Available: {available}")

        # Validate levels
        levels = sample_config["levels"]
        if not isinstance(levels, list) or not levels:
            raise ValueError(f"Sampling config {index}: 'levels' must be a non-empty list")

        supported_levels = category.get_supported_levels()
        for level in levels:
            if level not in supported_levels:
                raise ValueError(f"Sampling config {index}: Level {level} not supported by category '{category_name}'. Supported: {supported_levels}")

        # Validate task_types if specified
        if "task_types" in sample_config:
            task_types = sample_config["task_types"]
            if not isinstance(task_types, list):
                raise ValueError(f"Sampling config {index}: 'task_types' must be a list")

            # Check if at least one level contains each requested task type
            for task_type in task_types:
                task_found_in_any_level = False
                available_levels_for_task = []

                for level in levels:
                    available_tasks = category.get_task_types(level)
                    if task_type in available_tasks:
                        task_found_in_any_level = True
                        available_levels_for_task.append(level)

                if not task_found_in_any_level:
                    # Get all tasks across all specified levels for a better error message
                    all_available_tasks = set()
                    for level in levels:
                        all_available_tasks.update(category.get_task_types(level).keys())
                    raise ValueError(
                        f"Sampling config {index}: Task type '{task_type}' not available in category '{category_name}' in any of the specified levels {levels}. Available tasks: {sorted(list(all_available_tasks))}"
                    )

    def _generate_task_specs(self, sampling_config: List[Dict[str, Any]]) -> List[TaskSpec]:
        """Convert sampling config to specific task specifications."""
        task_specs = []

        for config in sampling_config:
            category_name = config["category"]
            levels = config["levels"]
            task_types = config.get("task_types", None)
            instances = config["instances_per_task"]

            # Get category
            category_obj = self.category_registry.get_category(category_name)

            for level in levels:
                level_tasks = category_obj.get_task_types(level)

                if task_types:
                    # Only generate specs for task types that exist in this level
                    for task_type in task_types:
                        if task_type in level_tasks:
                            for instance in range(instances):
                                task_specs.append(TaskSpec(category_name, task_type, level, instance))
                else:
                    # All tasks from this level
                    for task_type in level_tasks.keys():
                        for instance in range(instances):
                            task_specs.append(TaskSpec(category_name, task_type, level, instance))

        return task_specs

    def _generate_task_package(self, spec: TaskSpec, session: GenerationSession) -> TaskPackage:
        """Generate a single task package using existing generator."""
        # Create S3 manager (but don't upload yet)
        s3_manager, bucket_name = self._get_s3_setup()

        # Check if S3 upload is enabled for this session
        upload_s3 = session.output_config.get("upload_s3", False)

        # Use existing FrameworkTaskGenerator
        with FrameworkTaskGenerator(s3_manager, bucket_name) as generator:
            # Generate random seed for task variation
            import random

            seed = random.randint(1000, 9999)

            result = generator.generate_dynamic_task(
                task_type=spec.task_type,
                category=spec.category,
                level=spec.level,
                seed=seed,
                upload_to_s3=upload_s3,  # Upload to S3 immediately if enabled
                evaluation_mode=self._get_evaluation_mode(spec),
            )

            # Create task package
            package = TaskPackage(spec, result["task_id"])
            package.framework_config = result["framework_config"]
            package.metadata = result["metadata"]
            package.local_files = result["files_created"]
            package.s3_urls = result["s3_urls"]  # Will be populated if S3 upload was done

            # Copy files to permanent location immediately while temp files still exist
            package.create_package_directory(session.output_dir)
            package.save_local_files()
            package.save_metadata()
            package.save_framework_config()

            return package

    def _get_evaluation_mode(self, spec: TaskSpec) -> str:
        """Determine evaluation mode based on task level and category provider."""
        # Try to get evaluation mode from category provider
        try:
            from .category_registry import category_registry

            category = category_registry.get_category(spec.category)
            if category:
                config_provider = category.get_config_provider()
                if config_provider and config_provider.supports_task_type(spec.task_type):
                    return config_provider.get_evaluation_mode(spec.task_type, spec.level)
        except (AttributeError, ImportError):
            pass

        # Fallback: Level 1 typically uses compare_text_file, others use compare_answer
        return "compare_text_file" if spec.level == 1 else "compare_answer"

    def _get_s3_setup(self):
        """Get S3 manager setup."""
        from src.utils.s3_manager import S3Manager
        import os

        # Check if S3 credentials are available
        if not all(
            [
                os.getenv("AWS_ACCESS_KEY_ID"),
                os.getenv("AWS_SECRET_ACCESS_KEY"),
                os.getenv("AWS_REGION"),
            ]
        ):
            # Create mock S3Manager for local-only generation
            class MockS3Manager:
                def __init__(self):
                    pass

            return MockS3Manager(), "test-bucket"

        return S3Manager(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        ), os.getenv("S3_BUCKET_NAME", "thesis-bhavsar")

    def _print_session_summary(self, session: GenerationSession):
        """Print session generation summary."""
        print("\n📊 Generation Session Summary:")
        print(f"  Session ID: {session.session_id}")
        print(f"  Output Directory: {session.output_dir}")
        print(f"  ✅ Successful: {len(session.successful_tasks)}")
        print(f"  ❌ Failed: {len(session.failed_tasks)}")
        print("  �� Framework configs: framework_configs.json")
        print("  📋 Session manifest: manifest.json")

        if session.failed_tasks:
            print("\n❌ Failed tasks:")
            for failure in session.failed_tasks[:5]:  # Show first 5 failures
                print(f"  - {failure['spec']}: {failure['error']}")
            if len(session.failed_tasks) > 5:
                print(f"  ... and {len(session.failed_tasks) - 5} more failures")

    def get_available_categories(self) -> Dict[str, Any]:
        """Get information about all available categories."""
        categories_info = {}

        for category_name in self.category_registry.get_category_names():
            category = self.category_registry.get_category(category_name)
            categories_info[category_name] = {
                "supported_levels": category.get_supported_levels(),
                "tasks_by_level": {level: list(category.get_task_types(level).keys()) for level in category.get_supported_levels()},
            }

        return categories_info
