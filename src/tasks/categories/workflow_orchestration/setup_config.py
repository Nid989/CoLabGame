"""
Setup configuration for the Integrated Content Workflow category.
"""

from typing import Dict, Any, List


class WorkflowOrchestrationSetupConfig:
    """Setup configuration for integrated workflow tasks."""

    def __init__(self):
        """Initialize setup configuration."""
        pass

    def get_required_applications(self) -> List[str]:
        """Get list of applications required for this category."""
        return ["google-chrome", "libreoffice", "gimp", "gnome-terminal"]

    def get_setup_requirements(self) -> Dict[str, Any]:
        """Get setup requirements for the category."""
        return {
            "applications": self.get_required_applications(),
            "directories": ["/home/user/Desktop", "/home/user/Documents", "/home/user/Downloads"],
            "minimum_disk_space": "100MB",
            "estimated_completion_time": {1: "10-15 minutes", 2: "15-25 minutes", 3: "25-40 minutes"},
        }

    def get_application_launch_order(self, task_level: int) -> List[str]:
        """Get recommended application launch order for a given task level."""
        if task_level == 1:
            return ["google-chrome", "libreoffice", "gimp"]
        elif task_level == 2:
            return ["google-chrome", "libreoffice", "gimp"]
        elif task_level == 3:
            return ["google-chrome", "libreoffice", "gimp", "gnome-terminal"]
        else:
            return self.get_required_applications()

    def get_setup_instructions(self, task_type: str) -> str:
        """Get human-readable setup instructions for a task type."""
        instructions = {
            "basic_info_gathering": (
                "This task requires Chrome for web research, LibreOffice Writer for document creation, "
                "and GIMP for image processing. All applications will be automatically launched."
            )
        }
        return instructions.get(task_type, "Standard integrated workflow setup.")

    def validate_setup(self) -> bool:
        """Validate that the setup configuration is valid."""
        return True
