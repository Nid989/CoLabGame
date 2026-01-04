"""
Setup configuration logic for Information Synthesis & Presentation category.
Contains the logic for building setup configuration steps for tasks.
"""

from typing import Dict, Any, List


class ResearchSynthesisSetupConfig:
    """Configuration builder for task setup steps in the Information Synthesis & Presentation category."""

    def __init__(self):
        """Initialize setup configuration builder."""
        pass

    def build_config_steps(
        self,
        task_example: Dict[str, Any],
        s3_urls: Dict[str, str],
        files_created: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Build configuration steps for task setup.

        Args:
            task_example: Original test example
            s3_urls: S3 URLs for uploaded files
            files_created: Local file information

        Returns:
            List of configuration steps
        """
        steps = []
        level = task_example.get("level", 1)
        webpage_filename = task_example["webpage_filename"]

        # Step 1: Download main HTML file
        if "main_file" in s3_urls:
            steps.append(
                {
                    "type": "download",
                    "parameters": {
                        "files": [
                            {
                                "url": s3_urls["main_file"],
                                "path": f"/tmp/{webpage_filename}",
                            }
                        ]
                    },
                }
            )

        # Step 2: For Level 3, download additional files (downloadable content)
        if level == 3 and "additional_files" in s3_urls:
            additional_downloads = []
            for orig_filename, file_info in s3_urls["additional_files"].items():
                additional_downloads.append(
                    {
                        "url": file_info["url"],
                        "path": f"/tmp/files/{file_info['filename']}",
                    }
                )

            if additional_downloads:
                # Create files directory first
                steps.append(
                    {
                        "type": "command",
                        "parameters": {"command": ["mkdir", "-p", "/tmp/files"]},
                    }
                )

                # Download additional files
                steps.append({"type": "download", "parameters": {"files": additional_downloads}})

        # Step 3: Start local HTTP server for serving files (detached background process)
        steps.append(
            {
                "type": "execute",
                "parameters": {
                    "command": [
                        "python3",
                        "-c",
                        "import subprocess, os; "
                        "log=open('/tmp/http_server.log','a'); "
                        "p=subprocess.Popen(['python3','-m','http.server','8080','--directory','/tmp'], stdout=log, stderr=log, preexec_fn=os.setsid); "
                        "open('/tmp/http_server.pid','w').write(str(p.pid)); "
                        "print('Server started on port 8080')",
                    ]
                },
            }
        )

        # Step 4: Wait for server to start
        steps.append({"type": "sleep", "parameters": {"seconds": 2}})

        # Step 5: Launch Chrome with target webpage
        target_url = f"http://localhost:8080/{webpage_filename}"
        steps.append(
            {
                "type": "launch",
                "parameters": {"command": ["google-chrome", "--new-window", target_url]},
            }
        )

        # Step 6: Launch LibreOffice Impress
        steps.append({"type": "launch", "parameters": {"command": ["libreoffice", "--impress"]}})

        # Step 7: Activate Chrome window to start with focus on browser
        steps.append(
            {
                "type": "activate_window",
                "parameters": {"window_name": "Google Chrome", "strict": False},
            }
        )

        return steps
