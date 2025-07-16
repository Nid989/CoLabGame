import os
import io
import uuid
import shutil
import logging
import time
import tempfile
from PIL import Image
from typing import Optional, Dict, List

from .s3_manager import S3Manager

logger = logging.getLogger(__name__)


class ImageManager:
    """Handles saving and uploading of images during a game instance."""

    def __init__(
        self,
        game_id: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region: str,
        s3_bucket: str,
    ):
        """
        Initializes the ImageManager.
        Args:
            game_id: The unique identifier for the current game.
            aws_access_key_id: AWS access key.
            aws_secret_access_key: AWS secret key.
            aws_region: AWS region for the S3 bucket.
            s3_bucket: The name of the S3 bucket to upload images to.
        """
        self.game_id = game_id
        self.s3_bucket = s3_bucket
        # Add short UUID to ensure unique s3_prefix even with same game_id
        short_uuid = str(uuid.uuid4())[:8]
        self.s3_prefix = f"game_sessions/{game_id}_{short_uuid}"
        self.s3_manager = S3Manager(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

        self.temp_dir = tempfile.mkdtemp(prefix=f"game_sessions/{game_id}_{short_uuid}/")

        # Registry to track uploaded images: {image_id: local_path}
        self.image_registry: Dict[str, str] = {}
        # List to maintain order of uploaded images
        self.image_order: List[str] = []
        # Counter for generating sequential image IDs
        self.image_counter = 0

        logger.info(f"ImageManager initialized. Temp directory: {self.temp_dir}")

    def save_image(self, image_bytes: bytes) -> None:
        """
        Converts image bytes to PNG, saves it locally, uploads to S3, and registers the image.
        Args:
            image_bytes: The raw bytes of the image.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))

            # Generate chronologically sortable image ID
            timestamp = int(time.time() * 1000000)  # Microsecond precision
            self.image_counter += 1
            image_id = f"{timestamp:016d}_{self.image_counter:06d}"

            filename = f"{image_id}.png"
            local_path = os.path.join(self.temp_dir, filename)

            # Save image locally as PNG
            image.save(local_path, "PNG")
            logger.info(f"Saved image locally to {local_path}")

            # Upload to S3
            s3_key = f"{self.s3_prefix}/{filename}"
            self.s3_manager.upload_file(Filename=local_path, Bucket=self.s3_bucket, Key=s3_key)

            # Register the image
            self.image_registry[image_id] = local_path
            self.image_order.append(image_id)

        except Exception as e:
            logger.error(f"Failed to save or upload image: {e}")

    def get_image_path(self, image_id: str) -> Optional[str]:
        """
        Get the local path for a specific image ID.
        Args:
            image_id: The unique identifier for the image.
        Returns:
            The local file path of the image, or None if not found.
        """
        return self.image_registry.get(image_id)

    def get_latest_image_path(self) -> Optional[str]:
        """
        Get the local path of the most recently uploaded image.
        Returns:
            The local file path of the latest image, or None if no images exist.
        """
        if not self.image_order:
            return None
        latest_image_id = self.image_order[-1]
        return self.image_registry.get(latest_image_id)

    def get_latest_image_id(self) -> Optional[str]:
        """
        Get the ID of the most recently uploaded image.
        Returns:
            The image ID of the latest image, or None if no images exist.
        """
        return self.image_order[-1] if self.image_order else None

    def get_image_wget_link(self, image_id: str) -> Optional[str]:
        """
        Get the wget link for a specific image ID.
        Args:
            image_id: The unique identifier for the image.
        Returns:
            The wget link for the image, or None if not found.
        """
        if image_id not in self.image_registry:
            return None
        filename = f"{image_id}.png"
        s3_key = f"{self.s3_prefix}/{filename}"
        return self.s3_manager.get_wget_link(self.s3_bucket, s3_key)

    def get_latest_image_wget_link(self) -> Optional[str]:
        """
        Get the wget link for the most recently uploaded image.
        Returns:
            The wget link for the latest image, or None if no images exist.
        """
        latest_image_id = self.get_latest_image_id()
        if latest_image_id is None:
            return None
        return self.get_image_wget_link(latest_image_id)

    def cleanup(self):
        """Removes the local temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
