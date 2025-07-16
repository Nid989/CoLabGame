import os
import logging
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

logger = logging.getLogger(__name__)


class S3Manager:
    """A generic handler for AWS S3 operations."""

    def __init__(self, aws_access_key_id: str, aws_secret_access_key: str, region_name: str):
        """
        Initializes the S3 client with provided credentials.
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("Required package not installed. Run: pip install boto3")

        self.s3_client = boto3.client("s3", region_name=region_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        self.region_name = region_name
        logger.info("S3Manager initialized successfully.")

    def upload_file(self, Filename: str, Bucket: str, Key: str):
        """Uploads a single file to an S3 key."""
        try:
            self.s3_client.upload_file(Filename=Filename, Bucket=Bucket, Key=Key)
            logger.info(f"Uploaded {Filename} to s3://{Bucket}/{Key}")
        except Exception as e:
            logger.error(f"Failed to upload {Filename}: {e}")

    def upload_directory(self, local_dir: str, bucket_name: str, s3_prefix: str = "") -> Dict[str, str]:
        """
        Uploads the contents of a local directory to an S3 prefix.

        Returns:
            A dictionary mapping local filenames to their S3 object keys.
        """
        if s3_prefix and not s3_prefix.endswith("/"):
            s3_prefix += "/"

        s3_object_mapping = {}
        logger.info(f"Starting upload of directory '{local_dir}' to s3://{bucket_name}/{s3_prefix}")
        for filename in os.listdir(local_dir):
            local_path = os.path.join(local_dir, filename)
            if os.path.isfile(local_path):
                s3_key = f"{s3_prefix}{filename}"
                try:
                    self.upload_file(Filename=local_path, Bucket=bucket_name, Key=s3_key)
                    s3_object_mapping[filename] = s3_key
                except Exception as e:
                    logger.error(f"Failed to upload {filename}: {e}")

        logger.info("Finished uploading directory.")
        return s3_object_mapping

    def download_directory(self, bucket_name: str, s3_prefix: str, local_dir: str) -> Dict[str, str]:
        """
        Downloads all objects with a given S3 prefix to a local directory.

        Args:
            bucket_name: The S3 bucket name.
            s3_prefix: The S3 prefix to download (e.g., 'game_sessions/game_id_uuid/').
            local_dir: The local directory to save files to.

        Returns:
            A dictionary mapping S3 keys to local file paths.
        """
        os.makedirs(local_dir, exist_ok=True)

        downloaded_files = {}
        logger.info(f"Starting download of s3://{bucket_name}/{s3_prefix} to {local_dir}")

        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        s3_key = obj["Key"]

                        # Skip if it's the prefix itself (directory marker)
                        if s3_key == s3_prefix:
                            continue

                        # Extract filename from S3 key
                        filename = os.path.basename(s3_key)
                        local_path = os.path.join(local_dir, filename)

                        try:
                            self.s3_client.download_file(Bucket=bucket_name, Key=s3_key, Filename=local_path)
                            downloaded_files[s3_key] = local_path
                            logger.info(f"Downloaded {s3_key} to {local_path}")
                        except Exception as e:
                            logger.error(f"Failed to download {s3_key}: {e}")

            logger.info(f"Finished downloading directory. Downloaded {len(downloaded_files)} files.")
            return downloaded_files

        except Exception as e:
            logger.error(f"Failed to download directory {s3_prefix}: {e}")
            return {}

    def get_object_content(self, bucket_name: str, s3_key: str) -> str:
        """Retrieves the content of an S3 object as a string."""
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            return response["Body"].read().decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to get object {s3_key}: {e}")
            return None

    def get_wget_link(self, bucket_name: str, s3_key: str) -> str:
        """Generates a wget-suitable download link for a single S3 object."""
        return f"https://{bucket_name}.s3.{self.region_name}.amazonaws.com/{s3_key}"

    def delete_file(self, bucket_name: str, s3_key: str):
        """Deletes a specific file (object) from an S3 bucket."""
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
            logger.info(f"Successfully deleted s3://{bucket_name}/{s3_key}")
        except Exception as e:
            logger.error(f"Failed to delete {s3_key}: {e}")

    def empty_bucket(self, bucket_name: str):
        """
        Deletes ALL objects within an S3 bucket. This is a destructive operation.
        """
        logger.warning(f"WARNING: This will permanently delete ALL objects in the bucket '{bucket_name}'.")
        confirm = input(f"To confirm, please type the bucket name '{bucket_name}': ")
        if confirm != bucket_name:
            logger.info("Confirmation failed. Aborting bucket deletion.")
            return

        logger.info("Proceeding with bucket emptying...")
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket_name)

            objects_to_delete = {"Objects": []}
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        objects_to_delete["Objects"].append({"Key": obj["Key"]})
                        if len(objects_to_delete["Objects"]) >= 1000:
                            self.s3_client.delete_objects(Bucket=bucket_name, Delete=objects_to_delete)
                            objects_to_delete = {"Objects": []}

            if len(objects_to_delete["Objects"]) > 0:
                self.s3_client.delete_objects(Bucket=bucket_name, Delete=objects_to_delete)

            logger.info(f"Successfully emptied bucket '{bucket_name}'.")
        except Exception as e:
            logger.error(f"An error occurred while emptying bucket {bucket_name}: {e}")


if __name__ == "__main__":
    s3_manager = S3Manager(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    s3_manager.empty_bucket(bucket_name="creang-test-bucket")
