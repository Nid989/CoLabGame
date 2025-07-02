import os
import shutil
import tempfile
import atexit


from typing import Optional


class TemporaryImageManager:
    """Manages temporary image files that persist until the program termination.
    Uses tempfile for secure temporary file handling and caches files to avoid duplicates.
    """

    def __init__(self, game_id: Optional[str] = None):
        # Create a temporary directory that will be cleaned up at exit
        self.temp_dir = tempfile.mkdtemp()
        # Cache to store mapping of image content to file paths
        self.image_cache = {}
        # Control whether to preserve images during cleanup
        self.preserve_images = True
        # Store game_id for folder organization
        self.game_id = game_id
        atexit.register(self.cleanup)

    def save_image(self, image_binary: bytes) -> str:
        """Saves a binary image to a temporary file that persists until program exit.
        Returns cached path if the same image was saved before.
        Args:
            image_binary (bytes): Binary image data (PNG format)
        Returns:
            str: Path to saved image file
        """
        # Use image content as cache key
        image_hash = hash(image_binary)
        # Return cached path if image was saved before
        if image_hash in self.image_cache:
            return self.image_cache[image_hash]
        # Create new file if image hasn't been saved before
        tmp_file = tempfile.NamedTemporaryFile(suffix=".png", dir=self.temp_dir, delete=False)
        with tmp_file as f:
            f.write(image_binary)
            f.flush()
        # Cache the path
        self.image_cache[image_hash] = tmp_file.name
        return tmp_file.name

    def cleanup(self):
        """Removes the temporary directory and all files in it.
        If preserve_images is True, copies all images to a 'saved_images' subdirectory,
        optionally under a game_id folder if provided, before deletion.
        """
        print("inside cleanup")
        if os.path.exists(self.temp_dir):
            if self.preserve_images:
                current_dir = os.getcwd()
                # Base directory for saved images
                saved_images_dir = os.path.join(current_dir, "saved_images")
                print(saved_images_dir)
                # If game_id is provided, create a subdirectory with game_id
                if self.game_id:
                    saved_images_dir = os.path.join(saved_images_dir, self.game_id)
                os.makedirs(saved_images_dir, exist_ok=True)
                for image_hash, image_path in self.image_cache.items():
                    if os.path.exists(image_path):
                        filename = os.path.basename(image_path)
                        destination = os.path.join(saved_images_dir, filename)
                        shutil.copy2(image_path, destination)
            shutil.rmtree(self.temp_dir)
        self.image_cache.clear()
