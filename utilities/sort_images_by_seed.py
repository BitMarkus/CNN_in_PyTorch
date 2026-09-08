# ===== Standard Library Imports =====
import shutil
import re
from pathlib import Path
from typing import Dict, Optional
# ===== Own Modules =====
from settings import setting

class ImageOrganizerBySeed:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the image organizer by seed.
    # Args:
    #   verbose (bool): Print progress messages. Defaults to True.
    def __init__(self, verbose: bool = True) -> None:

        # Paths from settings
        self.input_folder = setting['pth_input']
        self.output_folder = setting['pth_output']

        self.verbose = verbose

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

    #############################################################################################################
    # METHODS

    # Extract seed from filename.
    # Args:
    #   filename (Path or str): The filename to process
    # Returns:
    #   Optional[str]: Seed number if found, None otherwise
    def extract_seed(self, filename) -> Optional[str]:
        if isinstance(filename, Path):
            filename_str = filename.stem
        else:
            filename_str = str(Path(filename).stem)

        match = re.match(r"s(\d+)_\d+_fib_morph", filename_str)
        if match:
            return match.group(1)

        return None

    # Get statistics about available images before organizing.
    # Returns:
    #   Dict[str, int]: Seed -> image count
    def get_statistics(self) -> Dict[str, int]:
        image_files = list(self.input_folder.glob("*.png"))
        seed_stats = {}

        for img_path in image_files:
            seed = self.extract_seed(img_path)
            if seed is not None:
                seed_stats[seed] = seed_stats.get(seed, 0) + 1

        return seed_stats

    # Organize all images into folders by seed number.
    # Returns:
    #   Dict[str, int]: Seed -> image count
    def organize_images_by_seed(self) -> Dict[str, int]:
        image_files = list(self.input_folder.glob("*.png"))

        if not image_files:
            print("No PNG files found in input folder!")
            return {}

        seed_folders = {}
        processed_count = 0

        print("Organizing images by seed...")

        for img_path in image_files:
            seed = self.extract_seed(img_path)

            if seed is not None:
                seed_folder = self.output_folder / f"seed_{seed}"
                seed_folder.mkdir(exist_ok=True)

                shutil.copy2(img_path, seed_folder / img_path.name)

                seed_folders[seed] = seed_folders.get(seed, 0) + 1
                processed_count += 1
            else:
                if self.verbose:
                    print(f"Could not extract seed from: {img_path.name}")

        print(f"\nOrganization complete!")
        print(f"Processed {processed_count} images")
        print(f"Created {len(seed_folders)} seed folders:")

        for seed, count in seed_folders.items():
            print(f"  seed_{seed}: {count} images")

        return seed_folders

    #############################################################################################################
    # CALL

    # Run the image organizer by seed.
    # Returns:
    #   Dict[str, int]: Seed -> image count
    def __call__(self) -> Dict[str, int]:
        return self.organize_images_by_seed()