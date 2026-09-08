# ===== Standard Library Imports =====
import shutil
from pathlib import Path
from typing import List, Dict
# ===== Own Modules =====
from settings import setting

class ImageCollector:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the image collector.
    # Args:
    #   extensions (list, optional): List of image extensions to collect. Defaults to ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'].
    #   verbose (bool): Print progress messages. Defaults to True.
    def __init__(
        self,
        extensions: List[str] = None,
        verbose: bool = True
    ) -> None:
        
        # Paths from settings
        self.input_folder = setting['pth_input']
        self.output_folder = setting['pth_output']

        self.extensions = extensions or ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif']
        self.verbose = verbose

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

    #############################################################################################################
    # METHODS

    # Get all image files from input folder with supported extensions.
    def get_images(self) -> List[Path]:
        image_set = set()

        for ext in self.extensions:
            for file_path in self.input_folder.rglob(f"*{ext}"):
                image_set.add(file_path)
            for file_path in self.input_folder.rglob(f"*{ext.upper()}"):
                image_set.add(file_path)
            for file_path in self.input_folder.rglob(f"*{ext.capitalize()}"):
                image_set.add(file_path)

        return list(image_set)

    # Get statistics about available images before collecting.
    def get_input_statistics(self) -> Dict[str, int]:
        image_files = self.get_images()
        folder_stats = {}

        for img_path in image_files:
            folder_name = img_path.parent.name
            folder_stats[folder_name] = folder_stats.get(folder_name, 0) + 1

        return folder_stats

    # Copy all images from all subfolders into one output folder.
    def collect_all_images(self) -> Dict:
        image_files = self.get_images()

        if not image_files:
            print(f"No supported image files found in {self.input_folder}!")
            print(f"Supported extensions: {self.extensions}")
            return {}

        processed_count = 0
        skipped_count = 0

        print(f"Collecting all images from {self.input_folder} into {self.output_folder}...")
        print(f"Looking for: {self.extensions}")
        print(f"Found {len(image_files)} images")
        print(f"Renaming format: folder_name_original_name")
        print("-" * 60)

        for img_path in image_files:
            parent_folder = img_path.parent.name
            original_stem = img_path.stem
            suffix = img_path.suffix

            new_filename = f"{parent_folder}_{original_stem}{suffix}"
            destination = self.output_folder / new_filename

            counter = 1
            while destination.exists():
                new_filename = f"{parent_folder}_{original_stem}_{counter}{suffix}"
                destination = self.output_folder / new_filename
                counter += 1

            try:
                shutil.copy2(img_path, destination)
                processed_count += 1
                if self.verbose:
                    print(f"✓ Copied: {img_path.name} -> {destination.name}")
            except Exception as e:
                print(f"✗ Error copying {img_path}: {e}")
                skipped_count += 1

        print(f"\n{'='*50}")
        print(f"COLLECTION COMPLETE!")
        print(f"Successfully copied: {processed_count} images")
        if skipped_count > 0:
            print(f"Skipped/Failed: {skipped_count} images")
        print(f"Total images in output folder: {len(list(self.output_folder.glob('*')))}")
        print(f"{'='*50}")

        return {
            'processed': processed_count,
            'skipped': skipped_count,
            'total_in_output': len(list(self.output_folder.glob('*')))
        }

    #############################################################################################################
    # CALL

    def __call__(self) -> Dict:
        return self.collect_all_images()