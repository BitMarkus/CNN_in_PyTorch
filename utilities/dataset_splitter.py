# ===== Standard Library Imports =====
import shutil
import random
from pathlib import Path
from typing import List, Dict
# ===== Own Modules =====
from settings import setting

class ImageDatasetSplitter:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the image splitter.
    # Args:
    #   ratios (list): List of ratios for splitting (e.g., [0.5, 0.5] for 50/50 split)
    #   random_seed (int): Random seed for reproducibility. Defaults to 42.
    def __init__(self, ratios: List[float] = None, random_seed: int = 42) -> None:

        # Paths from settings
        self.input_folder = setting['pth_input']
        self.output_folder = setting['pth_output']

        self.ratios = ratios or [0.3]
        self.random_seed = random_seed

        self._validate_ratios()

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

    #############################################################################################################
    # METHODS

    # Validate that ratios are valid (sum <= 1.0, all positive).
    def _validate_ratios(self) -> None:
        if not self.ratios:
            raise ValueError("Ratios list cannot be empty")

        if any(ratio <= 0 for ratio in self.ratios):
            raise ValueError("All ratios must be positive")

        total = sum(self.ratios)
        if total > 1.0:
            raise ValueError(f"Sum of ratios ({total}) cannot exceed 1.0")

    # Get all image files from input folder without duplicates.
    # Returns:
    #   List[Path]: List of unique image file paths
    def _get_all_images(self) -> List[Path]:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        all_images = []
        seen_files = set()

        for ext in image_extensions:
            for pattern in [f"*{ext}", f"*{ext.upper()}"]:
                for img_path in self.input_folder.glob(pattern):
                    img_lower = img_path.name.lower()
                    if img_lower not in seen_files:
                        seen_files.add(img_lower)
                        all_images.append(img_path)

        return sorted(all_images, key=lambda x: x.name.lower())

    # Calculate number of images for each split based on ratios.
    # Args:
    #   total_images (int): Total number of images
    # Returns:
    #   List[int]: Number of images per split
    def _calculate_split_counts(self, total_images: int) -> List[int]:
        split_counts = []
        remaining_images = total_images

        for i, ratio in enumerate(self.ratios):
            if i == len(self.ratios) - 1 and sum(self.ratios) == 1.0:
                count = remaining_images
            else:
                count = int(round(total_images * ratio))
                remaining_images -= count
            split_counts.append(count)

        if remaining_images > 0:
            split_counts.append(remaining_images)

        total_split = sum(split_counts)
        if total_split != total_images:
            diff = total_images - total_split
            split_counts[-1] += diff

        return split_counts

    # Split images into multiple datasets based on specified ratios.
    # Returns:
    #   dict: Split statistics
    def split_images(self) -> Dict:
        all_images = self._get_all_images()

        if not all_images:
            print("No image files found in input folder!")
            return {}

        total_images = len(all_images)
        print(f"Found {total_images} unique images in input folder")

        random.seed(self.random_seed)
        shuffled_images = all_images.copy()
        random.shuffle(shuffled_images)

        split_counts = self._calculate_split_counts(total_images)
        num_splits = len(split_counts)

        print(f"Splitting into {num_splits} datasets with counts: {split_counts}")

        splits = []
        start_idx = 0

        for i, count in enumerate(split_counts):
            end_idx = start_idx + count
            split = shuffled_images[start_idx:end_idx]
            splits.append(split)
            start_idx = end_idx

        # Clean up old dataset folders
        if self.output_folder.exists():
            for item in self.output_folder.glob("dataset_*"):
                if item.is_dir():
                    shutil.rmtree(item)

        statistics = {}

        for i, (split_images, count) in enumerate(zip(splits, split_counts), 1):
            actual_ratio = round(count / total_images, 2)

            folder_name = f"dataset_{i}_{actual_ratio:.2f}"
            dataset_folder = self.output_folder / folder_name
            dataset_folder.mkdir(exist_ok=True)

            for img_path in split_images:
                shutil.copy2(img_path, dataset_folder / img_path.name)

            if i <= len(self.ratios):
                target_ratio = self.ratios[i-1]
            else:
                target_ratio = 1.0 - sum(self.ratios)

            statistics[f"dataset_{i}"] = {
                'folder': folder_name,
                'target_ratio': round(target_ratio, 2),
                'actual_ratio': actual_ratio,
                'target_count': count,
                'actual_count': len(split_images),
                'image_list': [img.name for img in split_images]
            }

            print(f"Created {folder_name}: {len(split_images)} images ({actual_ratio:.0%})")

        return statistics

    # Print a detailed summary of the split operation.
    # Args:
    #   statistics (dict): Split statistics
    def print_split_summary(self, statistics: Dict) -> None:
        print("\n" + "="*60)
        print("SPLIT SUMMARY")
        print("="*60)

        print(f"\nInput folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Ratios: {[round(r, 2) for r in self.ratios]}")
        print(f"Random seed: {self.random_seed}")

        print("\nDataset details:")
        print("-"*60)
        print(f"{'Dataset':<20} {'Target Ratio':<15} {'Actual Ratio':<15} {'Images':<10}")
        print("-"*60)

        total_images = 0
        for dataset_name, stats in statistics.items():
            total_images += stats['actual_count']
            print(f"{stats['folder']:<20} {stats['target_ratio']:<15.2f} {stats['actual_ratio']:<15.2f} {stats['actual_count']:<10}")

        print("-"*60)
        print(f"{'TOTAL':<20} {'1.00':<15} {'1.00':<15} {total_images:<10}")

        print("\nFolder structure:")
        print(f"{self.output_folder}/")
        for i, (dataset_name, stats) in enumerate(statistics.items()):
            if i == len(statistics) - 1:
                print(f"  └── {stats['folder']}/ ({stats['actual_count']} images)")
            else:
                print(f"  ├── {stats['folder']}/ ({stats['actual_count']} images)")

    # Verify that all images were properly split and no duplicates exist.
    # Args:
    #   statistics (dict): Split statistics
    def verify_split(self, statistics: Dict) -> None:
        print("\n" + "="*60)
        print("VERIFICATION")
        print("="*60)

        all_images = []
        duplicates_found = False

        for dataset_name, stats in statistics.items():
            for img_name in stats['image_list']:
                if img_name in all_images:
                    duplicates_found = True
                    print(f"  Duplicate: {img_name}")
                all_images.append(img_name)

        if duplicates_found:
            print("✗ Found duplicate images across datasets")
        else:
            print("✓ No duplicates found across datasets")

        input_images = {img.name for img in self._get_all_images()}
        split_images = set(all_images)

        if len(input_images) != len(split_images):
            print(f"✗ Count mismatch: Input has {len(input_images)} unique images, splits have {len(split_images)} images")

        if input_images == split_images:
            print("✓ All input images are accounted for in the splits")
        else:
            missing = input_images - split_images
            extra = split_images - input_images

            if missing:
                print(f"✗ Missing {len(missing)} images from splits")
            if extra:
                print(f"✗ Found {len(extra)} extra images in splits")

    #############################################################################################################
    # CALL

    # Run the dataset splitter.
    # Returns:
    #   dict: Split statistics
    def __call__(self) -> Dict:
        return self.split_images()