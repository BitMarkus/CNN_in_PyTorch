# ===== Standard Library Imports =====
import shutil
import random
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
# ===== Own Modules =====
from settings import setting

class RandomDSRemover:
    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the random dataset remover.
    # Args:
    #   target_per_folder (int): Target number of images per folder. Defaults to 500.
    #   seed (int): Random seed for reproducibility. Defaults to 42.
    #   verbose (bool): Print progress messages. Defaults to True.
    def __init__(
        self,
        target_per_folder: int = 500,
        seed: int = 42,
        verbose: bool = True
    ) -> None:
        
        # Paths from settings
        self.input_folder = setting['pth_input']
        self.output_folder = setting['pth_output']

        self.target_per_folder = target_per_folder
        self.seed = seed
        self.verbose = verbose

        # Initialize random generator with seed for reproducibility
        random.seed(self.seed)

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self.stats = {
            'total_input_images': 0,
            'total_output_images': 0,
            'folders_processed': 0,
            'folders_skipped': 0,
            'folders_warning': 0,
            'selection_log': {}
        }

    #############################################################################################################
    # METHODS

    # Get all image files from a folder.
    def get_all_image_files(self, folder_path: Path) -> List[Path]:
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
        image_files = []

        for item in folder_path.iterdir():
            if item.is_file():
                suffix_lower = item.suffix.lower()
                if suffix_lower in image_extensions:
                    image_files.append(item)

        return sorted(image_files)

    # Reduce images in a single folder to target number.
    def reduce_folder_images(self, folder_path: Path, folder_name: str) -> Dict:
        all_images = self.get_all_image_files(folder_path)

        if not all_images:
            if self.verbose:
                print(f"  WARNING: No images found in folder: {folder_name}")
            return {
                'folder_name': folder_name,
                'total_images': 0,
                'selected_images': 0,
                'skipped': True,
                'warning': True,
                'warning_message': 'No images found',
                'selected_files': [],
                'excluded_files': []
            }

        total_count = len(all_images)

        if total_count < self.target_per_folder:
            self.stats['folders_warning'] += 1
            warning_msg = f"Insufficient images: {total_count} < {self.target_per_folder}"
            if self.verbose:
                print(f"  WARNING: {warning_msg}")

            output_subfolder = self.output_folder / folder_name
            output_subfolder.mkdir(parents=True, exist_ok=True)

            for img_path in all_images:
                shutil.copy2(img_path, output_subfolder / img_path.name)

            self.stats['total_input_images'] += total_count
            self.stats['total_output_images'] += total_count

            return {
                'folder_name': folder_name,
                'total_images': total_count,
                'selected_images': total_count,
                'skipped': True,
                'warning': True,
                'warning_message': warning_msg,
                'selected_files': [img.name for img in all_images],
                'excluded_files': []
            }

        if total_count == self.target_per_folder:
            if self.verbose:
                print(f"  Folder {folder_name}: {total_count} images (= target, copying all)")

            output_subfolder = self.output_folder / folder_name
            output_subfolder.mkdir(parents=True, exist_ok=True)

            for img_path in all_images:
                shutil.copy2(img_path, output_subfolder / img_path.name)

            self.stats['total_input_images'] += total_count
            self.stats['total_output_images'] += total_count
            self.stats['folders_skipped'] += 1

            return {
                'folder_name': folder_name,
                'total_images': total_count,
                'selected_images': total_count,
                'skipped': True,
                'warning': False,
                'warning_message': '',
                'selected_files': [img.name for img in all_images],
                'excluded_files': []
            }

        selected_images = random.sample(all_images, self.target_per_folder)
        excluded_images = [img for img in all_images if img not in selected_images]

        output_subfolder = self.output_folder / folder_name
        output_subfolder.mkdir(parents=True, exist_ok=True)

        for img_path in selected_images:
            shutil.copy2(img_path, output_subfolder / img_path.name)

        self.stats['total_input_images'] += total_count
        self.stats['total_output_images'] += len(selected_images)
        self.stats['folders_processed'] += 1

        if self.verbose:
            print(f"  Folder {folder_name}: {total_count} → {len(selected_images)} images "
                  f"(removed {total_count - len(selected_images)})")

        return {
            'folder_name': folder_name,
            'total_images': total_count,
            'selected_images': len(selected_images),
            'skipped': False,
            'warning': False,
            'warning_message': '',
            'selected_files': [img.name for img in selected_images],
            'excluded_files': [img.name for img in excluded_images]
        }

    # Process the entire dataset, reducing images in each folder.
    def process_dataset(self) -> Dict:
        print(f"{'='*60}")
        print("RANDOM DATASET REDUCER")
        print(f"{'='*60}")
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Target images per folder: {self.target_per_folder}")
        print(f"Random seed: {self.seed}")
        print(f"{'='*60}")

        if not self.input_folder.exists():
            print(f"ERROR: Input folder does not exist: {self.input_folder}")
            return self.stats

        input_items = list(self.input_folder.iterdir())
        has_subfolders = any(item.is_dir() for item in input_items)

        if has_subfolders:
            print("Detected subfolder structure. Processing each folder...")
            folders_to_process = [item for item in input_items if item.is_dir()]

            for folder_path in folders_to_process:
                folder_name = folder_path.name
                if self.verbose:
                    print(f"\nProcessing folder: {folder_name}")

                folder_result = self.reduce_folder_images(folder_path, folder_name)
                self.stats['selection_log'][folder_name] = folder_result
        else:
            print("No subfolders found. Processing as single folder...")
            folder_result = self.reduce_folder_images(self.input_folder, "all_images")
            self.stats['selection_log']["all_images"] = folder_result

        self._save_selection_log()
        self._print_summary()

        return self.stats

    # Save the selection log to a JSON file.
    def _save_selection_log(self) -> None:
        log_file = self.output_folder / "selection_log.json"

        log_data = {
            'settings': {
                'input_folder': str(self.input_folder),
                'output_folder': str(self.output_folder),
                'target_per_folder': self.target_per_folder,
                'random_seed': self.seed,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'statistics': {
                'total_input_images': self.stats['total_input_images'],
                'total_output_images': self.stats['total_output_images'],
                'folders_processed': self.stats['folders_processed'],
                'folders_skipped': self.stats['folders_skipped'],
                'folders_warning': self.stats['folders_warning']
            },
            'per_folder_selection': self.stats['selection_log']
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\nSelection log saved to: {log_file}")

    # Print a summary of the processing results.
    def _print_summary(self) -> None:
        print(f"\n{'='*60}")
        print("PROCESSING COMPLETE - SUMMARY")
        print(f"{'='*60}")

        if (self.stats['folders_processed'] == 0 and
            self.stats['folders_skipped'] == 0 and
            self.stats['folders_warning'] == 0):
            print("No folders were processed. Check input folder structure.")
            return

        print(f"Total folders processed: {self.stats['folders_processed']}")
        print(f"Total folders skipped (already <= target): {self.stats['folders_skipped']}")
        print(f"Total folders with warnings: {self.stats['folders_warning']}")
        print(f"Total input images: {self.stats['total_input_images']}")
        print(f"Total output images: {self.stats['total_output_images']}")

        if self.stats['total_input_images'] > 0:
            reduction_percent = ((self.stats['total_input_images'] - self.stats['total_output_images']) /
                               self.stats['total_input_images'] * 100)
            print(f"Overall reduction: {reduction_percent:.1f}%")

        print(f"\nPer-folder details:")
        for folder_name, folder_data in self.stats['selection_log'].items():
            if folder_data.get('warning', False):
                status = f"WARNING: {folder_data.get('warning_message', '')}"
            elif folder_data['skipped']:
                status = "SKIPPED (already <= target)"
            else:
                status = f"REDUCED ({folder_data['total_images']} → {folder_data['selected_images']})"
            print(f"  {folder_name}: {status}")

        if self.stats['folders_warning'] > 0:
            print(f"\n{'!'*60}")
            print(f"WARNING: {self.stats['folders_warning']} folder(s) have insufficient images!")
            print(f"Target was {self.target_per_folder} images per folder.")
            print(f"Check the summary.csv and selection_log.json for details.")
            print(f"{'!'*60}")

    #############################################################################################################
    # CALL

    def __call__(self) -> Dict:
        return self.process_dataset()