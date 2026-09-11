# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Standard Library Imports =====
import shutil
import json
from pathlib import Path
from typing import List, Set, Optional
from datetime import datetime
# ===== Own Modules =====
from settings import setting

class DatasetMerger:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the dataset merger.
    # Args:
    #   recursive_depth (int, optional): Maximum depth to scan. None = unlimited. Defaults to None.
    #   copy_duplicates (bool, optional): True = copy with rename, False = skip duplicates. Defaults to False.
    #   verbose (bool, optional): Print progress messages. Defaults to True.
    def __init__(
        self,
        recursive_depth: Optional[int] = None,
        copy_duplicates: bool = False,
        verbose: bool = True
    ) -> None:
        
        # Paths from settings
        self.input_folder = setting['pth_input']
        self.output_folder = setting['pth_output']

        self.recursive_depth = recursive_depth
        self.copy_duplicates = copy_duplicates
        self.verbose = verbose

        # Supported image extensions
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self.stats = {
            'total_images_found': 0,
            'total_images_copied': 0,
            'duplicates_skipped': 0,
            'duplicates_renamed': 0,
            'folders_scanned': 0,
            'files_skipped_non_image': 0,
            'collection_log': {}
        }

        # Track used filenames to handle duplicates
        self.used_filenames: Set[str] = set()

    #############################################################################################################
    # METHODS

    # Recursively get all image files from folder and all subfolders.
    def get_all_image_files_recursive(self, folder_path: Path, current_depth: int = 0, max_depth: Optional[int] = None) -> List[Path]:
        if max_depth is not None and current_depth > max_depth:
            return []

        image_files = []

        if not folder_path.exists():
            return []

        try:
            for item in folder_path.iterdir():
                if item.is_dir():
                    subfolder_images = self.get_all_image_files_recursive(
                        item, current_depth + 1, max_depth
                    )
                    image_files.extend(subfolder_images)
                    self.stats['folders_scanned'] += 1

                elif item.is_file():
                    if item.suffix.lower() in self.image_extensions:
                        image_files.append(item)
                    else:
                        self.stats['files_skipped_non_image'] += 1

        except PermissionError:
            if self.verbose:
                print(f"  WARNING: Permission denied accessing: {folder_path}")
        except Exception as e:
            if self.verbose:
                print(f"  WARNING: Error accessing {folder_path}: {e}")

        return image_files

    # Generate a unique filename to avoid collisions in the flat output folder.
    def generate_unique_filename(self, original_path: Path, relative_path: Path) -> str:
        original_name = original_path.name
        name_parts = original_name.rsplit('.', 1)
        base_name = name_parts[0]
        extension = f".{name_parts[1]}" if len(name_parts) > 1 else ""

        path_parts = list(relative_path.parent.parts) if relative_path.parent != Path('.') else []

        if path_parts:
            prefix = '_'.join(path_parts[-2:])
            new_name = f"{prefix}_{original_name}"
        else:
            counter = 1
            while True:
                new_name = f"{base_name}_{counter}{extension}"
                if new_name not in self.used_filenames:
                    break
                counter += 1

        return new_name

    # Recursively scan input folder and copy all images to output folder.
    def collect_and_copy_images(self) -> dict:
        print(f"{'='*60}")
        print("DATASET MERGER - FLATTEN NESTED STRUCTURE")
        print(f"{'='*60}")
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        print(f"Recursive depth: {'Unlimited' if self.recursive_depth is None else self.recursive_depth}")
        print(f"Copy duplicates: {self.copy_duplicates} {'(will rename)' if self.copy_duplicates else '(will skip)'}")
        print(f"{'='*60}")

        if not self.input_folder.exists():
            print(f"ERROR: Input folder does not exist: {self.input_folder}")
            return self.stats

        print("\nScanning for image files (this may take a moment)...")

        all_images = self.get_all_image_files_recursive(
            self.input_folder,
            max_depth=self.recursive_depth
        )

        self.stats['total_images_found'] = len(all_images)
        print(f"\nFound {self.stats['total_images_found']} image files in {self.stats['folders_scanned']} folders")

        if not all_images:
            print("No image files found. Exiting.")
            return self.stats

        print("\nCopying images to output folder...")

        for idx, img_path in enumerate(all_images, 1):
            try:
                relative_path = img_path.relative_to(self.input_folder)
            except ValueError:
                relative_path = Path(img_path.name)

            dest_filename = img_path.name

            if dest_filename in self.used_filenames:
                if self.copy_duplicates:
                    dest_filename = self.generate_unique_filename(img_path, relative_path)
                    self.stats['duplicates_renamed'] += 1
                    if self.verbose:
                        print(f"  NOTICE: Duplicate '{img_path.name}' from '{relative_path}' → renamed to '{dest_filename}'")
                else:
                    self.stats['duplicates_skipped'] += 1
                    if self.verbose:
                        print(f"  NOTICE: Skipping duplicate '{img_path.name}' from '{relative_path}'")
                    self.stats['collection_log'][str(relative_path)] = {
                        'source': str(img_path),
                        'destination': None,
                        'original_filename': img_path.name,
                        'final_filename': None,
                        'was_renamed': False,
                        'was_skipped': True,
                        'skip_reason': 'duplicate_filename'
                    }
                    continue

            self.used_filenames.add(dest_filename)
            dest_path = self.output_folder / dest_filename

            try:
                shutil.copy2(img_path, dest_path)
                self.stats['total_images_copied'] += 1

                self.stats['collection_log'][str(relative_path)] = {
                    'source': str(img_path),
                    'destination': str(dest_path),
                    'original_filename': img_path.name,
                    'final_filename': dest_filename,
                    'was_renamed': dest_filename != img_path.name,
                    'was_skipped': False
                }

                if self.verbose and idx % 100 == 0:
                    print(f"  Progress: {idx}/{self.stats['total_images_found']} files processed")

            except Exception as e:
                print(f"  ERROR copying {img_path}: {e}")

        print(f"\nCopy complete: {self.stats['total_images_copied']} images copied")

        if self.stats['duplicates_skipped'] > 0 or self.stats['duplicates_renamed'] > 0:
            print(f"\n{'!'*60}")
            print("DUPLICATE SUMMARY:")
            if self.copy_duplicates:
                print(f"  - {self.stats['duplicates_renamed']} duplicate(s) were renamed and copied")
            else:
                print(f"  - {self.stats['duplicates_skipped']} duplicate(s) were skipped (not copied)")
                print(f"  - Set copy_duplicates = True to copy and rename duplicates instead")
            print(f"{'!'*60}")

        self._save_collection_log()
        self._print_summary()

        return self.stats

    # Save the collection log to a JSON file.
    def _save_collection_log(self) -> None:
        log_file = self.output_folder / "collection_log.json"

        log_data = {
            'settings': {
                'input_folder': str(self.input_folder),
                'output_folder': str(self.output_folder),
                'recursive_depth': self.recursive_depth,
                'copy_duplicates': self.copy_duplicates,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'statistics': {
                'total_images_found': self.stats['total_images_found'],
                'total_images_copied': self.stats['total_images_copied'],
                'duplicates_skipped': self.stats['duplicates_skipped'],
                'duplicates_renamed': self.stats['duplicates_renamed'],
                'folders_scanned': self.stats['folders_scanned'],
                'files_skipped_non_image': self.stats['files_skipped_non_image']
            },
            'file_mappings': self.stats['collection_log']
        }

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"Collection log saved to: {log_file}")

    # Print a summary of the collection results.
    def _print_summary(self) -> None:
        print(f"\n{'='*60}")
        print("MERGER COMPLETE - SUMMARY")
        print(f"{'='*60}")

        print(f"Folders scanned: {self.stats['folders_scanned']}")
        print(f"Image files found: {self.stats['total_images_found']}")
        print(f"Image files copied: {self.stats['total_images_copied']}")
        print(f"Non-image files skipped: {self.stats['files_skipped_non_image']}")

        if self.copy_duplicates:
            print(f"Duplicates renamed & copied: {self.stats['duplicates_renamed']}")
        else:
            print(f"Duplicates skipped: {self.stats['duplicates_skipped']}")
            if self.stats['duplicates_skipped'] > 0:
                print(f"  (Set copy_duplicates = True to copy and rename duplicates instead)")

        if self.stats['total_images_found'] > self.stats['total_images_copied'] + self.stats['duplicates_skipped']:
            diff = self.stats['total_images_found'] - (self.stats['total_images_copied'] + self.stats['duplicates_skipped'])
            print(f"\nWARNING: {diff} files were not copied (check permissions or errors)")

    #############################################################################################################
    # CALL

    def __call__(self) -> dict:
        return self.collect_and_copy_images()