# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Own Modules =====
import functions as fn
from settings import setting
from . import (
    DatasetMerger,
    RandomDSRemover,
    ImageDatasetSplitter,
    DatasetSubtractor,
    ImageCollector,
    ImageOrganizerByFrame,
    ImageOrganizerBySeed
)

class Utilities:

    # Utilities Menu - Collection of helper scripts for dataset manipulation and analysis.

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the utilities menu.
    # Loads all settings from the configuration file.
    def __init__(self) -> None:

        # Dataset Merger settings
        self.merger_recursive_depth = setting.get("util_merger_recursive_depth", None)
        self.merger_copy_duplicates = setting.get("util_merger_copy_duplicates", False)
        self.merger_verbose = setting.get("util_merger_verbose", True)

        # Dataset Remover settings
        self.remover_target_per_folder = setting.get("util_remover_target_per_folder", 500)
        self.remover_seed = setting.get("util_remover_seed", 42)
        self.remover_verbose = setting.get("util_remover_verbose", True)

        # Dataset Splitter settings
        self.splitter_ratios = setting.get("util_splitter_ratios", [0.3])
        self.splitter_random_seed = setting.get("util_splitter_random_seed", 42)

        # Dataset Subtraction settings
        self.subtraction_dataset_a = setting.get("util_subtraction_dataset_a", "dataset_a")
        self.subtraction_dataset_b = setting.get("util_subtraction_dataset_b", "dataset_b")
        self.subtraction_result = setting.get("util_subtraction_result", "result_dataset")
        self.subtraction_verbose = setting.get("util_subtraction_verbose", True)

        # Merge Images from Folders settings
        self.merge_extensions = setting.get("util_merge_extensions", ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'])
        self.merge_verbose = setting.get("util_merge_verbose", True)

        # Sort Images by Frame settings
        self.sort_frame_format = setting.get("util_sort_frame_format", "flux")
        self.sort_frame_verbose = setting.get("util_sort_frame_verbose", True)

        # Sort Images by Seed settings
        self.sort_seed_verbose = setting.get("util_sort_seed_verbose", True)

    #############################################################################################################
    # METHODS

    # Display the utilities submenu and route to the selected utility.
    def menu(self) -> None:
        while True:
            print("\n:UTILITIES MENU:")
            print("1) Dataset Merger (Flatten nested folder structure)")
            print("2) Dataset Remover (Randomly reduce images per folder)")
            print("3) Dataset Splitter (Split images by ratios)")
            print("4) Dataset Subtraction (Subtract one dataset from another)")
            print("5) Merge Images from Folders (Collect with folder prefix)")
            print("6) Sort Images by Frame (Organize morphing series by frame)")
            print("7) Sort Images by Seed (Organize images by seed)")
            print("8) Back to Main Menu")

            choice = fn.input_int("Please choose: ")

            if choice == 1:
                self._run_dataset_merger()
            elif choice == 2:
                self._run_dataset_remover()
            elif choice == 3:
                self._run_dataset_splitter()
            elif choice == 4:
                self._run_dataset_subtraction()
            elif choice == 5:
                self._run_merge_images_from_folders()
            elif choice == 6:
                self._run_sort_images_by_frame()
            elif choice == 7:
                self._run_sort_images_by_seed()
            elif choice == 8:
                print("\nReturning to main menu...")
                break
            else:
                print("Not a valid option!")

    # Run the Dataset Merger utility.
    def _run_dataset_merger(self) -> None:
        print("\n:DATASET MERGER:")
        print("  Input: input/ (images in nested subfolders)")
        print("  Output: output/ (all images flattened)")
        print("  Note: This copies images, does NOT move them.")
        print(f"  Copy duplicates: {'Yes (with rename)' if self.merger_copy_duplicates else 'No (skip)'}")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            merger = DatasetMerger(
                recursive_depth=self.merger_recursive_depth,
                copy_duplicates=self.merger_copy_duplicates,
                verbose=self.merger_verbose
            )
            merger.collect_and_copy_images()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

    # Run the Dataset Remover utility.
    def _run_dataset_remover(self) -> None:
        print("\n:DATASET REMOVER:")
        print("  Input: input/ (images organized in folders)")
        print("  Output: output/ (reduced dataset)")
        print("  Note: Randomly selects images from each folder.")
        print(f"  Target per folder: {self.remover_target_per_folder}")
        print(f"  Random seed: {self.remover_seed}")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            remover = RandomDSRemover(
                target_per_folder=self.remover_target_per_folder,
                seed=self.remover_seed,
                verbose=self.remover_verbose
            )
            remover.process_dataset()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

    # Run the Dataset Splitter utility.
    def _run_dataset_splitter(self) -> None:
        print("\n:DATASET SPLITTER:")
        print("  Input: input/ (images to split)")
        print("  Output: output/ (split into dataset_X folders)")
        print(f"  Ratios: {self.splitter_ratios}")
        print(f"  Random seed: {self.splitter_random_seed}")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            splitter = ImageDatasetSplitter(
                ratios=self.splitter_ratios,
                random_seed=self.splitter_random_seed
            )
            stats = splitter.split_images()
            if stats:
                splitter.print_split_summary(stats)
                splitter.verify_split(stats)
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

    # Run the Dataset Subtraction utility.
    def _run_dataset_subtraction(self) -> None:
        print("\n:DATASET SUBTRACTION:")
        print("  Input: input/dataset_a/ and input/dataset_b/")
        print("  Output: output/result_dataset/")
        print("  Note: Subtracts common images between two datasets.")
        print("  You will be prompted to choose the subtraction direction.")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            subtractor = DatasetSubtractor(
                dataset_a_name=self.subtraction_dataset_a,
                dataset_b_name=self.subtraction_dataset_b,
                result_name=self.subtraction_result,
                verbose=self.subtraction_verbose
            )
            statistics = subtractor.subtract_datasets()
            subtractor.print_statistics(statistics)
            subtractor.verify_dataset_integrity(statistics)
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

    # Run the Merge Images from Folders utility.
    def _run_merge_images_from_folders(self) -> None:
        print("\n:MERGE IMAGES FROM FOLDERS:")
        print("  Input: input/ (images in subfolders)")
        print("  Output: output/ (images renamed as folder_original)")
        print("  Note: Adds parent folder name as prefix to each image.")
        print(f"  Extensions: {self.merge_extensions}")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            collector = ImageCollector(
                extensions=self.merge_extensions,
                verbose=self.merge_verbose
            )
            stats = collector.get_input_statistics()
            print(f"\nFound images by folder:")
            for folder, count in stats.items():
                print(f"  {folder}: {count} images")
            print(f"Total images found: {sum(stats.values())}")

            confirm2 = input(f"\nCopy {sum(stats.values())} images to output/? (yes/no): ").strip().lower()
            if confirm2 in ['yes', 'y']:
                collector.collect_all_images()
            else:
                print("Operation cancelled.")
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

    # Run the Sort Images by Frame utility.
    def _run_sort_images_by_frame(self) -> None:
        print("\n:SORT IMAGES BY FRAME:")
        print("  Input: input/ (morphing series images)")
        print("  Output: output/ (organized into frame_X folders)")
        print("  Note: Supports both Flux and StyleGAN naming formats.")
        print(f"  Format: {self.sort_frame_format}")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            organizer = ImageOrganizerByFrame(
                format=self.sort_frame_format,
                verbose=self.sort_frame_verbose
            )
            organizer.print_statistics()
            confirm2 = input("\nOrganize images? (yes/no): ").strip().lower()
            if confirm2 in ['yes', 'y']:
                organizer.organize_images_by_frame()
            else:
                print("Operation cancelled.")
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

    # Run the Sort Images by Seed utility.
    def _run_sort_images_by_seed(self) -> None:
        print("\n:SORT IMAGES BY SEED:")
        print("  Input: input/ (images with s{seed}_ pattern in filename)")
        print("  Output: output/ (organized into seed_X folders)")
        print("  Note: Extracts seed from filenames and groups by seed.")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            organizer = ImageOrganizerBySeed(
                verbose=self.sort_seed_verbose
            )
            stats = organizer.get_statistics()
            print("\nFound images by seed:")
            for seed, count in stats.items():
                print(f"  Seed {seed}: {count} images")
            print(f"Total images: {sum(stats.values())}")

            confirm2 = input("\nOrganize images? (yes/no): ").strip().lower()
            if confirm2 in ['yes', 'y']:
                organizer.organize_images_by_seed()
            else:
                print("Operation cancelled.")
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()