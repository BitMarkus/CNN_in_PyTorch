# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Standard Library Imports =====
import shutil
from itertools import product
from pathlib import Path
# ===== Own Modules =====
from settings import setting

class DatasetGenerator():

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the dataset generator.
    # Supports three data source modes:
    #   - "mixed": Single input folder with both synthetic and real images (backward compatible)
    #   - "synthetic_only": Train on synthetic images, validate/test on real images
    #   - "real_only": Use only real images for everything
    # Args:
    #   mode (str): "acv" for automatic cross-validation, "gen" for standalone generation
    def __init__(self, mode: str) -> None:

        # Passed parameters
        self.mode = mode

        # Settings parameters
        self.input_dir = setting['pth_ds_gen_input_mixed']
        self.input_dir_synthetic = setting.get('pth_ds_gen_input_synthetic', None)
        self.input_dir_real = setting.get('pth_ds_gen_input_real', None)
        self.training_data_source = setting.get('train_data_source', 'mixed')
        self.output_dir = setting['pth_ds_gen_output']
        self.wt_lines = setting['wt_lines']
        self.ko_lines = setting['ko_lines']
        self.train_dir = setting['pth_train']
        self.test_dir = setting['pth_test']
        self.acv_results_dir = setting['pth_output'] / "cross_validation"

    #############################################################################################################
    # METHODS

    # Create train/test folders with WT/KO subdirectories.
    # Args:
    #   dataset_path (Path): Path to the dataset folder
    # Returns:
    #   tuple: (train_dir, test_dir) or False on failure
    def _create_folder_structure(self, dataset_path: Path):
        if self.mode == "acv":
            self.acv_results_dir.mkdir(exist_ok=True)
            train_dir = self.train_dir
            test_dir = self.test_dir
        elif self.mode == "gen":
            train_dir = dataset_path / "train"
            test_dir = dataset_path / "test"
        else:
            return False

        (train_dir / "WT").mkdir(parents=True, exist_ok=True)
        (train_dir / "KO").mkdir(parents=True, exist_ok=True)
        (test_dir / "WT").mkdir(parents=True, exist_ok=True)
        (test_dir / "KO").mkdir(parents=True, exist_ok=True)

        return train_dir, test_dir

    # Get the correct source directory based on data source type.
    # Args:
    #   is_training_cell_line (bool): True if this cell line is for training
    #   cell_line (str): Name of the cell line
    # Returns:
    #   tuple: (source_dir_for_copying, source_dir_for_counting)
    def _get_source_dirs(self, is_training_cell_line: bool, cell_line: str) -> tuple:
        if self.training_data_source == 'mixed' or self.input_dir_real is None:
            return self.input_dir / cell_line, self.input_dir / cell_line

        elif self.training_data_source == 'synthetic_only':
            if is_training_cell_line:
                return self.input_dir_synthetic / cell_line, self.input_dir_synthetic / cell_line
            else:
                return self.input_dir_real / cell_line, self.input_dir_real / cell_line

        elif self.training_data_source == 'real_only':
            return self.input_dir_real / cell_line, self.input_dir_real / cell_line

        else:
            return self.input_dir / cell_line, self.input_dir / cell_line

    # Copy images from source directory to destination and update counts.
    # Args:
    #   src_dir (Path): Source directory
    #   dest_dir (Path): Destination directory
    #   counts_dict (dict): Dictionary to update with image count
    #   key (str): Key for the counts dictionary
    def _copy_images(self, src_dir: Path, dest_dir: Path, counts_dict: dict, key: str) -> None:
        counts_dict[key] = len(list(src_dir.iterdir()))
        for img in src_dir.iterdir():
            if img.is_file():
                shutil.copy2(img, dest_dir / img.name)

    # Generate and save dataset metadata to a text file.
    # Args:
    #   dataset_path (Path): Path to the dataset folder
    #   dataset_idx (int): Index of the dataset
    #   train_counts (dict): Training image counts per cell line
    #   test_counts (dict): Test image counts per cell line
    #   test_wt (str): WT cell line used for testing
    #   test_ko (str): KO cell line used for testing
    #   data_source_info (str): Additional info about data source mode
    def _generate_metadata(
        self,
        dataset_path: Path,
        dataset_idx: int,
        train_counts: dict,
        test_counts: dict,
        test_wt: str,
        test_ko: str,
        data_source_info: str = ""
    ) -> None:
        metadata_lines = [
            f"Dataset {dataset_idx} Configuration:",
            f"Data Source Mode: {self.training_data_source}",
            data_source_info,
            "",
            "=== TRAINING DATA ===",
            "WT lines:"
        ]
        metadata_lines.extend(f"- {line}: {count} images" for line, count in train_counts["WT"].items())
        metadata_lines.append(f"Total WT training images: {sum(train_counts['WT'].values())}\n")
        metadata_lines.append("KO lines:")
        metadata_lines.extend(f"- {line}: {count} images" for line, count in train_counts["KO"].items())
        metadata_lines.append(f"Total KO training images: {sum(train_counts['KO'].values())}\n")
        metadata_lines.append("=== TESTING DATA ===")
        metadata_lines.append(f"WT test line: {test_wt} ({test_counts['WT'].get(test_wt, 0)} images)")
        metadata_lines.append(f"KO test line: {test_ko} ({test_counts['KO'].get(test_ko, 0)} images)")

        with open(dataset_path / f"dataset_{dataset_idx}_info.txt", 'w') as f:
            f.write('\n'.join(metadata_lines))

    # Generate a single dataset for a specific (test_wt, test_ko) pair.
    # Args:
    #   test_wt (str): WT cell line to hold out for testing
    #   test_ko (str): KO cell line to hold out for testing
    #   dataset_idx (int): Index of the dataset
    # Returns:
    #   Path or False: Path to the generated dataset, or False on failure
    def generate_dataset(self, test_wt: str, test_ko: str, dataset_idx: int):
        if self.mode == "acv":
            dataset_path = self.acv_results_dir / f"dataset_{dataset_idx}"
        elif self.mode == "gen":
            dataset_path = self.output_dir / f"dataset_{dataset_idx}"
        else:
            return False

        dataset_path.mkdir(exist_ok=True)
        train_dir, test_dir = self._create_folder_structure(dataset_path)

        train_counts = {"WT": {}, "KO": {}}
        test_counts = {"WT": {}, "KO": {}}

        data_source_info = ""
        if self.training_data_source == 'synthetic_only':
            data_source_info = "TRAINING: Synthetic images only\nVALIDATION/TESTING: Real images only"

        # Process WT lines
        for wt in self.wt_lines:
            is_training_cell_line = (wt != test_wt)
            src_dir, count_src_dir = self._get_source_dirs(is_training_cell_line, wt)

            if not src_dir.exists():
                print(f"WARNING: Source directory {src_dir} does not exist!")
                continue

            dest_dir = (test_dir if wt == test_wt else train_dir) / "WT"
            dest_dir.mkdir(parents=True, exist_ok=True)

            if src_dir.exists():
                self._copy_images(src_dir, dest_dir,
                                 test_counts["WT"] if wt == test_wt else train_counts["WT"],
                                 wt)
            else:
                if wt == test_wt:
                    test_counts["WT"][wt] = 0
                else:
                    train_counts["WT"][wt] = 0

        # Process KO lines
        for ko in self.ko_lines:
            is_training_cell_line = (ko != test_ko)
            src_dir, count_src_dir = self._get_source_dirs(is_training_cell_line, ko)

            if not src_dir.exists():
                print(f"WARNING: Source directory {src_dir} does not exist!")
                continue

            dest_dir = (test_dir if ko == test_ko else train_dir) / "KO"
            dest_dir.mkdir(parents=True, exist_ok=True)

            if src_dir.exists():
                self._copy_images(src_dir, dest_dir,
                                 test_counts["KO"] if ko == test_ko else train_counts["KO"],
                                 ko)
            else:
                if ko == test_ko:
                    test_counts["KO"][ko] = 0
                else:
                    train_counts["KO"][ko] = 0

        self._generate_metadata(dataset_path, dataset_idx, train_counts, test_counts, test_wt, test_ko, data_source_info)

        print(f"Dataset {dataset_idx}: {self.training_data_source} mode")
        print(f"  Training images: WT={sum(train_counts['WT'].values())}, KO={sum(train_counts['KO'].values())}")
        print(f"  Test images: WT={sum(test_counts['WT'].values())}, KO={sum(test_counts['KO'].values())}")

        return dataset_path

    # Generate all dataset combinations (legacy method, prefer iterative approach).
    # Returns:
    #   bool: True if successful
    def generate_all_datasets(self) -> bool:
        for dataset_idx, (test_wt, test_ko) in enumerate(product(self.wt_lines, self.ko_lines), 1):
            self.generate_dataset(test_wt, test_ko, dataset_idx)
        return True

    # Delete temporary train and test images (preserves results/metadata).
    # Args:
    #   dataset_path (Path): Path to the dataset folder
    def cleanup(self, dataset_path: Path) -> None:
        dirs_to_remove = [
            dataset_path / "train",
            dataset_path / "test"
        ]
        for dir_path in dirs_to_remove:
            if dir_path.exists():
                shutil.rmtree(dir_path)

    # Return a list of all possible (test_wt, test_ko) configurations as dicts.
    # Example: [{"test_wt": "wt1", "test_ko": "ko1", "dataset_idx": 1}, ...]
    # Returns:
    #   list: List of configuration dictionaries
    def get_dataset_configs(self) -> list:
        return [
            {"test_wt": wt, "test_ko": ko, "dataset_idx": idx}
            for idx, (wt, ko) in enumerate(product(self.wt_lines, self.ko_lines), 1)
        ]