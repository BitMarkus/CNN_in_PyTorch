#####################
# Dataset generator #
#####################

import os
import shutil
from itertools import product
# Own modules
from settings import setting

class DatasetGenerator():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
        # Settings parameters
        self.input_dir = setting['pth_ds_gen_input']
        self.output_dir = setting['pth_ds_gen_output']
        self.wt_lines = setting['wt_lines']
        self.ko_lines = setting['ko_lines']

    #############################################################################################################
    # METHODS:

    # Create train/test/result folders and WT/KO subdirectories
    def _create_folder_structure(self, dataset_path):

        train_dir = os.path.join(dataset_path, "train")
        test_dir = os.path.join(dataset_path, "test")
        result_dir = os.path.join(dataset_path, "result")
        
        os.makedirs(os.path.join(train_dir, "WT"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "KO"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "WT"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "KO"), exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        
        return train_dir, test_dir, result_dir

    # Copy images from src_dir to dest_dir and update counts_dict
    def _copy_images(self, src_dir, dest_dir, counts_dict, key):

        counts_dict[key] = len(os.listdir(src_dir))
        for img in os.listdir(src_dir):
            src_path = os.path.join(src_dir, img)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, os.path.join(dest_dir, img))

    # Generate and save dataset metadata
    def _generate_metadata(self, dataset_path, dataset_idx, train_counts, test_counts, test_wt, test_ko):
        metadata_lines = [
            f"Dataset {dataset_idx} Configuration:\n",
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

        with open(os.path.join(dataset_path, f"dataset_{dataset_idx}_info.txt"), 'w') as f:
            f.write('\n'.join(metadata_lines))

    # Generate a single dataset for a specific (test_wt, test_ko) pair
    def generate_dataset(self, test_wt, test_ko, dataset_idx):

        dataset_path = os.path.join(self.output_dir, f"dataset_{dataset_idx}")
        os.makedirs(dataset_path, exist_ok=True)
        train_dir, test_dir, _ = self._create_folder_structure(dataset_path)

        # Initialize counters
        train_counts = {"WT": {}, "KO": {}}
        test_counts = {"WT": {}, "KO": {}}

        # Process WT lines
        for wt in self.wt_lines:
            src_dir = os.path.join(self.input_dir, wt)
            dest_dir = os.path.join(test_dir if wt == test_wt else train_dir, "WT")
            self._copy_images(src_dir, dest_dir, 
                             test_counts["WT"] if wt == test_wt else train_counts["WT"], 
                             wt)

        # Process KO lines
        for ko in self.ko_lines:
            src_dir = os.path.join(self.input_dir, ko)
            dest_dir = os.path.join(test_dir if ko == test_ko else train_dir, "KO")
            self._copy_images(src_dir, dest_dir, 
                             test_counts["KO"] if ko == test_ko else train_counts["KO"], 
                             ko)

        self._generate_metadata(dataset_path, dataset_idx, train_counts, test_counts, test_wt, test_ko)

        return dataset_path
    
    # Generate all dataset combinations (legacy method, prefer iterative approach)
    def generate_all_datasets(self):

        for dataset_idx, (test_wt, test_ko) in enumerate(product(self.wt_lines, self.ko_lines), 1):
            self.generate_dataset(test_wt, test_ko, dataset_idx)

        return True

    # Delete training and test images (preserves results/metadata)
    def cleanup(self, dataset_path):

        dirs_to_remove = [
            os.path.join(dataset_path, "train"),
            os.path.join(dataset_path, "test")
        ]
        for dir_path in dirs_to_remove:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

    # Returns a list of all possible (test_wt, test_ko) configurations as dicts
    # Example output: [{"test_wt": "wt1", "test_ko": "ko1", "dataset_idx": 1}, {"test_wt": "wt1", "test_ko": "ko2", "dataset_idx": 2},...]
    def get_dataset_configs(self):
        return [
            {"test_wt": wt, "test_ko": ko, "dataset_idx": idx}
            for idx, (wt, ko) in enumerate(product(self.wt_lines, self.ko_lines), 1)
        ]
