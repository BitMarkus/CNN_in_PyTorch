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

    def generate_datasets(self):

        # Make folders for each dataset combination
        for dataset_idx, (test_wt, test_ko) in enumerate(product(self.wt_lines, self.ko_lines), 1):

            # Create dataset folder structure
            dataset_path = os.path.join(self.output_dir, f"dataset_{dataset_idx}")
            os.makedirs(dataset_path, exist_ok=True)
            # Create subdirectories
            train_dir = os.path.join(dataset_path, "train")
            test_dir = os.path.join(dataset_path, "prediction")
            result_dir = os.path.join(dataset_path, "result")
            os.makedirs(os.path.join(train_dir, "WT"), exist_ok=True)
            os.makedirs(os.path.join(train_dir, "KO"), exist_ok=True)
            os.makedirs(os.path.join(test_dir, "WT"), exist_ok=True)
            os.makedirs(os.path.join(test_dir, "KO"), exist_ok=True)
            os.makedirs(result_dir, exist_ok=True)

            # Initialize counters for metadata
            train_counts = {"WT": {}, "KO": {}}
            test_counts = {"WT": {}, "KO": {}}

            # Process WT (healthy) lines
            for wt in self.wt_lines:
                src_dir = os.path.join(self.input_dir, wt)
                if wt == test_wt:
                    # Copy to test/WT
                    dest_dir = os.path.join(test_dir, "WT")
                    test_counts["WT"][wt] = len(os.listdir(src_dir))
                else:
                    # Copy to train/WT
                    dest_dir = os.path.join(train_dir, "WT")
                    train_counts["WT"][wt] = len(os.listdir(src_dir))
                
                # Copy all images
                for img in os.listdir(src_dir):
                    src_path = os.path.join(src_dir, img)
                    if os.path.isfile(src_path):  # Skip subdirectories
                        shutil.copy2(src_path, os.path.join(dest_dir, img))

            # Process KO (sick) lines
            for ko in self.ko_lines:
                src_dir = os.path.join(self.input_dir, ko)
                if ko == test_ko:
                    # Copy to test/KO
                    dest_dir = os.path.join(test_dir, "KO")
                    test_counts["KO"][ko] = len(os.listdir(src_dir))
                else:
                    # Copy to train/KO
                    dest_dir = os.path.join(train_dir, "KO")
                    train_counts["KO"][ko] = len(os.listdir(src_dir))
                
                # Copy all images without renaming
                for img in os.listdir(src_dir):
                    src_path = os.path.join(src_dir, img)
                    if os.path.isfile(src_path):  # Skip subdirectories
                        shutil.copy2(src_path, os.path.join(dest_dir, img))

            # Create metadata file
            metadata_lines = [
                f"Dataset {dataset_idx} Configuration:",
                "",
                "=== TRAINING DATA ===",
                "Healthy (WT) lines:"
            ]
            metadata_lines.extend(f"- {line}: {count} images" for line, count in train_counts["WT"].items())
            metadata_lines.append(f"Total WT training images: {sum(train_counts['WT'].values())}")
            metadata_lines.append("")
            metadata_lines.append("Diseased (KO) lines:")
            metadata_lines.extend(f"- {line}: {count} images" for line, count in train_counts["KO"].items())
            metadata_lines.append(f"Total KO training images: {sum(train_counts['KO'].values())}")
            metadata_lines.append("")
            metadata_lines.append("=== TESTING DATA ===")
            metadata_lines.append(f"Healthy (WT) test line: {test_wt} ({test_counts['WT'].get(test_wt, 0)} images)")
            metadata_lines.append(f"Diseased (KO) test line: {test_ko} ({test_counts['KO'].get(test_ko, 0)} images)")

            metadata = '\n'.join(metadata_lines)

            with open(os.path.join(dataset_path, "dataset_info.txt"), 'w') as f:
                f.write('\n'.join(metadata_lines))

        return True