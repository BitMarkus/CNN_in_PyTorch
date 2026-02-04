"""
DATASET GENERATOR FOR CROSS-VALIDATION

Purpose:
Generates train/test datasets for leave-one-cell-line-out cross-validation by
copying images from source folders to temporary training/testing directories.

Key Features:
1. Supports three data source modes:
   - "mixed": Uses single input folder with both synthetic and real images (backward compatible)
   - "synthetic_only": Train on synthetic images, validate/test on real images
   - "real_only": Use only real images for everything
2. Automatic folder creation and cleanup
3. Metadata generation for each dataset
4. Iterative or batch dataset generation

Folder Requirements:
For "synthetic_only" mode:
├── dataset_gen/
│   ├── input_synthetic/    # Training data (synthetic only)
│   │   ├── WT1/
│   │   │   ├── s001.jpg
│   │   │   └── s002.jpg
│   │   ├── WT2/
│   │   └── KO1/
│   └── input_real/         # Validation/Test data (real only)
│       ├── WT1/
│       │   ├── real1.jpg
│       │   └── real2.jpg
│       ├── WT2/
│       └── KO1/

For "mixed" mode (backward compatible):
├── dataset_gen/
│   └── input_mixed/              # Both synthetic and real images mixed
│       ├── WT1/
│       │   ├── s001.jpg
│       │   └── real1.jpg
│       ├── WT2/
│       └── KO1/

How It Works:
1. For each fold, determines which WT and KO cell lines are left out for testing
2. Copies images from appropriate source folders to:
   - data/train/: For training cell lines
   - data/test/: For test cell lines
3. If train_data_source = "synthetic_only":
   - Training cell lines: Copies from input_synthetic/
   - Test cell lines: Copies from input_real/
4. Records counts and generates metadata

Usage:
# In automatic cross-validation mode:
ds_gen = DatasetGenerator(mode="acv")
configs = ds_gen.get_dataset_configs()  # Get all fold configurations
dataset_dir = ds_gen.generate_dataset(test_wt="WT1", test_ko="KO1", dataset_idx=1)

# Or generate all datasets at once:
ds_gen.generate_all_datasets()

Important Methods:
- generate_dataset(): Creates single train/test split
- get_dataset_configs(): Returns list of all fold configurations
- cleanup(): Removes temporary training/testing images
- _get_source_dirs(): Determines correct source folder based on training_data_source

Note: This class is typically used internally by AutoCrossValidation.
"""

import shutil
from itertools import product
from pathlib import Path
# Own modules
from settings import setting

class DatasetGenerator():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self, mode):
        # Passed parameters
        self.mode = mode

        # Settings parameters
        self.input_dir = setting['pth_ds_gen_input']  # For backward compatibility
        self.input_dir_synthetic = setting.get('pth_ds_gen_input_synthetic', None)
        self.input_dir_real = setting.get('pth_ds_gen_input_real', None)
        self.training_data_source = setting.get('train_data_source', 'mixed')  # Default to mixed
        self.output_dir = setting['pth_ds_gen_output']
        self.wt_lines = setting['wt_lines']
        self.ko_lines = setting['ko_lines']
        self.train_dir = setting['pth_train']
        self.test_dir = setting['pth_test']
        self.acv_results_dir = setting['pth_acv_results']

    #############################################################################################################
    # METHODS:

    # Create train/test/result folders and WT/KO subdirectories
    def _create_folder_structure(self, dataset_path):

        # Folder for training and test images
        # If mode = acv (automatic cross validation), training and test images go to the data folder for training
        # the datafolder goes into the folder acv_results and a folder results will be generated within
        # If mode = gen, training and test images go to the dataset_gen/output folder/dataset_x folder
        # and NO folder for results for the dataset will be created
        if(self.mode == "acv"):
            # Create folder for automatic cross validation results
            self.acv_results_dir.mkdir(exist_ok=True)
            # Define directory for train and test images
            train_dir = self.train_dir
            test_dir = self.test_dir
        elif(self.mode == "gen"):
            # Define directory for train and test images
            train_dir = dataset_path / "train"
            test_dir = dataset_path / "test"
        else:
            return False
        
        (train_dir / "WT").mkdir(parents=True, exist_ok=True)
        (train_dir / "KO").mkdir(parents=True, exist_ok=True)
        (test_dir / "WT").mkdir(parents=True, exist_ok=True)
        (test_dir / "KO").mkdir(parents=True, exist_ok=True)
        
        return train_dir, test_dir

    # NEW: Get the correct source directory based on data source type
    def _get_source_dirs(self, is_training_cell_line, cell_line):
        """
        Returns (source_dir_for_copying, source_dir_for_counting)
        Based on training_data_source setting and whether this is a training or test cell line
        """
        if self.training_data_source == 'mixed' or self.input_dir_real is None:
            # Backward compatibility mode - use single input directory
            return self.input_dir / cell_line, self.input_dir / cell_line
        
        elif self.training_data_source == 'synthetic_only':
            if is_training_cell_line:
                # Training cell line: use synthetic images only
                return self.input_dir_synthetic / cell_line, self.input_dir_synthetic / cell_line
            else:
                # Test cell line: use real images only (for validation/testing)
                return self.input_dir_real / cell_line, self.input_dir_real / cell_line
        
        elif self.training_data_source == 'real_only':
            # Always use real images
            return self.input_dir_real / cell_line, self.input_dir_real / cell_line
        
        else:
            # Fallback to backward compatibility
            return self.input_dir / cell_line, self.input_dir / cell_line

    # Copy images from src_dir to dest_dir and update counts_dict
    def _copy_images(self, src_dir, dest_dir, counts_dict, key):
        counts_dict[key] = len(list(src_dir.iterdir()))
        for img in src_dir.iterdir():
            if img.is_file():
                shutil.copy2(img, dest_dir / img.name)

    # Generate and save dataset metadata
    def _generate_metadata(self, dataset_path, dataset_idx, train_counts, test_counts, test_wt, test_ko, data_source_info=""):
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

    # Generate a single dataset for a specific (test_wt, test_ko) pair
    def generate_dataset(self, test_wt, test_ko, dataset_idx):
        if self.mode == "acv":
            dataset_path = self.acv_results_dir / f"dataset_{dataset_idx}"
        elif self.mode == "gen":
            dataset_path = self.output_dir / f"dataset_{dataset_idx}"
        else:
            return False

        dataset_path.mkdir(exist_ok=True)
        train_dir, test_dir = self._create_folder_structure(dataset_path)

        # Initialize counters
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
            
            # Copy images
            if src_dir.exists():
                self._copy_images(src_dir, dest_dir, 
                                 test_counts["WT"] if wt == test_wt else train_counts["WT"], 
                                 wt)
            else:
                # Set count to 0 if directory doesn't exist
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
            
            # Copy images
            if src_dir.exists():
                self._copy_images(src_dir, dest_dir, 
                                 test_counts["KO"] if ko == test_ko else train_counts["KO"], 
                                 ko)
            else:
                # Set count to 0 if directory doesn't exist
                if ko == test_ko:
                    test_counts["KO"][ko] = 0
                else:
                    train_counts["KO"][ko] = 0

        self._generate_metadata(dataset_path, dataset_idx, train_counts, test_counts, test_wt, test_ko, data_source_info)

        # Print summary
        print(f"Dataset {dataset_idx}: {self.training_data_source} mode")
        print(f"  Training images: WT={sum(train_counts['WT'].values())}, KO={sum(train_counts['KO'].values())}")
        print(f"  Test images: WT={sum(test_counts['WT'].values())}, KO={sum(test_counts['KO'].values())}")

        return dataset_path
    
    # Generate all dataset combinations (legacy method, prefer iterative approach)
    def generate_all_datasets(self):

        for dataset_idx, (test_wt, test_ko) in enumerate(product(self.wt_lines, self.ko_lines), 1):
            self.generate_dataset(test_wt, test_ko, dataset_idx)

        return True

    # Delete training and test images (preserves results/metadata)
    def cleanup(self, dataset_path):

        # Delete images
        dirs_to_remove = [
            dataset_path / "train",
            dataset_path / "test"
        ]
        for dir_path in dirs_to_remove:
            if dir_path.exists():
                shutil.rmtree(dir_path)

    # Returns a list of all possible (test_wt, test_ko) configurations as dicts
    # Example output: [{"test_wt": "wt1", "test_ko": "ko1", "dataset_idx": 1}, {"test_wt": "wt1", "test_ko": "ko2", "dataset_idx": 2},...]
    def get_dataset_configs(self):
        return [
            {"test_wt": wt, "test_ko": ko, "dataset_idx": idx}
            for idx, (wt, ko) in enumerate(product(self.wt_lines, self.ko_lines), 1)
        ]