"""
ClassSorter - High-Confidence Image Selection for LoRA Training
================================================================================

Purpose:
--------
This script analyzes images using a trained CNN model to select (the highest-confidence)
examples for each class.

Key Features:
-------------
1. **Flexible Input Structure**: Works with ANY folder organization:
   - Images can be in nested subfolders (any depth)
   - Images can be directly in the prediction folder
   - Folder names are ignored unless they match known classes

2. **Smart Ground Truth Detection**:
   - If an image is in a folder named after a known class (from settings), 
     that folder name is used as the true class for accuracy checking
   - If an image is in an unknown folder (not in class list) OR directly in root,
     no ground truth is available and all predictions are accepted

3. **Intelligent Filtering**: 
   - Logit threshold filtering (removes uncertain predictions)
   - Confidence-based filtering (softmax probability)
   - Combined mode for maximum quality

4. **Multiple Selection Modes**:
   - 'top_n': Select top N most confident images per class
   - 'threshold': Select all images with confidence >= threshold
   - 'interval': Select all images with confidence between min and max values

5. **Comprehensive Output**:
   - Organizes selected images into class folders (named from settings)
   - Saves detailed statistics (CSV, JSON)
   - Creates README with complete documentation
   - Generates universal statistics (confidence & logit distributions)

How Ground Truth Detection Works:
---------------------------------
The script determines true class labels ONLY from parent folder names that match 
known classes from your settings file:

Example 1 - Known Class Folder (Ground Truth Available):
    prediction/WT_KM/cell.png → True class = "WT_KM" (if "WT_KM" in settings classes)
    → Prediction checked for correctness before selection

Example 2 - Unknown Folder (No Ground Truth):
    prediction/unknown_folder/cell.png → True class = None (folder not in class list)
    → All predictions accepted regardless of "correctness"

Example 3 - Root Directory (No Ground Truth):
    prediction/cell.png → True class = None (no parent folder)
    → All predictions accepted regardless of "correctness"

This design gives maximum flexibility:
- Use known class folders for validation/accuracy tracking
- Use any other folder structure for unlabeled data
- Mix both types freely in the same input directory

Logit Threshold Filtering:
-------------------------
Problem: Softmax creates forced probability distributions even when model is uncertain.
Example: Logits [-2, -1.8, -1.9] → Softmax [0.25, 0.45, 0.3] shows "45% confidence"
         but the model is actually uncertain (all logits are negative).

Solution: Use logit threshold to filter out images where max_logit < threshold.
• max_logit > 0: Model thinks image belongs to this class (positive evidence)
• max_logit < 0: Model is uncertain or thinks it does NOT belong (negative evidence)
• Recommended threshold: 0.0 (filters out all negative-evidence images)

Output Files:
-------------
output_dir/
├── [class_folders]/        # Selected images organized by predicted class
├── selection_statistics.csv    # Per-class statistics
├── selected_images_details.csv # Detailed info for each selected image
├── selection_config.json       # Configuration used
├── logit_statistics.json       # Detailed logit distribution analysis
├── README.txt                  # Complete documentation
└── statistics.txt              # Universal confidence and logit statistics

Settings (from settings file):
------------------------------
- classes: List of class names (output folders and ground truth reference)
- sort_selection_mode: 'top_n', 'threshold', or 'interval'
- sort_selection_value: N (int), threshold (float), or [min, max] (list)
- sort_filter_mode: 'confidence_only', 'logits_only', or 'combined'
- sort_logit_threshold: Float value for logit filtering
- sort_rename_images: Boolean for filename formatting
- sort_pred_batch_size: Batch size for inference

Author: [Your Name/Project]
Date: [Current Date]
Project: Fibroblast LoRA Training - Synthetic Data Curation
"""

import torch
import pandas as pd
import shutil
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
import json
import numpy as np
# Own modules
from dataset import Dataset
from settings import setting
from model import CNN_Model


class ClassSorter:
    
    def __init__(self, device):

        self.device = device
        
        # Load all configuration from settings
        self.selection_mode = setting['sort_selection_mode']
        self.selection_value = setting['sort_selection_value']
        # For interval mode, we need min and max values
        if self.selection_mode == 'interval':
            if isinstance(self.selection_value, (list, tuple)) and len(self.selection_value) == 2:
                self.confidence_min = self.selection_value[0]
                self.confidence_max = self.selection_value[1]
            else:
                raise ValueError("For 'interval' mode, sort_selection_value must be a list/tuple of [min, max]")
        else:
            self.confidence_min = None
            self.confidence_max = None
            
        # Filtering settings
        self.filter_mode = setting['sort_filter_mode']
        self.confidence_threshold = setting['sort_selection_value'] if self.selection_mode == 'threshold' else None
        self.logit_threshold = setting['sort_logit_threshold']

        # Confidence intervals (in percentage points, e.g., [10,20,30,...,90])
        self.sort_intervals = setting.get('sort_intervals', [10, 20, 30, 40, 50, 60, 70, 80, 90])
        # Logit intervals - values that make sense for your model's logit range
        self.logit_intervals = setting.get('logit_intervals', [-10, -5, -2, 0, 2, 5, 10])
        
        # Paths
        self.pth_prediction = setting['pth_prediction'].resolve()
        self.sort_output_dir = setting['pth_sort_output'].resolve()
        # Model checkpoint path
        self.pth_checkpoint = setting['pth_checkpoint'].resolve()
        # List of class names (from settings - these are the OUTPUT class folders)
        self.classes = setting['classes']  
        # Inference batch size
        self.batch_size_pred = setting['sort_pred_batch_size']  
        # File naming option
        self.rename_images = setting['sort_rename_images']
        
        # Track ground truth availability
        self.has_ground_truth = False
        self.true_class_mapping = {}  # Maps image path -> true class name (only if folder matches known class)
        self.unknown_folders = set()  # Track folders not in class list
        self.known_folders = set()    # Track folders that match class list
        
        # Validate configuration
        self._validate_configuration()
        # Set up paths
        self._setup_paths()

        # Initialize dataset and load prediction data
        self.ds = Dataset()
        if not self.ds.load_pred_dataset():
            raise ValueError("Failed to load prediction dataset")
        
        # Create model wrapper
        self.cnn_wrapper = CNN_Model()  
        # Load model wrapper with model information
        print(f"Creating new {self.cnn_wrapper.cnn_type} network...")
        # Get actual model (nn.Module)
        self.cnn = self.cnn_wrapper.load_model(device).to(device)
        print("New network was successfully created.")   
        
        # Load checkpoint
        self.checkpoint_loaded = False
        self.loaded_checkpoint_name = None
        self._load_checkpoint()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'correct_predictions': 0,
            'passed_confidence': 0,
            'passed_logits': 0,
            'passed_both': 0,
            'per_class': {cls: {'total': 0, 'correct': 0, 'selected': 0} for cls in self.classes}
        }
        
        self.interval_stats = {
            'confidence_intervals': self._create_interval_buckets(self.sort_intervals),
            'logit_intervals': self._create_logit_buckets(self.logit_intervals),
            'per_class_confidence': {cls: self._create_interval_buckets(self.sort_intervals) for cls in self.classes},
            'per_class_logits': {cls: self._create_logit_buckets(self.logit_intervals) for cls in self.classes}
        }
        
        # Display configuration
        self._print_configuration()
    
    def _create_interval_buckets(self, intervals):
        """Create bucket structure for confidence intervals"""
        buckets = {}
        
        # Add buckets for each interval
        for i in range(len(intervals) - 1):
            lower = intervals[i]
            upper = intervals[i + 1]
            bucket_name = f"{lower}-{upper}"
            buckets[bucket_name] = 0
        
        # Add bucket for < min
        buckets[f"<{intervals[0]}"] = 0
        
        # Add bucket for >= max
        buckets[f">={intervals[-1]}"] = 0
        
        return buckets
    
    def _create_logit_buckets(self, intervals):
        """Create bucket structure for logit intervals"""
        buckets = {}
        
        # Add buckets for each interval
        for i in range(len(intervals) - 1):
            lower = intervals[i]
            upper = intervals[i + 1]
            bucket_name = f"{lower}-{upper}"
            buckets[bucket_name] = 0
        
        # Add bucket for < min
        min_bucket = f"<{intervals[0]}"
        buckets[min_bucket] = 0
        
        # Add bucket for >= max
        max_bucket = f">={intervals[-1]}"
        buckets[max_bucket] = 0
        
        return buckets
    
    def _assign_to_confidence_bucket(self, confidence):
        """Assign a confidence value to the appropriate bucket"""
        confidence_pct = confidence * 100  # Convert to percentage
        
        # Check if below minimum
        if confidence_pct < self.sort_intervals[0]:
            return f"<{self.sort_intervals[0]}"
        
        # Check each interval
        for i in range(len(self.sort_intervals) - 1):
            lower = self.sort_intervals[i]
            upper = self.sort_intervals[i + 1]
            if lower <= confidence_pct < upper:
                return f"{lower}-{upper}"
        
        # If we get here, it's >= max
        return f">={self.sort_intervals[-1]}"
    
    def _assign_to_logit_bucket(self, logit_value):
        """Assign a logit value to the appropriate bucket"""
        
        # Check if below minimum
        if logit_value < self.logit_intervals[0]:
            return f"<{self.logit_intervals[0]}"
        
        # Check each interval
        for i in range(len(self.logit_intervals) - 1):
            lower = self.logit_intervals[i]
            upper = self.logit_intervals[i + 1]
            if lower <= logit_value < upper:
                return f"{lower}-{upper}"
        
        # If we get here, it's >= max
        return f">={self.logit_intervals[-1]}"
    
    def _analyze_folder_structure(self):
        """
        Analyze the input folder structure to determine ground truth availability.
        Only uses folder names as ground truth if they match a known class from settings.
        Unknown folders are treated as unlabeled (no ground truth).
        """
        print("\n" + "="*60)
        print("Analyzing input folder structure...")
        print("="*60)
        
        true_class_mapping = {}
        unknown_folders = set()
        known_folders = set()
        
        # Convert class list to set for faster lookup
        known_classes_set = set(self.classes)
        
        # Recursively find all image files and determine ground truth
        all_images = self._get_all_image_files_recursive(self.pth_prediction)
        
        for img_path in all_images:
            # Get the parent folder name (if any)
            parent_folder = img_path.parent
            folder_name = parent_folder.name if parent_folder != self.pth_prediction else None
            
            # Determine if we have ground truth for this image
            if folder_name and folder_name in known_classes_set:
                # Image is in a folder that matches a known class
                true_class_mapping[str(img_path)] = folder_name
                known_folders.add(folder_name)
            else:
                # Image is either:
                # 1. Directly in root folder (no parent folder)
                # 2. In an unknown folder (folder name not in class list)
                true_class_mapping[str(img_path)] = None
                if folder_name and folder_name not in known_classes_set:
                    unknown_folders.add(folder_name)
        
        self.has_ground_truth = len([v for v in true_class_mapping.values() if v is not None]) > 0
        self.true_class_mapping = true_class_mapping
        self.unknown_folders = unknown_folders
        self.known_folders = known_folders
        
        # Print analysis results
        total_images = len(all_images)
        images_with_gt = sum(1 for v in true_class_mapping.values() if v is not None)
        images_without_gt = total_images - images_with_gt
        
        print(f"\n📊 Folder Structure Analysis:")
        print(f"   Total images found: {total_images}")
        print(f"   Images with ground truth (in known class folders): {images_with_gt}")
        print(f"   Images without ground truth (root or unknown folders): {images_without_gt}")
        
        if known_folders:
            print(f"\n✅ Known class folders detected (will use for accuracy checking):")
            for folder in sorted(known_folders):
                count = sum(1 for k, v in true_class_mapping.items() if v == folder)
                print(f"   - {folder}/ ({count} images)")
        
        if unknown_folders:
            print(f"\n⚠️  Unknown folders detected (will be treated as unlabeled):")
            for folder in sorted(unknown_folders):
                count = sum(1 for img_path in all_images 
                           if img_path.parent.name == folder and str(img_path) in true_class_mapping)
                print(f"   - {folder}/ ({count} images) - NOT in class list")
            print(f"\n   → Images in these folders will be treated like root directory images")
            print(f"   → All predictions will be accepted regardless of 'correctness'")
        
        if not known_folders:
            print(f"\n📌 No known class folders found.")
            print(f"   → All images will be treated as unlabeled (no ground truth)")
            print(f"   → All predictions will be accepted for selection")
        
        return true_class_mapping
    
    def _get_all_image_files_recursive(self, folder_path: Path) -> List[Path]:
        """Recursively get all image files from a folder and its subfolders"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
        image_files = []
        
        if not folder_path.exists():
            return image_files
        
        try:
            for item in folder_path.iterdir():
                if item.is_dir():
                    image_files.extend(self._get_all_image_files_recursive(item))
                elif item.is_file() and item.suffix.lower() in image_extensions:
                    image_files.append(item)
        except PermissionError:
            print(f"  WARNING: Permission denied accessing: {folder_path}")
        
        return image_files
    
    def get_true_class_for_image(self, image_path: str) -> Optional[str]:
        """
        Get the true class for an image.
        Returns None if no ground truth available (image in root or unknown folder).
        Returns class name only if parent folder matches a known class.
        """
        return self.true_class_mapping.get(image_path)
    
    # Validate the loaded configuration
    def _validate_configuration(self):
        
        # Validate selection mode
        valid_modes = ['top_n', 'threshold', 'interval']
        if self.selection_mode not in valid_modes:
            raise ValueError(f"Invalid selection_mode in settings: {self.selection_mode}. Must be 'top_n', 'threshold', or 'interval'")
        
        # Validate selection value based on mode
        if self.selection_mode == 'top_n':
            if not isinstance(self.selection_value, int):
                raise ValueError(f"For 'top_n' mode, sort_selection_value must be integer, got {type(self.selection_value)}")
            if self.selection_value <= 0:
                raise ValueError(f"For 'top_n' mode, sort_selection_value must be positive, got {self.selection_value}")
        
        elif self.selection_mode == 'threshold':
            if not isinstance(self.selection_value, (int, float)):
                raise ValueError(f"For 'threshold' mode, sort_selection_value must be float, got {type(self.selection_value)}")
            if not (0.0 <= self.selection_value <= 1.0):
                raise ValueError(f"For 'threshold' mode, sort_selection_value must be between 0.0 and 1.0, got {self.selection_value}")
        
        elif self.selection_mode == 'interval':
            if not isinstance(self.selection_value, (list, tuple)) or len(self.selection_value) != 2:
                raise ValueError(f"For 'interval' mode, sort_selection_value must be a list/tuple of [min, max], got {self.selection_value}")
            min_val, max_val = self.selection_value
            if not all(isinstance(v, (int, float)) for v in [min_val, max_val]):
                raise ValueError(f"Interval values must be numeric, got {min_val} ({type(min_val)}) and {max_val} ({type(max_val)})")
            if not (0.0 <= min_val <= max_val <= 1.0):
                raise ValueError(f"Interval must satisfy 0.0 <= min <= max <= 1.0, got min={min_val}, max={max_val}")
        
        # Validate filter mode
        if self.filter_mode not in ['confidence_only', 'logits_only', 'combined']:
            raise ValueError(f"sort_filter_mode must be 'confidence_only', 'logits_only', or 'combined', got {self.filter_mode}")
        
        # Validate logit threshold
        if not isinstance(self.logit_threshold, (int, float)):
            raise ValueError(f"sort_logit_threshold must be numeric, got {type(self.logit_threshold)}")
        
        # Validate rename_images setting
        if not isinstance(self.rename_images, bool):
            raise ValueError(f"sort_rename_images must be boolean, got {type(self.rename_images)}")
        
        # Validate input directory exists
        if not self.pth_prediction.exists():
            raise ValueError(f"Input directory does not exist: {self.pth_prediction}")
        
        # Create output directory if it doesn't exist
        self.sort_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up input and output paths with timestamp for uniqueness
    def _setup_paths(self):
        
        # Create timestamped output directory
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        if self.selection_mode == 'top_n':
            output_name = f"top{self.selection_value}"
        elif self.selection_mode == 'threshold':
            output_name = f"thresh{self.selection_value:.2f}"
        else:  # interval
            output_name = f"interval{self.confidence_min:.2f}-{self.confidence_max:.2f}"
        
        # Add filter mode and thresholds to folder name
        if self.filter_mode == 'confidence_only':
            output_name += f"_conf-only"
        elif self.filter_mode == 'logits_only':
            output_name += f"_logit{self.logit_threshold}-only"
        elif self.filter_mode == 'combined':
            if self.selection_mode == 'threshold':
                output_name += f"_conf{self.selection_value:.2f}+logit{self.logit_threshold}"
            elif self.selection_mode == 'interval':
                output_name += f"_conf{self.confidence_min:.2f}-{self.confidence_max:.2f}+logit{self.logit_threshold}"
            else:  # top_n
                output_name += f"_top{self.selection_value}+logit{self.logit_threshold}"
        
        self.output_dir = self.sort_output_dir / output_name / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print the current configuration
    def _print_configuration(self):

        print(f"\n{'='*60}")
        print("CLASS SORTER - Configuration")
        print(f"{'='*60}")
        print(f"Input directory: {self.pth_prediction}")
        print(f"Output directory: {self.output_dir}")
        print(f"Selection mode: {self.selection_mode}")
        print(f"Selection value: {self.selection_value}")
        print(f"Filter mode: {self.filter_mode}")
        
        if self.filter_mode == 'confidence_only':
            print(f"Filtering: CONFIDENCE ONLY")
            if self.selection_mode == 'threshold':
                print(f"Confidence threshold: >= {self.selection_value:.3f}")
            elif self.selection_mode == 'interval':
                print(f"Confidence interval: {self.confidence_min:.3f} <= confidence <= {self.confidence_max:.3f}")
            else:  # top_n
                print(f"Top N images: {self.selection_value} per class (sorted by confidence)")
        
        elif self.filter_mode == 'logits_only':
            print(f"Filtering: LOGITS ONLY")
            print(f"Logit threshold: max_logit >= {self.logit_threshold}")
            print(f"  Interpretation:")
            print(f"    • max_logit > 0: Model thinks it's more likely than not")
            print(f"    • max_logit > 2: Model is quite confident")
            print(f"    • max_logit < 0: Model is uncertain/negative")
            if self.selection_mode == 'threshold':
                print(f"Note: Filtering by LOGITS ONLY - confidence threshold is NOT applied")
            elif self.selection_mode == 'interval':
                print(f"Note: Filtering by LOGITS ONLY - confidence interval is NOT applied")
            else:  # top_n
                print(f"Note: Selecting top {self.selection_value} images by logit value")
        
        elif self.filter_mode == 'combined':
            print(f"Filtering: COMBINED (BOTH)")
            if self.selection_mode == 'threshold':
                print(f"Confidence threshold: >= {self.selection_value:.3f}")
                print(f"Logit threshold: max_logit >= {self.logit_threshold}")
                print(f"Note: Images must pass BOTH thresholds")
            elif self.selection_mode == 'interval':
                print(f"Confidence interval: {self.confidence_min:.3f} <= confidence <= {self.confidence_max:.3f}")
                print(f"Logit threshold: max_logit >= {self.logit_threshold}")
                print(f"Note: Images must pass BOTH the confidence interval AND logit threshold")
            else:  # top_n
                print(f"Top N images: {self.selection_value} per class")
                print(f"Logit threshold: max_logit >= {self.logit_threshold}")
                print(f"Note: First filter by logit threshold, then sort remaining by confidence")
        
        print(f"\nOutput Classes (from settings): {len(self.classes)} classes")
        print(f"  {', '.join(self.classes)}")
        
        print(f"\nUniversal Statistics:")
        print(f"  Confidence intervals: {self.sort_intervals}")
        print(f"  Logit intervals: {self.logit_intervals}")
        
        print(f"Rename images: {self.rename_images}")
        if self.rename_images:
            print(f"  Filename format: Auto-matched to filter mode")
        print(f"Checkpoint directory: {self.pth_checkpoint}")
        print(f"{'='*60}\n")
    
    # Load model weights from a selected checkpoint
    def _load_checkpoint(self):

        # Get all checkpoint files
        checkpoint_files = []
        for file in self.pth_checkpoint.iterdir():
            if file.is_file() and file.suffix in ['.pt', '.model']:
                checkpoint_files.append(file)
        
        if not checkpoint_files:
            print("The checkpoint folder is empty!")
            self.checkpoint_loaded = False
            self.loaded_checkpoint_name = "untrained"
            print("WARNING: Using untrained weights!")
            return False
        
        # If only one checkpoint exists
        if len(checkpoint_files) == 1:
            checkpoint_file = checkpoint_files[0]
            print(f"\nFound single checkpoint: {checkpoint_file.name}")
            print("Loading automatically...")
        else:
            # Show interactive table for multiple checkpoints
            print(f"\nFound {len(checkpoint_files)} checkpoints:")
            print("-" * 100)
            print(f"{'ID':<5} {'Checkpoint File':<60} {'Size (MB)':<10} {'Info':<20}")
            print("-" * 100)
            
            for idx, file in enumerate(checkpoint_files, 1):
                size_mb = file.stat().st_size / (1024 * 1024)
                # Try to infer number of classes from filename (common naming pattern)
                filename_lower = file.name.lower()
                if '2class' in filename_lower or '2_class' in filename_lower or 'binary' in filename_lower:
                    info = "Likely 2-class"
                elif '9class' in filename_lower or '9_class' in filename_lower or 'multiclass' in filename_lower:
                    info = "Likely 9-class"
                else:
                    info = "Unknown"
                print(f"{idx:<5} {file.name:<60} {size_mb:<10.1f} {info:<20}")
            
            print("-" * 100)
            
            # SHOW CURRENT SETTINGS CLASS COUNT
            print(f"\n{'!'*60}")
            print(f"CURRENT SETTINGS: {len(self.classes)} CLASSES DEFINED")
            print(f"Classes: {', '.join(self.classes)}")
            print(f"{'!'*60}")
            print("\nMake sure you select a checkpoint trained for the correct number of classes!")
            print("  - For 2 classes (WT/KO): Select a binary classifier checkpoint")
            print("  - For 9 classes (cell lines): Select a multi-class checkpoint")
            print()
            
            # Let user select
            while True:
                try:
                    choice = input("Select a checkpoint (enter ID): ").strip()
                    if not choice:
                        print("No selection made. Using untrained weights.")
                        self.checkpoint_loaded = False
                        self.loaded_checkpoint_name = "untrained"
                        print("WARNING: Using untrained weights!")
                        return False
                    
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(checkpoint_files):
                        checkpoint_file = checkpoint_files[choice_idx]
                        break
                    else:
                        print(f"Invalid selection. Please enter a number between 1 and {len(checkpoint_files)}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        try:
            # Load checkpoint and verify class compatibility
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            
            # Try to infer number of classes from checkpoint
            if hasattr(self.cnn, 'fc'):
                # For standard CNN with fc layer
                checkpoint_num_classes = None
                if 'fc.weight' in checkpoint:
                    checkpoint_num_classes = checkpoint['fc.weight'].shape[0]
                elif 'fc.bias' in checkpoint:
                    checkpoint_num_classes = checkpoint['fc.bias'].shape[0]
                
                if checkpoint_num_classes is not None:
                    if checkpoint_num_classes != len(self.classes):
                        print(f"\n{'!'*60}")
                        print(f"WARNING: Checkpoint has {checkpoint_num_classes} classes,")
                        print(f"         but settings define {len(self.classes)} classes!")
                        print(f"         This may cause errors or incorrect predictions.")
                        print(f"{'!'*60}")
                        proceed = input("\nContinue anyway? (y/n): ").strip().lower()
                        if proceed != 'y':
                            print("Aborting. Please select a compatible checkpoint or update settings.")
                            return False
            
            self.cnn.load_state_dict(checkpoint)
            self.checkpoint_loaded = True
            self.loaded_checkpoint_name = checkpoint_file.stem
            print(f"Successfully loaded weights from {checkpoint_file.name}")
            return True
        except FileNotFoundError as e:
            print(f"\nError loading checkpoint: {str(e)}")
            print(f"Full path attempted: {checkpoint_file}")
            self.checkpoint_loaded = False
            self.loaded_checkpoint_name = "untrained"
            print("WARNING: Using untrained weights!")
            return False
        except Exception as e:
            print(f"\nError loading checkpoint: {str(e)}")
            self.checkpoint_loaded = False
            self.loaded_checkpoint_name = "untrained"
            print("WARNING: Using untrained weights!")
            return False
    
    # Get indices of ALL images (no folder filtering)
    def get_all_image_indices(self, dataset):
        """Get indices of ALL images in the dataset (no folder filtering)"""
        return list(range(len(dataset.samples)))
    
    # Analyze all images to collect confidence data for each class
    def analyze_images(self) -> Dict[str, List[Dict]]:
        """
        Analyze all images in the input directory (any folder structure).
        Predicts class for each image and organizes by predicted class.
        """

        # First, analyze folder structure to determine ground truth availability
        self._analyze_folder_structure()
        
        print(f"\n{'='*60}")
        print(f"Analyzing all images (any folder structure)")
        print(f"Filter mode: {self.filter_mode}")
        print(f"Selection mode: {self.selection_mode}")
        if self.filter_mode in ['logits_only', 'combined']:
            print(f"Logit threshold: {self.logit_threshold}")
        if self.filter_mode in ['confidence_only', 'combined']:
            if self.selection_mode == 'threshold':
                print(f"Confidence threshold: >= {self.selection_value:.3f}")
            elif self.selection_mode == 'interval':
                print(f"Confidence interval: {self.confidence_min:.3f} <= confidence <= {self.confidence_max:.3f}")
        elif self.filter_mode == 'logits_only' and self.selection_mode in ['threshold', 'interval']:
            print(f"Filtering by LOGITS ONLY (no confidence filtering)")
        
        if not self.has_ground_truth:
            print("\n📌 NOTE: No ground truth available for any images.")
            print("      All predictions will be treated as 'correct' for selection.")
        else:
            print("\n📌 NOTE: Some images have ground truth (from known class folders).")
            print("      Only correctly predicted images from those folders will be selected.")
            print("      Images without ground truth (root/unknown folders) are always accepted.")
        print(f"{'='*60}")
        
        # Dictionary to store all image data by PREDICTED class
        class_image_data = {cls: [] for cls in self.classes}
        
        # Get ALL image indices (no folder filtering)
        all_indices = self.get_all_image_indices(self.ds.ds_pred.dataset)
        
        if not all_indices:
            print("Warning: No images found in input directory!")
            return class_image_data
        
        # Process all images in batches
        subset = Subset(self.ds.ds_pred.dataset, all_indices)
        loader = DataLoader(subset, batch_size=self.batch_size_pred, shuffle=False)
        
        self.cnn.eval()
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Analyzing images", unit="img")
            
            for batch_idx, (images, _) in enumerate(pbar):
                outputs = self.cnn(images.to(self.device))
                
                # Calculate softmax probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                # Get raw logits for analysis
                raw_logits = outputs.cpu().numpy()
                
                # Process each image in the batch
                for i in range(len(images)):
                    pred_class_idx = predicted[i].item()
                    confidence = confidences[i].item()
                    predicted_class = self.classes[pred_class_idx]
                    
                    # Get the original image path
                    sample_idx = all_indices[batch_idx * self.batch_size_pred + i]
                    original_path = self.ds.ds_pred.dataset.samples[sample_idx][0]
                    
                    # Get true class (if available from known folder)
                    true_class = self.get_true_class_for_image(str(original_path))
                    
                    # Determine if prediction is correct
                    # Only check correctness if we have ground truth (image in known class folder)
                    if true_class is not None:
                        is_correct = (predicted_class == true_class)
                        accuracy_note = f" (ground truth: {true_class})"
                    else:
                        is_correct = True  # No ground truth - accept all predictions
                        accuracy_note = " (no ground truth)"
                    
                    # Calculate logit statistics
                    current_logits = raw_logits[i]
                    max_logit = current_logits.max()
                    min_logit = current_logits.min()
                    logit_range = max_logit - min_logit
                    
                    confidence_bucket = self._assign_to_confidence_bucket(confidence)
                    logit_bucket = self._assign_to_logit_bucket(max_logit)
                    
                    # Update global interval stats
                    self.interval_stats['confidence_intervals'][confidence_bucket] += 1
                    self.interval_stats['logit_intervals'][logit_bucket] += 1
                    
                    # Update per-class interval stats (by predicted class)
                    self.interval_stats['per_class_confidence'][predicted_class][confidence_bucket] += 1
                    self.interval_stats['per_class_logits'][predicted_class][logit_bucket] += 1
                    
                    # Check if image passes filters
                    passes_confidence = True
                    passes_logits = True
                    
                    # Check confidence criteria (for confidence_only or combined modes)
                    if self.filter_mode in ['confidence_only', 'combined']:
                        if self.selection_mode == 'threshold':
                            # Apply confidence threshold
                            passes_confidence = (confidence >= self.selection_value)
                        elif self.selection_mode == 'interval':
                            # Apply confidence interval
                            passes_confidence = (self.confidence_min <= confidence <= self.confidence_max)
                        else:  # top_n
                            # For top_n mode, don't apply threshold during analysis
                            passes_confidence = True
                        
                        if passes_confidence:
                            self.stats['passed_confidence'] += 1
                    else:
                        # For logits_only mode, don't check confidence at all
                        passes_confidence = True
                    
                    # Check logit threshold (ONLY for logits_only or combined modes)
                    if self.filter_mode in ['logits_only', 'combined']:
                        passes_logits = (max_logit >= self.logit_threshold)
                        if passes_logits:
                            self.stats['passed_logits'] += 1
                    else:
                        # For confidence_only mode, don't check logits at all
                        passes_logits = True
                    
                    # Check combined
                    if self.filter_mode == 'combined':
                        passes_both = passes_confidence and passes_logits
                        if passes_both:
                            self.stats['passed_both'] += 1
                    
                    # Store image data with logit information
                    image_data = {
                        'original_path': Path(original_path),
                        'confidence': confidence,
                        'predicted_class': predicted_class,
                        'true_class': true_class if true_class else "unknown",
                        'is_correct': is_correct,
                        'has_ground_truth': true_class is not None,
                        # Logit information
                        'max_logit': float(max_logit),
                        'min_logit': float(min_logit),
                        'logit_range': float(logit_range),
                        'passes_confidence': passes_confidence,
                        'passes_logits': passes_logits,
                        'passes_both': passes_confidence and passes_logits,
                        'all_logits': current_logits.tolist(),
                        'confidence_bucket': confidence_bucket,
                        'logit_bucket': logit_bucket
                    }
                    
                    # Update statistics
                    self.stats['total_processed'] += 1
                    self.stats['per_class'][predicted_class]['total'] += 1
                    
                    if image_data['is_correct']:
                        self.stats['correct_predictions'] += 1
                        self.stats['per_class'][predicted_class]['correct'] += 1
                        
                        # Store image based on filter mode
                        store_image = False
                        
                        if self.filter_mode == 'confidence_only':
                            # For 'top_n' mode, store all correct images (filter later)
                            # For 'threshold' or 'interval' mode, only store if passes confidence criteria
                            if self.selection_mode == 'top_n':
                                store_image = True
                            else:  # threshold or interval mode
                                store_image = passes_confidence
                        
                        elif self.filter_mode == 'logits_only':
                            # For 'top_n' mode, store all correct images (filter later by sorting)
                            # For 'threshold' or 'interval' mode, only store if passes logit threshold
                            if self.selection_mode == 'top_n':
                                store_image = True
                            else:  # threshold or interval mode
                                store_image = passes_logits
                        
                        elif self.filter_mode == 'combined':
                            # Store if passes both confidence criteria AND logit threshold
                            # For top_n mode, store all correct (filter later)
                            if self.selection_mode == 'top_n':
                                store_image = True
                            else:  # threshold or interval mode
                                store_image = passes_confidence and passes_logits
                        
                        if store_image:
                            class_image_data[predicted_class].append(image_data)
        
        # Print summary of found images per class
        print("\n📊 Images per predicted class:")
        for cls in self.classes:
            count = len(class_image_data[cls])
            if count > 0:
                print(f"  {cls}: {count} images selected for consideration")
            else:
                print(f"  {cls}: 0 images")
        
        return class_image_data
    
    # Select images based on the chosen selection mode
    def select_images(self, class_image_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:

        selected_images = {}
        
        print(f"\n{'='*60}")
        print(f"Selecting images using {self.filter_mode} filtering")
        print(f"Selection mode: {self.selection_mode}")
        if self.filter_mode in ['logits_only', 'combined']:
            print(f"Logit threshold: >= {self.logit_threshold}")
        if self.filter_mode in ['confidence_only', 'combined']:
            if self.selection_mode == 'threshold':
                print(f"Confidence threshold: >= {self.selection_value:.3f}")
            elif self.selection_mode == 'interval':
                print(f"Confidence interval: {self.confidence_min:.3f} <= confidence <= {self.confidence_max:.3f}")
        elif self.filter_mode == 'logits_only' and self.selection_mode in ['threshold', 'interval']:
            print(f"Filtering by LOGITS ONLY (no confidence filtering)")
        print(f"{'='*60}")
        
        for class_name, images in tqdm(class_image_data.items(), desc="Selecting images"):
            if not images:
                print(f"Warning: No correctly predicted images for class '{class_name}'")
                selected_images[class_name] = []
                continue
            
            # Start with all stored images
            filtered_images = images
            
            # For 'confidence_only' with 'top_n' mode, apply confidence sorting
            if self.filter_mode == 'confidence_only' and self.selection_mode == 'top_n':
                # Sort by confidence (descending)
                sorted_images = sorted(filtered_images, key=lambda x: x['confidence'], reverse=True)
                # Select top N
                n = min(self.selection_value, len(sorted_images))
                selected = sorted_images[:n]
            
            # For 'logits_only' with 'top_n' mode, sort by logits
            elif self.filter_mode == 'logits_only' and self.selection_mode == 'top_n':
                # Sort by max_logit (descending)
                sorted_images = sorted(filtered_images, key=lambda x: x['max_logit'], reverse=True)
                # Select top N
                n = min(self.selection_value, len(sorted_images))
                selected = sorted_images[:n]
            
            # For 'combined' with 'top_n' mode: first filter by logit threshold, then sort by confidence
            elif self.filter_mode == 'combined' and self.selection_mode == 'top_n':
                # First, filter by logit threshold
                filtered_by_logits = [img for img in filtered_images if img['passes_logits']]
                # Then sort by confidence (descending)
                sorted_images = sorted(filtered_by_logits, key=lambda x: x['confidence'], reverse=True)
                # Select top N
                n = min(self.selection_value, len(sorted_images))
                selected = sorted_images[:n]
            
            # For 'threshold' or 'interval' mode, images are already filtered in analyze_images()
            elif self.selection_mode in ['threshold', 'interval']:
                # All stored images already pass the required filter(s)
                selected = filtered_images
            
            else:
                selected = []
            
            # Store selected images
            selected_images[class_name] = selected
            self.stats['per_class'][class_name]['selected'] = len(selected)
            
            if selected:
                if self.filter_mode == 'logits_only':
                    top_logit = selected[0]['max_logit']
                    top_conf = selected[0]['confidence']
                    print(f"  {class_name}: {len(selected)} images selected "
                          f"(top logit: {top_logit:.2f}, conf: {top_conf:.3f})")
                else:
                    top_conf = selected[0]['confidence']
                    top_logit = selected[0]['max_logit']
                    print(f"  {class_name}: {len(selected)} images selected "
                          f"(conf: {top_conf:.3f}, max_logit: {top_logit:.2f})")
            else:
                print(f"  {class_name}: 0 images selected")
        
        return selected_images
    
    # Copy selected images to output directory
    def copy_and_rename_images(self, selected_images: Dict[str, List[Dict]]):
        print(f"\n{'='*60}")
        print(f"Copying and renaming selected images")
        print(f"Output directory: {self.output_dir}")
        print(f"Output class folders: {', '.join(self.classes)}")
        print(f"{'='*60}")
        
        total_copied = 0
        
        for class_name, images in tqdm(selected_images.items(), desc="Copying images"):
            if not images:
                continue
            
            # Create class directory (named after the class from settings)
            class_dir = self.output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy each image
            for img_data in images:
                src_path = img_data['original_path']
                confidence_pct = int(img_data['confidence'] * 100)
                max_logit_val = img_data['max_logit']
                
                # Create new filename based on filter mode
                if self.rename_images:
                    original_stem = src_path.stem.rstrip('_')
                    
                    # Determine filename format based on filter mode
                    if self.filter_mode == 'confidence_only':
                        # Only include confidence
                        if self.selection_mode == 'interval':
                            # For interval mode, indicate it's within the interval
                            new_stem = f"{original_stem}_conf{confidence_pct}-in-interval-{img_data['predicted_class']}"
                        else:
                            new_stem = f"{original_stem}_conf{confidence_pct}-{img_data['predicted_class']}"
                    
                    elif self.filter_mode == 'logits_only':
                        # Only include logits
                        new_stem = f"{original_stem}_logit{max_logit_val:.1f}-{img_data['predicted_class']}"
                    
                    elif self.filter_mode == 'combined':
                        # Include both confidence and logits
                        if self.selection_mode == 'interval':
                            # For interval mode, indicate it's within the interval
                            new_stem = f"{original_stem}_conf{confidence_pct}-in-interval_logit{max_logit_val:.1f}-{img_data['predicted_class']}"
                        else:
                            new_stem = f"{original_stem}_conf{confidence_pct}_logit{max_logit_val:.1f}-{img_data['predicted_class']}"
                    
                    else:
                        # Fallback: only confidence
                        new_stem = f"{original_stem}_conf{confidence_pct}-{img_data['predicted_class']}"
                    
                    new_filename = f"{new_stem}{src_path.suffix}"
                else:
                    # Keep original filename
                    new_filename = src_path.name
                
                dst_path = class_dir / new_filename
                
                try:
                    # Copy and rename
                    shutil.copy2(src_path, dst_path)
                    total_copied += 1
                except Exception as e:
                    print(f"Error copying {src_path}: {e}")
        
        print(f"\nSuccessfully copied {total_copied} images to {self.output_dir}")

    def _save_interval_statistics(self):
        """Save confidence and logit interval statistics to a text file"""
        
        stats_path = self.output_dir / "statistics.txt"
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("CLASS SORTER - UNIVERSAL IMAGE STATISTICS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Checkpoint: {self.loaded_checkpoint_name}\n")
            f.write(f"Total Images Processed: {self.stats['total_processed']}\n")
            f.write(f"Overall Accuracy: {self.stats['correct_predictions']/self.stats['total_processed']:.2%}\n\n")
            
            # Ground truth availability info
            f.write("-" * 70 + "\n")
            f.write("GROUND TRUTH AVAILABILITY\n")
            f.write("-" * 70 + "\n")
            if self.has_ground_truth:
                f.write(f"Ground truth available for: {len([v for v in self.true_class_mapping.values() if v is not None])} images\n")
                f.write(f"  (Images in known class folders)\n")
                f.write(f"No ground truth for: {sum(1 for v in self.true_class_mapping.values() if v is None)} images\n")
                f.write(f"  (Images in root or unknown folders)\n")
            else:
                f.write("No ground truth available for any images.\n")
                f.write("All predictions were treated as 'correct' for selection.\n")
            f.write("\n")
            
            # Confidence Intervals
            f.write("-" * 70 + "\n")
            f.write("CONFIDENCE DISTRIBUTION (all images, all classes)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Intervals defined as: {self.sort_intervals}\n\n")
            
            total = self.stats['total_processed']
            cumulative = 0
            
            # Sort buckets in logical order
            confidence_keys = [k for k in self.interval_stats['confidence_intervals'].keys() if k]
            bucket_order = sorted(confidence_keys, 
                                key=lambda x: (
                                    -float('inf') if x.startswith('<') else 
                                    (float('inf') if x.startswith('>=') else float(x.split('-')[0]))
                                ))
            
            for bucket in bucket_order:
                count = self.interval_stats['confidence_intervals'][bucket]
                percentage = (count / total * 100) if total > 0 else 0
                cumulative += count
                cum_percentage = (cumulative / total * 100) if total > 0 else 0
                
                # Create a simple histogram bar
                bar_length = int(percentage / 2)
                bar = "#" * bar_length
                
                f.write(f"{bucket:>10} : {count:6d} images ({percentage:5.1f}%) {bar}\n")
                f.write(f"           Cumulative: {cumulative:6d} ({cum_percentage:5.1f}%)\n\n")
            
            # Per-class confidence intervals
            f.write("\n" + "-" * 70 + "\n")
            f.write("PER-CLASS CONFIDENCE DISTRIBUTION\n")
            f.write("-" * 70 + "\n\n")
            
            for class_name in self.classes:
                class_total = self.stats['per_class'][class_name]['total']
                if class_total == 0:
                    continue
                    
                f.write(f"\n{class_name} (Total: {class_total} images):\n")
                f.write("=" * 40 + "\n")
                
                for bucket in bucket_order:
                    count = self.interval_stats['per_class_confidence'][class_name].get(bucket, 0)
                    percentage = (count / class_total * 100) if class_total > 0 else 0
                    
                    bar_length = int(percentage / 2)
                    bar = "#" * bar_length
                    
                    f.write(f"{bucket:>10} : {count:4d} ({percentage:5.1f}%) {bar}\n")
            
            # Logit Intervals
            f.write("\n\n" + "-" * 70 + "\n")
            f.write("LOGIT DISTRIBUTION (max_logit values, all images, all classes)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Intervals defined as: {self.logit_intervals}\n\n")
            f.write("Interpretation of max_logit values:\n")
            f.write("  • > 0: Positive evidence (model thinks it belongs)\n")
            f.write("  • > 2: Quite confident\n")
            f.write("  • > 5: Very confident\n")
            f.write("  • < 0: Negative evidence (uncertain/does not belong)\n")
            f.write("  • < -2: Strong negative evidence\n\n")
            
            total = self.stats['total_processed']
            cumulative = 0

            # Get all logit keys
            logit_keys = [k for k in self.interval_stats['logit_intervals'].keys() if k and k != '']
            
            # Sort logit buckets
            logit_bucket_order = sorted(logit_keys,
                                    key=lambda x: (
                                        -float('inf') if x.startswith('<') else 
                                        (float('inf') if x.startswith('>=') else 
                                            float(x.split('-')[0] if x[0] != '-' else '-' + x.split('-')[1]))
                                    ))
            
            for bucket in logit_bucket_order:
                count = self.interval_stats['logit_intervals'][bucket]
                percentage = (count / total * 100) if total > 0 else 0
                cumulative += count
                cum_percentage = (cumulative / total * 100) if total > 0 else 0
                
                bar_length = int(percentage / 2)
                bar = "#" * bar_length
                
                f.write(f"{bucket:>10} : {count:6d} images ({percentage:5.1f}%) {bar}\n")
                f.write(f"           Cumulative: {cumulative:6d} ({cum_percentage:5.1f}%)\n\n")
            
            # Per-class logit intervals
            f.write("\n" + "-" * 70 + "\n")
            f.write("PER-CLASS LOGIT DISTRIBUTION\n")
            f.write("-" * 70 + "\n\n")
            
            for class_name in self.classes:
                class_total = self.stats['per_class'][class_name]['total']
                if class_total == 0:
                    continue
                    
                f.write(f"\n{class_name} (Total: {class_total} images):\n")
                f.write("=" * 40 + "\n")
                
                for bucket in logit_bucket_order:
                    count = self.interval_stats['per_class_logits'][class_name].get(bucket, 0)
                    percentage = (count / class_total * 100) if class_total > 0 else 0
                    
                    bar_length = int(percentage / 2)
                    bar = "#" * bar_length
                    
                    f.write(f"{bucket:>10} : {count:4d} ({percentage:5.1f}%) {bar}\n")
            
            # Summary statistics
            f.write("\n\n" + "=" * 70 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("Note: For detailed per-image statistics including mean/median values,\n")
            f.write("      please refer to selection_statistics.csv and selected_images_details.csv\n\n")
            
            # Selection criteria summary
            f.write("Selection Criteria Used:\n")
            f.write(f"  • Selection Mode: {self.selection_mode}\n")
            f.write(f"  • Selection Value: {self.selection_value}\n")
            f.write(f"  • Filter Mode: {self.filter_mode}\n")
            f.write(f"  • Logit Threshold: {self.logit_threshold}\n\n")
            
            f.write("=" * 70 + "\n")
        
        print(f"Saved universal statistics to: {stats_path}")
        return stats_path

    # Save statistics and configuration to CSV and JSON files
    def save_statistics(self, class_image_data: Dict[str, List[Dict]], 
                       selected_images: Dict[str, List[Dict]]):

        print(f"\n{'='*60}")
        print("Saving statistics")
        print(f"{'='*60}")
        
        # Prepare statistics data
        stats_data = []
        
        for class_name in self.classes:
            all_images = class_image_data.get(class_name, [])
            selected = selected_images.get(class_name, [])
            
            if all_images:
                avg_confidence_all = sum(img['confidence'] for img in all_images) / len(all_images)
                avg_confidence_selected = sum(img['confidence'] for img in selected) / len(selected) if selected else 0
                avg_max_logit_all = sum(img['max_logit'] for img in all_images) / len(all_images)
                avg_max_logit_selected = sum(img['max_logit'] for img in selected) / len(selected) if selected else 0
            else:
                avg_confidence_all = avg_confidence_selected = 0
                avg_max_logit_all = avg_max_logit_selected = 0
            
            stats_data.append({
                'class': class_name,
                'total_images': self.stats['per_class'][class_name]['total'],
                'correct_predictions': self.stats['per_class'][class_name]['correct'],
                'accuracy': self.stats['per_class'][class_name]['correct'] / self.stats['per_class'][class_name]['total'] if self.stats['per_class'][class_name]['total'] > 0 else 0,
                'selected_images': self.stats['per_class'][class_name]['selected'],
                'avg_confidence_all': avg_confidence_all,
                'avg_confidence_selected': avg_confidence_selected,
                'avg_max_logit_all': avg_max_logit_all,
                'avg_max_logit_selected': avg_max_logit_selected,
                'selection_ratio': self.stats['per_class'][class_name]['selected'] / self.stats['per_class'][class_name]['correct'] if self.stats['per_class'][class_name]['correct'] > 0 else 0
            })
        
        # Save to CSV
        df_stats = pd.DataFrame(stats_data)
        stats_csv_path = self.output_dir / "selection_statistics.csv"
        df_stats.to_csv(stats_csv_path, index=False)
        print(f"Saved statistics to: {stats_csv_path}")
        
        # Save detailed image list
        detailed_data = []
        for class_name, images in selected_images.items():
            for img_data in images:
                detailed_data.append({
                    'class': class_name,
                    'original_path': str(img_data['original_path']),
                    'confidence': img_data['confidence'],
                    'predicted_class': img_data['predicted_class'],
                    'true_class': img_data['true_class'],
                    'max_logit': img_data['max_logit'],
                    'min_logit': img_data['min_logit'],
                    'logit_range': img_data['logit_range'],
                    'passes_confidence': img_data.get('passes_confidence', True),
                    'passes_logits': img_data.get('passes_logits', True),
                    'passes_both': img_data.get('passes_both', True),
                    'all_logits': img_data.get('all_logits', []),
                    'confidence_bucket': img_data.get('confidence_bucket', ''),
                    'logit_bucket': img_data.get('logit_bucket', '')
                })
        
        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            detailed_csv_path = self.output_dir / "selected_images_details.csv"
            df_detailed.to_csv(detailed_csv_path, index=False)
            print(f"Saved detailed image list to: {detailed_csv_path}")
        
        # Save configuration
        config = {
            'selection_mode': self.selection_mode,
            'selection_value': self.selection_value,
            'filter_mode': self.filter_mode,
            'logit_threshold': self.logit_threshold if self.filter_mode in ['logits_only', 'combined'] else None,
            'confidence_threshold': self.selection_value if (self.selection_mode == 'threshold' and self.filter_mode in ['confidence_only', 'combined']) else None,
            'confidence_interval': [self.confidence_min, self.confidence_max] if self.selection_mode == 'interval' else None,
            'loaded_checkpoint': self.loaded_checkpoint_name,
            'classes': self.classes,
            'input_directory': str(self.pth_prediction),
            'output_directory': str(self.output_dir),
            'total_processed': self.stats['total_processed'],
            'total_correct': self.stats['correct_predictions'],
            'passed_confidence': self.stats.get('passed_confidence', 0),
            'passed_logits': self.stats.get('passed_logits', 0),
            'passed_both': self.stats.get('passed_both', 0),
            'overall_accuracy': self.stats['correct_predictions'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0,
            'rename_images': self.rename_images,
            'sort_intervals': self.sort_intervals,
            'logit_intervals': self.logit_intervals,
            'has_ground_truth': self.has_ground_truth,
            'unknown_folders': list(self.unknown_folders) if self.unknown_folders else []
        }
        
        config_json_path = self.output_dir / "selection_config.json"
        with open(config_json_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved configuration to: {config_json_path}")
        
        # Save logit and interval statistics summary
        self._save_logit_statistics(class_image_data, selected_images)
        self._save_interval_statistics()

    # Save detailed logit statistics
    def _save_logit_statistics(self, class_image_data: Dict[str, List[Dict]], selected_images: Dict[str, List[Dict]]):
        """Save detailed logit statistics to JSON file"""
        
        logit_stats = {
            'filter_mode': self.filter_mode,
            'logit_threshold': self.logit_threshold if self.filter_mode in ['logits_only', 'combined'] else None,
            'confidence_threshold': self.selection_value if (self.selection_mode == 'threshold' and self.filter_mode in ['confidence_only', 'combined']) else None,
            'confidence_interval': [self.confidence_min, self.confidence_max] if self.selection_mode == 'interval' else None,
            'rename_images': self.rename_images,
            'logit_interpretation': {
                'max_logit_greater_than_0': 'Model thinks the image belongs to this class (positive evidence)',
                'max_logit_less_than_0': 'Model is uncertain or thinks it does NOT belong (negative evidence)',
                'max_logit_greater_than_2': 'Model is quite confident',
                'max_logit_greater_than_5': 'Model is very confident'
            },
            'per_class_logit_statistics': {}
        }
        
        for class_name in self.classes:
            all_images = class_image_data.get(class_name, [])
            selected = selected_images.get(class_name, [])
            
            if all_images:
                # Calculate various logit statistics
                max_logits_all = [img['max_logit'] for img in all_images]
                max_logits_selected = [img['max_logit'] for img in selected]
                
                logit_stats['per_class_logit_statistics'][class_name] = {
                    'all_images_count': len(all_images),
                    'selected_images_count': len(selected),
                    'max_logit_all': {
                        'mean': float(np.mean(max_logits_all)),
                        'median': float(np.median(max_logits_all)),
                        'std': float(np.std(max_logits_all)),
                        'min': float(np.min(max_logits_all)) if max_logits_all else 0,
                        'max': float(np.max(max_logits_all)) if max_logits_all else 0,
                        'percentile_25': float(np.percentile(max_logits_all, 25)) if max_logits_all else 0,
                        'percentile_75': float(np.percentile(max_logits_all, 75)) if max_logits_all else 0
                    },
                    'max_logit_selected': {
                        'mean': float(np.mean(max_logits_selected)) if max_logits_selected else 0,
                        'median': float(np.median(max_logits_selected)) if max_logits_selected else 0,
                        'std': float(np.std(max_logits_selected)) if max_logits_selected else 0,
                        'min': float(np.min(max_logits_selected)) if max_logits_selected else 0,
                        'max': float(np.max(max_logits_selected)) if max_logits_selected else 0
                    },
                    'filter_compliance': {
                        'images_passing_confidence': sum(1 for img in all_images if img.get('passes_confidence', True)),
                        'images_passing_logits': sum(1 for img in all_images if img.get('passes_logits', True)),
                        'images_passing_both': sum(1 for img in all_images if img.get('passes_both', True))
                    }
                }
        
        # Save logit statistics to JSON
        logit_stats_path = self.output_dir / "logit_statistics.json"
        with open(logit_stats_path, 'w') as f:
            json.dump(logit_stats, f, indent=2)
        print(f"Saved logit statistics to: {logit_stats_path}")
    
    # Create a README file explaining the filtering process
    def create_readme(self):

        readme_path = self.output_dir / "README.txt"
        
        with open(readme_path, 'w') as f:
            f.write("High-Confidence Image Selection for LoRA Training\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Input Directory: {self.pth_prediction}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write(f"Selection Mode: {self.selection_mode}\n")
            f.write(f"Selection Value: {self.selection_value}\n")
            f.write(f"Filter Mode: {self.filter_mode}\n")
            
            if self.filter_mode == 'confidence_only':
                f.write(f"Filtering: CONFIDENCE ONLY\n")
                if self.selection_mode == 'threshold':
                    f.write(f"Confidence threshold: >= {self.selection_value:.3f}\n")
                elif self.selection_mode == 'interval':
                    f.write(f"Confidence interval: {self.confidence_min:.3f} <= confidence <= {self.confidence_max:.3f}\n")
                else:  
                    f.write(f"Top N images: {self.selection_value} per class (sorted by confidence)\n")
            
            elif self.filter_mode == 'logits_only':
                f.write(f"Filtering: LOGITS ONLY\n")
                f.write(f"Logit threshold: max_logit >= {self.logit_threshold}\n")
                if self.selection_mode == 'threshold':
                    f.write(f"Note: Filtering by LOGITS ONLY - confidence threshold is NOT applied\n")
                elif self.selection_mode == 'interval':
                    f.write(f"Note: Filtering by LOGITS ONLY - confidence interval is NOT applied\n")
                else:
                    f.write(f"Top N images: {self.selection_value} per class (sorted by logit value)\n")
            
            elif self.filter_mode == 'combined':
                f.write(f"Filtering: COMBINED (BOTH)\n")
                f.write(f"Logit threshold: max_logit >= {self.logit_threshold}\n")
                if self.selection_mode == 'threshold':
                    f.write(f"Confidence threshold: >= {self.selection_value:.3f}\n")
                    f.write(f"Note: Images must pass BOTH thresholds\n")
                elif self.selection_mode == 'interval':
                    f.write(f"Confidence interval: {self.confidence_min:.3f} <= confidence <= {self.confidence_max:.3f}\n")
                    f.write(f"Note: Images must pass BOTH the confidence interval AND logit threshold\n")
                else:
                    f.write(f"Top N images: {self.selection_value} per class\n")
                    f.write(f"Note: First filter by logit threshold, then sort remaining by confidence\n")
            
            f.write(f"Rename images: {self.rename_images}\n")
            f.write(f"Number of Classes: {len(self.classes)}\n")
            f.write(f"Classes: {', '.join(self.classes)}\n")
            f.write(f"Loaded Checkpoint: {self.loaded_checkpoint_name}\n\n")
            
            # Ground truth information
            f.write("GROUND TRUTH INFORMATION:\n")
            f.write("-" * 40 + "\n")
            if self.has_ground_truth:
                f.write("Ground truth is available for images in known class folders.\n")
                f.write(f"Known class folders detected: {', '.join(sorted(self.known_folders))}\n")
                if self.unknown_folders:
                    f.write(f"Unknown folders (treated as unlabeled): {', '.join(sorted(self.unknown_folders))}\n")
                f.write("\n")
                f.write("  • Images in known class folders: Only correctly predicted images are selected\n")
                f.write("  • Images in unknown folders or root: All predictions are accepted\n")
            else:
                f.write("No ground truth available for any images.\n")
                f.write("All predictions are treated as 'correct' for selection purposes.\n")
            f.write("\n")
            
            f.write(f"Total Images Processed: {self.stats['total_processed']}\n")
            f.write(f"Correct Predictions: {self.stats['correct_predictions']}\n")
            f.write(f"Overall Accuracy: {self.stats['correct_predictions']/self.stats['total_processed']:.2%}\n")
            
            if self.filter_mode == 'confidence_only':
                f.write(f"Images Passing Confidence Criteria: {self.stats.get('passed_confidence', 0)}\n")
            elif self.filter_mode == 'logits_only':
                f.write(f"Images Passing Logit Threshold: {self.stats.get('passed_logits', 0)}\n")
            elif self.filter_mode == 'combined':
                f.write(f"Images Passing Both Criteria: {self.stats.get('passed_both', 0)}\n")
            
            f.write("\n")
            
            f.write("Filter Mode Explanation:\n")
            f.write("-" * 40 + "\n")
            if self.filter_mode == 'confidence_only':
                f.write("Filtering by SOFTMAX CONFIDENCE only\n")
                f.write("  • Images selected based on softmax probability\n")
                f.write("  • Ignores raw logit values\n")
                f.write("  • May include images where model is uncertain (low/negative logits)\n")
                if self.selection_mode == 'interval':
                    f.write("  • Selects images with confidence values within a specific range\n")
                    f.write("  • Useful for excluding both low-confidence and extremely high-confidence images\n")
            
            elif self.filter_mode == 'logits_only':
                f.write("Filtering by RAW LOGITS only\n")
                f.write("  • Images selected based on max_logit value\n")
                f.write("  • Filters out uncertain predictions (max_logit < 0)\n")
                f.write("  • For 'top_n' mode: selects images with highest logits\n")
                f.write("  • For 'threshold' or 'interval' mode: applies logit threshold ONLY\n")
            
            elif self.filter_mode == 'combined':
                f.write("Filtering by BOTH confidence AND logits\n")
                f.write("  • For 'threshold' mode: images must pass BOTH thresholds\n")
                f.write("  • For 'interval' mode: images must fall within confidence interval AND pass logit threshold\n")
                f.write("  • For 'top_n' mode: first filter by logit threshold, then sort by confidence\n")
                f.write("  • Removes uncertain predictions (max_logit < threshold)\n")
                f.write("  • Ensures both high confidence AND positive evidence\n")
            
            f.write("\nLogit Threshold Explanation:\n")
            f.write("-" * 40 + "\n")
            f.write("The logit threshold filters out images where the model's maximum logit\n")
            f.write("value is below the specified threshold.\n\n")
            f.write("Interpretation of max_logit values:\n")
            f.write("  • max_logit > 0: Model thinks image belongs to this class\n")
            f.write("  • max_logit > 2: Model is quite confident\n")
            f.write("  • max_logit > 5: Model is very confident\n")
            f.write("  • max_logit < 0: Model is uncertain/negative evidence\n")
            f.write("  • max_logit < -2: Strong evidence AGAINST this class\n")
            f.write("\nSoftmax confidence can be misleading when logits are all low.\n")
            f.write("For example, logits [-2, -1.8, -1.9] give softmax [0.25, 0.45, 0.3]\n")
            f.write("which looks like 45% confidence, but the model is actually uncertain.\n\n")
            
            if self.rename_images:
                f.write("\nFilename Convention (auto-matched to filter mode):\n")
                f.write("-" * 40 + "\n")
                
                if self.filter_mode == 'confidence_only':
                    if self.selection_mode == 'interval':
                        f.write("Example: image_name_conf95-in-interval-KO_1096-01.png\n")
                        f.write("  - conf95: 95% confidence (softmax probability)\n")
                        f.write("  - in-interval: Indicates image falls within the specified confidence interval\n")
                        f.write("  - KO_1096-01: Predicted class\n")
                    else:
                        f.write("Example: image_name_conf95-KO_1096-01.png\n")
                        f.write("  - conf95: 95% confidence (softmax probability)\n")
                        f.write("  - KO_1096-01: Predicted class\n")
                
                elif self.filter_mode == 'logits_only':
                    f.write("Example: image_name_logit3.2-KO_1096-01.png\n")
                    f.write("  - logit3.2: Maximum logit value (raw model output)\n")
                    f.write("  - KO_1096-01: Predicted class\n")
                
                elif self.filter_mode == 'combined':
                    if self.selection_mode == 'interval':
                        f.write("Example: image_name_conf95-in-interval_logit3.2-KO_1096-01.png\n")
                        f.write("  - conf95: 95% confidence (softmax probability)\n")
                        f.write("  - in-interval: Indicates image falls within the specified confidence interval\n")
                        f.write("  - logit3.2: Maximum logit value (raw model output)\n")
                        f.write("  - KO_1096-01: Predicted class\n")
                    else:
                        f.write("Example: image_name_conf95_logit3.2-KO_1096-01.png\n")
                        f.write("  - conf95: 95% confidence (softmax probability)\n")
                        f.write("  - logit3.2: Maximum logit value (raw model output)\n")
                        f.write("  - KO_1096-01: Predicted class\n")
                
                f.write("\n")
            
            f.write("Per-Class Statistics:\n")
            f.write("-" * 40 + "\n")
            for class_name in self.classes:
                stats = self.stats['per_class'][class_name]
                if stats['total'] > 0:
                    accuracy = stats['correct'] / stats['total']
                    f.write(f"{class_name}:\n")
                    f.write(f"  Total: {stats['total']}\n")
                    f.write(f"  Correct: {stats['correct']} ({accuracy:.2%})\n")
                    f.write(f"  Selected: {stats['selected']}\n\n")
        
        print(f"Created README file: {readme_path}")
    
    # Run:
    def run(self):

        print(f"\n{'='*60}")
        print("CLASS SORTER - High-Confidence Image Selection")
        print(f"Flexible Input Mode - ANY folder structure supported")
        print(f"{'='*60}")
        
        try:
            # Step 1: Analyze all images to collect confidence data
            class_image_data = self.analyze_images()
            
            # Step 2: Select images based on criteria
            selected_images = self.select_images(class_image_data)
            
            # Step 3: Copy and rename selected images
            self.copy_and_rename_images(selected_images)
            
            # Step 4: Save statistics and configuration
            self.save_statistics(class_image_data, selected_images)
            
            # Step 5: Create README
            self.create_readme()
            
            # Final summary
            total_selected = sum(len(imgs) for imgs in selected_images.values())
            print(f"\n{'='*60}")
            print("SELECTION COMPLETE!")
            print(f"{'='*60}")
            print(f"Total images processed: {self.stats['total_processed']}")
            
            if self.has_ground_truth:
                print(f"Correct predictions: {self.stats['correct_predictions']} ({self.stats['correct_predictions']/self.stats['total_processed']:.2%})")
            else:
                print(f"All predictions accepted (no ground truth available)")
            
            if self.filter_mode == 'confidence_only':
                print(f"Images passing confidence criteria: {self.stats.get('passed_confidence', 0)}")
            elif self.filter_mode == 'logits_only':
                print(f"Images passing logit threshold: {self.stats.get('passed_logits', 0)}")
            elif self.filter_mode == 'combined':
                print(f"Images passing both criteria: {self.stats.get('passed_both', 0)}")
            
            print(f"Images selected: {total_selected}")
            print(f"Output directory: {self.output_dir}")
            print(f"\nUniversal statistics saved to: {self.output_dir / 'statistics.txt'}")
            
            print(f"\nReady for LoRA training with clean, high-confidence examples!")
            
            return self.output_dir
            
        except Exception as e:
            print(f"\nError during execution: {e}")
            import traceback
            traceback.print_exc()
            return None