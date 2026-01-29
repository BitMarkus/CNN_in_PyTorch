"""
ClassSorter - High-Confidence Image Selection for LoRA Training

This script analyzes images using a trained CNN model to select the highest-confidence
examples for each class. It creates a filtered dataset ideal for training conditional LoRAs.

The script works with images organized in class folders within a prediction directory.
It predicts each image's class, keeps only correctly predicted ones, and selects the
most confident examples based on the chosen selection mode.

Logit Threshold Filtering:
Problem: Softmax creates forced probability distributions even when model is uncertain.
Example: Logits [-2, -1.8, -1.9] → Softmax [0.25, 0.45, 0.3] shows "45% confidence"
         but the model is actually uncertain (all logits are negative).

Solution: Use logit threshold to filter out images where max_logit < threshold.
• max_logit > 0: Model thinks image belongs to this class (positive evidence)
• max_logit < 0: Model is uncertain or thinks it does NOT belong (negative evidence)
• Recommended threshold: 0.0 (filters out all negative-evidence images)

Features:
1. Works with arbitrary number of classes (2, 9, or more)
2. Two selection modes: Top N images or confidence threshold
3. Logit threshold filtering to remove misleading high-softmax predictions
4. Only keeps correctly predicted images
5. Renames files with confidence scores and predicted class
6. Includes logit values in filenames when threshold is used
7. Creates organized output folder structure with comprehensive statistics

Selection Modes:
1. 'top_n': Select top N most confident images per class
2. 'threshold': Select all images with confidence >= threshold

Additional Filtering:
• Logit threshold: Filter images where max_logit >= threshold
• When enabled, prevents selection of images where model lacks positive evidence

Output Files:
1. selection_statistics.csv - Per-class statistics
2. selected_images_details.csv - Detailed info for each selected image
3. selection_config.json - Configuration used
4. logit_statistics.json - Detailed logit distribution analysis
5. README.txt - Complete documentation of the filtering process
"""

import torch
import pandas as pd
import shutil
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
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
        # NEW: Logit threshold settings
        self.use_logit_threshold = setting['sort_use_logit_threshold']
        self.logit_threshold = setting['sort_logit_threshold']
        # Paths - FIXED: Use pth_prediction instead of pth_sort_input
        self.pth_prediction = setting['pth_prediction'].resolve()
        self.sort_output_dir = setting['pth_sort_output'].resolve()
        # Model checkpoint path
        self.pth_checkpoint = setting['pth_checkpoint'].resolve()
        # List of class names
        self.classes = setting['classes']  
        # Inference batch size
        self.batch_size_pred = setting['sort_pred_batch_size']  
        # File naming options
        self.rename_with_confidence = setting['sort_rename_with_confidence'] 
        
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
        
        # Statistics tracking - UPDATED to include logit stats
        self.stats = {
            'total_processed': 0,
            'correct_predictions': 0,
            'passed_logit_threshold': 0,
            'per_class': {cls: {'total': 0, 'correct': 0, 'selected': 0} for cls in self.classes}
        }
        # Display configuration
        self._print_configuration()
    
    # Validate the loaded configuration
    def _validate_configuration(self):
        
        # Validate selection mode
        if self.selection_mode not in ['top_n', 'threshold']:
            raise ValueError(f"Invalid selection_mode in settings: {self.selection_mode}. Must be 'top_n' or 'threshold'")
        
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
        
        # Validate logit threshold settings
        if not isinstance(self.use_logit_threshold, bool):
            raise ValueError(f"sort_use_logit_threshold must be boolean, got {type(self.use_logit_threshold)}")
        
        if not isinstance(self.logit_threshold, (int, float)):
            raise ValueError(f"sort_logit_threshold must be numeric, got {type(self.logit_threshold)}")
        
        # Validate input directory exists
        if not self.pth_prediction.exists():
            raise ValueError(f"Input directory does not exist: {self.pth_prediction}")
        
        # Create output directory if it doesn't exist - FIXED
        self.sort_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up input and output paths with timestamp for uniqueness
    def _setup_paths(self):
        
        # Create timestamped output directory
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        if self.selection_mode == 'top_n':
            output_name = f"top{self.selection_value}"
        else:
            output_name = f"thresh{self.selection_value:.2f}"
        
        # Add logit threshold info to folder name if used
        if self.use_logit_threshold:
            output_name += f"_logit{self.logit_threshold}"
        
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
        print(f"Use logit threshold: {self.use_logit_threshold}")
        if self.use_logit_threshold:
            print(f"Logit threshold: {self.logit_threshold}")
            print(f"  Interpretation: Keep images where max_logit >= {self.logit_threshold}")
            print(f"  What this means:")
            print(f"    • max_logit > 0: Model thinks it's more likely than not")
            print(f"    • max_logit > 2: Model is quite confident")
            print(f"    • max_logit < 0: Model is uncertain/negative")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Classes: {', '.join(self.classes)}")
        print(f"Rename with confidence: {self.rename_with_confidence}")
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
            print("-" * 80)
            print(f"{'ID':<5} {'Checkpoint File':<60} {'Size (MB)':<10}")
            print("-" * 80)
            
            for idx, file in enumerate(checkpoint_files, 1):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"{idx:<5} {file.name:<60} {size_mb:.1f}")
            
            print("-" * 80)
            
            # Let user select
            while True:
                try:
                    choice = input("\nSelect a checkpoint (enter ID): ").strip()
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
            self.cnn.load_state_dict(torch.load(checkpoint_file))
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
    
    # Return indices of samples belonging to a specific folder
    def get_folder_indices(self, dataset, folder_name):
        return [i for i, (path, _) in enumerate(dataset.samples) 
                if Path(path).parent.name == folder_name]
    
    # Analyze all images to collect confidence data for each class
    # Returns:
    # Dictionary with structure: {class_name: [image_data_dict, ...]}
    # where image_data_dict contains:
    #    'original_path': Path to original image
    #    'confidence': float confidence score
    #    'predicted_class': str class name
    #    'true_class': str class name (from folder name)
    def analyze_images(self) -> Dict[str, List[Dict]]:

        print(f"\n{'='*60}")
        print(f"Analyzing all {len(self.classes)} classes")
        print(f"Selection mode: {self.selection_mode}")
        print(f"Selection value: {self.selection_value}")
        if self.use_logit_threshold:
            print(f"Logit threshold: {self.logit_threshold}")
        print(f"{'='*60}")
        
        # Dictionary to store all image data by true class
        class_image_data = {cls: [] for cls in self.classes}
        
        # Process each class folder
        for class_name in self.classes:
            folder_indices = self.get_folder_indices(self.ds.ds_pred.dataset, class_name)
            if not folder_indices:
                print(f"Warning: No images found for class: {class_name}")
                continue
            
            subset = Subset(self.ds.ds_pred.dataset, folder_indices)
            loader = DataLoader(subset, batch_size=self.batch_size_pred, shuffle=False)
            
            self.cnn.eval()
            with torch.no_grad():
                pbar = tqdm(loader, desc=f"Analyzing {class_name}", unit="img")
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
                        sample_idx = folder_indices[batch_idx * self.batch_size_pred + i]
                        original_path = self.ds.ds_pred.dataset.samples[sample_idx][0]
                        
                        # Calculate logit statistics
                        current_logits = raw_logits[i]
                        max_logit = current_logits.max()
                        min_logit = current_logits.min()
                        logit_range = max_logit - min_logit
                        
                        # Check if logits pass threshold
                        passes_logit_threshold = True  # Default if not using threshold
                        if self.use_logit_threshold:
                            passes_logit_threshold = (max_logit >= self.logit_threshold)
                            if passes_logit_threshold:
                                self.stats['passed_logit_threshold'] += 1
                        
                        # Store image data with logit information
                        image_data = {
                            'original_path': Path(original_path),
                            'confidence': confidence,
                            'predicted_class': predicted_class,
                            'true_class': class_name,
                            'is_correct': predicted_class == class_name,
                            # NEW: Logit information
                            'max_logit': float(max_logit),
                            'min_logit': float(min_logit),
                            'logit_range': float(logit_range),
                            'passes_logit_threshold': passes_logit_threshold,
                            'all_logits': current_logits.tolist()  # Store all logits for export
                        }
                        
                        # Update statistics
                        self.stats['total_processed'] += 1
                        self.stats['per_class'][class_name]['total'] += 1
                        
                        if image_data['is_correct']:
                            self.stats['correct_predictions'] += 1
                            self.stats['per_class'][class_name]['correct'] += 1
                            
                            # Only store correctly predicted images
                            # AND only if they pass logit threshold (when enabled)
                            if not self.use_logit_threshold or passes_logit_threshold:
                                class_image_data[class_name].append(image_data)
        
        return class_image_data
    
    # Select images based on the chosen selection mode.
    # Args: class_image_data: Dictionary from analyze_images()   
    # Returns: Dictionary with selected images per class   
    def select_images(self, class_image_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:

        selected_images = {}
        
        print(f"\n{'='*60}")
        print(f"Selecting images using {self.selection_mode} mode")
        if self.use_logit_threshold:
            print(f"Logit threshold filtering: Enabled (≥ {self.logit_threshold})")
        print(f"{'='*60}")
        
        for class_name, images in tqdm(class_image_data.items(), desc="Selecting images"):
            if not images:
                print(f"Warning: No correctly predicted images for class '{class_name}'")
                selected_images[class_name] = []
                continue
            
            # Sort by confidence (descending)
            sorted_images = sorted(images, key=lambda x: x['confidence'], reverse=True)
            
            if self.selection_mode == 'top_n':
                # Select top N images
                n = min(self.selection_value, len(sorted_images))
                selected = sorted_images[:n]
                
            elif self.selection_mode == 'threshold':
                # Select all images above threshold
                selected = [img for img in sorted_images if img['confidence'] >= self.selection_value]
            
            # Store selected images
            selected_images[class_name] = selected
            self.stats['per_class'][class_name]['selected'] = len(selected)
            
            if selected:
                top_conf = selected[0]['confidence']
                top_logit = selected[0]['max_logit']
                print(f"  {class_name}: {len(selected)}/{len(images)} images selected "
                      f"(conf: {top_conf:.3f}, max_logit: {top_logit:.2f})")
            else:
                print(f"  {class_name}: 0 images selected")
        
        return selected_images
    
    # Copy selected images to output directory and rename with confidence info.
    # Args: selected_images: Dictionary from select_images()    
    def copy_and_rename_images(self, selected_images: Dict[str, List[Dict]]):
        print(f"\n{'='*60}")
        print(f"Copying and renaming selected images")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}")
        
        total_copied = 0
        
        for class_name, images in tqdm(selected_images.items(), desc="Copying images"):
            if not images:
                continue
            
            # Create class directory
            class_dir = self.output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy each image
            for img_data in images:
                src_path = img_data['original_path']
                confidence_pct = int(img_data['confidence'] * 100)
                max_logit_val = img_data['max_logit']
                
                # Create new filename
                if self.rename_with_confidence:
                    original_stem = src_path.stem.rstrip('_')
                    # Include logit value in filename if threshold is used
                    if self.use_logit_threshold:
                        new_stem = f"{original_stem}_conf{confidence_pct}_logit{max_logit_val:.1f}-{img_data['predicted_class']}"
                    else:
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
                    'passes_logit_threshold': img_data.get('passes_logit_threshold', True),
                    'all_logits': img_data.get('all_logits', [])  # NEW: Include all logits
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
            'use_logit_threshold': self.use_logit_threshold,
            'logit_threshold': self.logit_threshold,
            'loaded_checkpoint': self.loaded_checkpoint_name,
            'classes': self.classes,
            'input_directory': str(self.pth_prediction),
            'output_directory': str(self.output_dir),
            'total_processed': self.stats['total_processed'],
            'total_correct': self.stats['correct_predictions'],
            'passed_logit_threshold': self.stats.get('passed_logit_threshold', 0),
            'overall_accuracy': self.stats['correct_predictions'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0,
            'rename_with_confidence': self.rename_with_confidence
        }
        
        config_json_path = self.output_dir / "selection_config.json"
        with open(config_json_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved configuration to: {config_json_path}")
        
        # NEW: Save logit statistics summary
        self._save_logit_statistics(class_image_data, selected_images)
    
    # NEW: Save detailed logit statistics
    def _save_logit_statistics(self, class_image_data: Dict[str, List[Dict]], 
                              selected_images: Dict[str, List[Dict]]):
        """Save detailed logit statistics to JSON file"""
        
        logit_stats = {
            'logit_threshold_applied': self.use_logit_threshold,
            'logit_threshold_value': self.logit_threshold if self.use_logit_threshold else None,
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
                    'logit_threshold_compliance': {
                        'images_above_threshold': sum(1 for img in all_images if img['max_logit'] >= self.logit_threshold) if self.use_logit_threshold else len(all_images),
                        'images_below_threshold': sum(1 for img in all_images if img['max_logit'] < self.logit_threshold) if self.use_logit_threshold else 0
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
            f.write(f"Use Logit Threshold: {self.use_logit_threshold}\n")
            if self.use_logit_threshold:
                f.write(f"Logit Threshold: {self.logit_threshold}\n")
            f.write(f"Number of Classes: {len(self.classes)}\n")
            f.write(f"Classes: {', '.join(self.classes)}\n")
            f.write(f"Loaded Checkpoint: {self.loaded_checkpoint_name}\n")
            f.write(f"Rename with Confidence: {self.rename_with_confidence}\n\n")
            
            f.write(f"Total Images Processed: {self.stats['total_processed']}\n")
            f.write(f"Correct Predictions: {self.stats['correct_predictions']}\n")
            f.write(f"Overall Accuracy: {self.stats['correct_predictions']/self.stats['total_processed']:.2%}\n")
            if self.use_logit_threshold:
                f.write(f"Images Passing Logit Threshold: {self.stats.get('passed_logit_threshold', 0)}\n")
            f.write("\n")
            
            f.write("Logit Threshold Explanation:\n")
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
            
            if self.rename_with_confidence:
                f.write("\nFilename Convention:\n")
                f.write("-" * 40 + "\n")
                f.write("Images are renamed with confidence percentage and predicted class:\n")
                if self.use_logit_threshold:
                    f.write("Example: image_name_conf95_logit3.2-KO_1096-01.png\n")
                    f.write("  - conf95: 95% confidence (softmax probability)\n")
                    f.write("  - logit3.2: Maximum logit value (raw model output)\n")
                    f.write("  - KO_1096-01: Predicted class\n")
                else:
                    f.write("Example: image_name_conf95-KO_1096-01.png\n")
                    f.write("  - conf95: 95% confidence\n")
                    f.write("  - KO_1096-01: Predicted class\n\n")
        
        print(f"Created README file: {readme_path}")
    
    # Run:
    def run(self):

        print(f"\n{'='*60}")
        print("CLASS SORTER - High-Confidence Image Selection")
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
            print(f"Correct predictions: {self.stats['correct_predictions']} ({self.stats['correct_predictions']/self.stats['total_processed']:.2%})")
            if self.use_logit_threshold:
                print(f"Images passing logit threshold: {self.stats.get('passed_logit_threshold', 0)}")
            print(f"Images selected: {total_selected}")
            print(f"Output directory: {self.output_dir}")
            print(f"\nReady for LoRA training with clean, high-confidence examples!")
            
            return self.output_dir
            
        except Exception as e:
            print(f"\nError during execution: {e}")
            import traceback
            traceback.print_exc()
            return None