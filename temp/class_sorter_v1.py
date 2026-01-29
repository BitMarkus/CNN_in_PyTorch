"""
ClassSorter - High-Confidence Image Selection for LoRA Training

This script analyzes images using a trained CNN model to select the highest-confidence
examples for each class. It creates a filtered dataset ideal for training conditional LoRAs.

The script works with images organized in class folders within a prediction directory.
It predicts each image's class, keeps only correctly predicted ones, and selects the
most confident examples based on the chosen selection mode.

Features:
1. Works with arbitrary number of classes (2, 9, or more)
2. Two selection modes: Top N images or confidence threshold
3. Only keeps correctly predicted images
4. Renames files with confidence scores and predicted class
5. Creates organized output folder structure

Selection Modes:
1. 'top_n': Select top N most confident images per class
2. 'threshold': Select all images with confidence >= threshold

Usage:
Configure settings in settings.py, then run the script.
"""

import torch
import pandas as pd
import shutil
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Union
import json
import sys
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
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'correct_predictions': 0,
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
        
        self.output_dir = self.sort_output_dir / f"{output_name}_{timestamp}"
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
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidences, predicted = torch.max(probabilities, 1)
                    
                    # Process each image in the batch
                    for i in range(len(images)):
                        pred_class_idx = predicted[i].item()
                        confidence = confidences[i].item()
                        predicted_class = self.classes[pred_class_idx]
                        
                        # Get the original image path
                        sample_idx = folder_indices[batch_idx * self.batch_size_pred + i]
                        original_path = self.ds.ds_pred.dataset.samples[sample_idx][0]
                        
                        # Store image data
                        image_data = {
                            'original_path': Path(original_path),
                            'confidence': confidence,
                            'predicted_class': predicted_class,
                            'true_class': class_name,
                            'is_correct': predicted_class == class_name
                        }
                        
                        # Update statistics
                        self.stats['total_processed'] += 1
                        self.stats['per_class'][class_name]['total'] += 1
                        
                        if image_data['is_correct']:
                            self.stats['correct_predictions'] += 1
                            self.stats['per_class'][class_name]['correct'] += 1
                            
                            # Only store correctly predicted images
                            class_image_data[class_name].append(image_data)
        
        return class_image_data
    
    # Select images based on the chosen selection mode.
    # Args: class_image_data: Dictionary from analyze_images()   
    # Returns: Dictionary with selected images per class   
    def select_images(self, class_image_data: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:

        selected_images = {}
        
        print(f"\n{'='*60}")
        print(f"Selecting images using {self.selection_mode} mode")
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
                print(f"  {class_name}: {len(selected)}/{len(images)} images selected "
                      f"(top confidence: {selected[0]['confidence']:.3f})")
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
                
                # Create new filename
                if self.rename_with_confidence:
                    original_stem = src_path.stem.rstrip('_')
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
            else:
                avg_confidence_all = avg_confidence_selected = 0
            
            stats_data.append({
                'class': class_name,
                'total_images': self.stats['per_class'][class_name]['total'],
                'correct_predictions': self.stats['per_class'][class_name]['correct'],
                'accuracy': self.stats['per_class'][class_name]['correct'] / self.stats['per_class'][class_name]['total'] if self.stats['per_class'][class_name]['total'] > 0 else 0,
                'selected_images': self.stats['per_class'][class_name]['selected'],
                'avg_confidence_all': avg_confidence_all,
                'avg_confidence_selected': avg_confidence_selected,
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
                    'true_class': img_data['true_class']
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
            'loaded_checkpoint': self.loaded_checkpoint_name,
            'classes': self.classes,
            'input_directory': str(self.pth_prediction),
            'output_directory': str(self.output_dir),
            'total_processed': self.stats['total_processed'],
            'total_correct': self.stats['correct_predictions'],
            'overall_accuracy': self.stats['correct_predictions'] / self.stats['total_processed'] if self.stats['total_processed'] > 0 else 0,
            'rename_with_confidence': self.rename_with_confidence
        }
        
        config_json_path = self.output_dir / "selection_config.json"
        with open(config_json_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved configuration to: {config_json_path}")
    
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
            f.write(f"Number of Classes: {len(self.classes)}\n")
            f.write(f"Classes: {', '.join(self.classes)}\n")
            f.write(f"Loaded Checkpoint: {self.loaded_checkpoint_name}\n")
            f.write(f"Rename with Confidence: {self.rename_with_confidence}\n\n")
            
            f.write(f"Total Images Processed: {self.stats['total_processed']}\n")
            f.write(f"Correct Predictions: {self.stats['correct_predictions']}\n")
            f.write(f"Overall Accuracy: {self.stats['correct_predictions']/self.stats['total_processed']:.2%}\n\n")
            
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
                f.write("Example: image_name_conf95-KO_1096-01.png\n")
                f.write("  - conf95: 95% confidence\n")
                f.write("  - KO_1096-01: Predicted class\n\n")
            
            f.write("\nUsage for LoRA Training:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Use this filtered dataset for training\n")
            f.write("2. Create captions using only class token (e.g., '<WT_JG>')\n")
            f.write("3. Recommended training parameters:\n")
            f.write("   - Learning rate: 2e-4\n")
            f.write("   - LoRA dim: 128\n")
            f.write("   - Alpha: 64\n")
            f.write("   - Steps: 14,000\n")
        
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
            print(f"Images selected: {total_selected}")
            print(f"Output directory: {self.output_dir}")
            print(f"\nReady for LoRA training with clean, high-confidence examples!")
            
            return self.output_dir
            
        except Exception as e:
            print(f"\nError during execution: {e}")
            import traceback
            traceback.print_exc()
            return None