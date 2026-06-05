"""
CONFIDENCE ANALYZER FOR CROSS-VALIDATION RESULTS
[Keep your existing docstring here - unchanged]
"""

import os
import re
import shutil
import torch
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from itertools import product
import json
from sklearn.metrics import balanced_accuracy_score
from pathlib import Path
from torch.amp import autocast 
import numpy as np
# Own modules
from dataset import Dataset
from model import CNN_Model
from settings import setting

# Confidence analyzer
class ConfidenceAnalyzer:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device):

        # Passed parameters
        self.device = device
        
        # Settings parameters
        # Paths
        self.pth_acv_results = Path(setting['pth_acv_results']).absolute()
        self.pth_ds_gen_input = Path(setting['pth_ds_gen_input_real']).absolute()
        self.pth_test = Path(setting['pth_test']).absolute()
        self.pth_conf_analizer_results = Path(setting['pth_conf_analizer_results']).absolute()
        
        # Training data source configuration
        self.training_data_source = setting.get('train_data_source', 'mixed')
        self.pth_ds_gen_input_synthetic = Path(setting.get('pth_ds_gen_input_synthetic', '')).absolute() if setting.get('pth_ds_gen_input_synthetic') else None
        self.pth_ds_gen_input_real = Path(setting.get('pth_ds_gen_input_real', '')).absolute() if setting.get('pth_ds_gen_input_real') else None
        
        # Classes and cell lines
        self.classes = setting['classes']
        self.wt_lines = setting['wt_lines']
        self.ko_lines = setting['ko_lines']

        # Confidence intervals and filter types
        self.min_conf = setting['ca_min_conf']
        self.max_conf = setting['ca_max_conf']
        self.filter_type = setting['ca_filter_type']

        # Maximum number of checkpoints which for a dataset
        self.max_ckpts = setting['ca_max_ckpts']
        # Selection method for selecting the "best" checkpoints
        self.ckpt_select_method = setting['ca_ckpt_select_method']
        
        # Composite score settings (for composite_score selection method)
        self.penalty_weight = setting.get('chckpt_penalty_weight', 2.0)
        self.min_class_acc_threshold = setting.get('chckpt_min_class_acc_threshold', 0.60)
        
        # Which confusion matrices to use for checkpoint selection
        # Options: "validation", "test"
        cm_setting = setting.get('ca_use_test_cm', 'validation')
        if isinstance(cm_setting, bool):
            # Backward compatibility: True = "test", False = "validation"
            self.cm_source = "test" if cm_setting else "validation"
        else:
            self.cm_source = cm_setting.lower()
        
        # MAP the setting to actual file pattern (your files use "val" not "validation")
        if self.cm_source == "validation":
            self.cm_file_pattern = "val"
        else:
            self.cm_file_pattern = "test"
        
        # Which split to use for analysis
        self.split_to_use = setting.get('ca_split_to_use', 'validation')

        # Create model wrapper
        self.cnn_wrapper = CNN_Model()  
        # Load model wrapper with model information
        print(f"Creating new {self.cnn_wrapper.cnn_type} network...")
        # Get actual model (nn.Module)
        self.cnn = self.cnn_wrapper.load_model(device).to(device)
        print("New network was successfully created.")   

        # Create required directories
        self.pth_test.mkdir(parents=True, exist_ok=True)
        self.pth_conf_analizer_results.mkdir(parents=True, exist_ok=True)
        # List of confidences for each testing for each image
        self.image_history = defaultdict(list)
        
        # Print configuration
        print(f"\nConfidence Analyzer Configuration:")
        print(f"  Training data source: {self.training_data_source}")
        print(f"  Using {self.cm_source.upper()} confusion matrices (file pattern: '{self.cm_file_pattern}')")
        print(f"  Using '{self.split_to_use.upper()}' split for analysis")
        if self.ckpt_select_method == 'composite_score':
            print(f"  Checkpoint selection method: COMPOSITE SCORE (penalty_weight={self.penalty_weight}, min_class_acc_threshold={self.min_class_acc_threshold:.0%})")
        else:
            print(f"  Checkpoint selection method: {self.ckpt_select_method}")
        print(f"  Mixed folder: {self.pth_ds_gen_input}")
        if self.pth_ds_gen_input_synthetic:
            print(f"  Synthetic folder: {self.pth_ds_gen_input_synthetic}")
        if self.pth_ds_gen_input_real:
            print(f"  Real folder: {self.pth_ds_gen_input_real}")
        
        # Validate folder existence
        if self.training_data_source in ['synthetic_only', 'real_only']:
            if not self.pth_ds_gen_input_real or not self.pth_ds_gen_input_real.exists():
                print(f"  ⚠️  WARNING: Real folder not found or not configured")
                print(f"     Make sure 'pth_ds_gen_input_real' is set correctly in settings.py")
        
        if not self.pth_ds_gen_input.exists():
            print(f"  ⚠️  WARNING: Mixed folder not found: {self.pth_ds_gen_input}")

    #############################################################################################################
    # METHODS:

    # Helper method to extract epoch number from filename
    def _extract_epoch_from_filename(self, filename):
        """Extract epoch number from checkpoint filename"""
        
        # Your format: ckpt_pretr_densenet121_e02_bal0.669_comp0.415_ds1.pt
        # This captures the number after _e and before _
        match = re.search(r'_e(\d+)_', filename)
        if match:
            return int(match.group(1))
        
        # Original patterns for backward compatibility
        match = re.search(r'_e(\d+)[_\.]', filename)
        if match:
            return int(match.group(1))
        
        match = re.search(r'epoch[_\-](\d+)', filename)
        if match:
            return int(match.group(1))
        
        match = re.search(r'^(\d+)\.', filename)
        if match:
            return int(match.group(1))
        
        return None

    # Calculate composite score (matches train.py implementation)
    def _calculate_composite_score(self, class_accuracies, overall_accuracy, penalty_weight=2.0):
        """
        Calculate a composite score that balances overall accuracy with class performance parity.
        
        Formula: Composite = (Overall_Accuracy) - penalty_weight * (Standard_Deviation_of_Class_Accuracies)
        
        Args:
            class_accuracies: dict of per-class accuracies (e.g., {'WT': 0.85, 'KO': 0.75})
            overall_accuracy: mean accuracy across all classes
            penalty_weight: how heavily to penalize class imbalance (higher = stricter)
        
        Returns:
            composite_score: higher is better
            class_std: standard deviation of class accuracies (for logging)
            min_class_acc: minimum class accuracy (for logging)
        """
        acc_values = list(class_accuracies.values())
        class_std = np.std(acc_values)
        min_class_acc = min(acc_values)
        
        # Composite score: overall accuracy minus weighted penalty for imbalance
        composite_score = overall_accuracy - (penalty_weight * class_std)
        
        return composite_score, class_std, min_class_acc

    # Get only dataset folders that exist in acv_results
    def _get_available_datasets(self):
        datasets = {}
        for idx, (wt, ko) in enumerate(product(self.wt_lines, self.ko_lines), 1):
            dataset_path = self.pth_acv_results / f"dataset_{idx}"
            if dataset_path.exists():
                datasets[idx] = {'test_wt': wt, 'test_ko': ko, 'dataset_idx': idx}
        return datasets 

    # Generates test set directly from input files
    def _generate_test_set(self, test_wt, test_ko, current_dataset=None, total_datasets=None):
        """Generate test set using appropriate source directory based on configuration"""
        
        # Clear and recreate test folder
        shutil.rmtree(self.pth_test, ignore_errors=True)
        for cls in self.classes:
            (self.pth_test / cls).mkdir(parents=True)
        
        # Determine source directory based on training_data_source
        if self.training_data_source in ['synthetic_only', 'real_only'] and self.pth_ds_gen_input_real:
            source_dir = self.pth_ds_gen_input_real
        else:
            source_dir = self.pth_ds_gen_input
        
        if current_dataset is not None and total_datasets is not None:
            tqdm.write(f"\n>> PROCESSING DATASET {current_dataset} OF {total_datasets}:")
        
        tqdm.write(f"Generating test set with WT: {test_wt}, KO: {test_ko}")
        tqdm.write(f"Source directory: {source_dir}")
        
        # Copy WT images
        wt_source = source_dir / test_wt
        if wt_source.exists():
            shutil.copytree(wt_source, self.pth_test / 'WT', dirs_exist_ok=True)
            wt_count = len(list(wt_source.glob("*.*")))
            tqdm.write(f"  Copied {wt_count} WT images from {wt_source}")
        else:
            tqdm.write(f"  ⚠️  WARNING: WT source folder not found: {wt_source}")
        
        # Copy KO images
        ko_source = source_dir / test_ko
        if ko_source.exists():
            shutil.copytree(ko_source, self.pth_test / 'KO', dirs_exist_ok=True)
            ko_count = len(list(ko_source.glob("*.*")))
            tqdm.write(f"  Copied {ko_count} KO images from {ko_source}")
        else:
            tqdm.write(f"  ⚠️  WARNING: KO source folder not found: {ko_source}")
        
        total_images = len(list((self.pth_test / 'WT').glob("*.*"))) + \
                      len(list((self.pth_test / 'KO').glob("*.*")))
        if total_images == 0:
            tqdm.write(f"  ⚠️  CRITICAL: No images copied to test folder!")

    # Select top checkpoints based on specified metric
    def _select_checkpoints_by_metric(self, checkpoint_files, plots_path, top_n=3, method='balanced_sum'):
        scores = []
        
        for checkpoint_file in checkpoint_files:
            # Extract epoch number
            epoch_num = self._extract_epoch_from_filename(checkpoint_file)
            if epoch_num is None:
                continue
            
            # Find JSON file using your actual naming pattern
            json_path = None
            
            # Pattern 1: With leading zero (e02)
            candidates = list(plots_path.glob(f"*_e{epoch_num:02d}_*_val_cm.json"))
            if not candidates:
                # Pattern 2: Without leading zero (e2) - fallback
                candidates = list(plots_path.glob(f"*_e{epoch_num}_*_val_cm.json"))
            if not candidates:
                # Pattern 3: More flexible (anywhere in filename)
                candidates = list(plots_path.glob(f"*e{epoch_num:02d}*val_cm.json"))
            if not candidates:
                candidates = list(plots_path.glob(f"*e{epoch_num}*val_cm.json"))
            
            if not candidates:
                tqdm.write(f"Could not find val JSON for {checkpoint_file} (epoch {epoch_num})")
                continue
            
            json_path = candidates[0]
            
            try:
                with open(json_path, 'r') as f:
                    cm_data = json.load(f)
                
                # Extract accuracies from the JSON structure (from functions.py)
                if 'class_accuracy' in cm_data:
                    wt_acc = cm_data['class_accuracy'].get(self.classes[0], 0)
                    ko_acc = cm_data['class_accuracy'].get(self.classes[1], 0)
                    overall_acc = cm_data.get('overall_accuracy', 0)
                else:
                    wt_acc = ko_acc = overall_acc = 0
                
                # Calculate score based on selection method
                if method == 'balanced_sum':
                    score = (wt_acc + ko_acc) - abs(wt_acc - ko_acc)
                elif method == 'f1_score':
                    score = 2 * (wt_acc * ko_acc) / (wt_acc + ko_acc) if (wt_acc + ko_acc) > 0 else 0
                elif method == 'min_difference':
                    score = min(wt_acc, ko_acc)
                elif method == 'balanced_accuracy':
                    if 'true_labels' in cm_data and 'predicted_labels' in cm_data:
                        y_true = cm_data['true_labels']
                        y_pred = cm_data['predicted_labels']
                        score = balanced_accuracy_score(y_true, y_pred)
                    else:
                        score = (wt_acc + ko_acc) / 2
                elif method == 'composite_score':
                    # Calculate composite score using per-class accuracies
                    class_accuracies = {
                        self.classes[0]: wt_acc,
                        self.classes[1]: ko_acc
                    }
                    score, class_std, min_class_acc = self._calculate_composite_score(
                        class_accuracies, overall_acc, penalty_weight=self.penalty_weight
                    )
                    # Also store additional info for logging
                    tqdm.write(f"    Composite score for epoch {epoch_num}: {score:.4f} (std={class_std:.4f}, min_acc={min_class_acc:.2%})")
                else:
                    raise ValueError(f"Unknown selection method: {method}")
                
                scores.append((checkpoint_file, score, wt_acc, ko_acc, overall_acc))
                
            except Exception as e:
                tqdm.write(f"Error processing {json_path.name}: {str(e)}")
                continue
        
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [x[0] for x in scores[:top_n]]
        
        tqdm.write(f"\nSelected top {len(selected)} checkpoints by '{method}':")
        for i, (ckpt, score, wt_acc, ko_acc, overall_acc) in enumerate(scores[:top_n]):
            if method == 'composite_score':
                tqdm.write(f"  {i+1}. {ckpt}: composite={score:.4f} (WT={wt_acc:.2%}, KO={ko_acc:.2%}, overall={overall_acc:.2%})")
            else:
                tqdm.write(f"  {i+1}. {ckpt}: score={score:.4f} (WT={wt_acc:.2%}, KO={ko_acc:.2%}, overall={overall_acc:.2%})")
        
        return selected

    # Analyze a single dataset's checkpoints
    def _analyze_single_dataset(self, dataset_num, total_datasets):
        dataset_results = {}
        checkpoints_path = self.pth_acv_results / f"dataset_{dataset_num}" / 'checkpoints'
        plots_path = self.pth_acv_results / f"dataset_{dataset_num}" / 'plots'
        
        tqdm.write(f"\nLooking for {self.cm_source.upper()} confusion matrices in: {plots_path}")
        
        # Get all checkpoint files that have corresponding JSON files
        checkpoint_files = []
        
        for f in checkpoints_path.iterdir():
            if f.suffix == '.model' or f.suffix == '.pt':
                epoch_num = self._extract_epoch_from_filename(f.name)
                
                if epoch_num is not None:
                    # Find JSON file using your actual naming pattern
                    json_path = None
                    
                    # Pattern 1: With leading zero (e02) - most common
                    candidates = list(plots_path.glob(f"*_e{epoch_num:02d}_*_val_cm.json"))
                    if not candidates:
                        # Pattern 2: Without leading zero (e2) - fallback
                        candidates = list(plots_path.glob(f"*_e{epoch_num}_*_val_cm.json"))
                    if not candidates:
                        # Pattern 3: More flexible (anywhere in filename)
                        candidates = list(plots_path.glob(f"*e{epoch_num:02d}*val_cm.json"))
                    if not candidates:
                        candidates = list(plots_path.glob(f"*e{epoch_num}*val_cm.json"))
                    
                    if candidates:
                        json_path = candidates[0]
                        checkpoint_files.append(f.name)
                        tqdm.write(f"  ✓ Found: {f.name} -> {json_path.name}")
                    else:
                        tqdm.write(f"  ⚠️  No {self.cm_file_pattern} JSON found for {f.name} (epoch {epoch_num})")
                else:
                    tqdm.write(f"  ⚠️  Could not extract epoch from {f.name}")
        
        if not checkpoint_files:
            tqdm.write(f"\nERROR: No checkpoints with corresponding {self.cm_source.upper()} JSON files found!")
            tqdm.write(f"Checked in: {plots_path}")
            return {}
        
        # Select checkpoints if needed
        if self.max_ckpts is not None and len(checkpoint_files) > self.max_ckpts:
            checkpoint_files = self._select_checkpoints_by_metric(
                checkpoint_files, plots_path, top_n=self.max_ckpts, method=self.ckpt_select_method
            )
        else:
            tqdm.write(f"\nUsing all {len(checkpoint_files)} checkpoint(s):")
            for checkpoint_file in checkpoint_files:
                epoch_num = self._extract_epoch_from_filename(checkpoint_file)
                if epoch_num is not None:
                    # Find the corresponding JSON file for this checkpoint
                    candidates = list(plots_path.glob(f"*_e{epoch_num:02d}_*_val_cm.json"))
                    if not candidates:
                        candidates = list(plots_path.glob(f"*_e{epoch_num}_*_val_cm.json"))
                    if not candidates:
                        candidates = list(plots_path.glob(f"*e{epoch_num:02d}*val_cm.json"))
                    if not candidates:
                        candidates = list(plots_path.glob(f"*e{epoch_num}*val_cm.json"))
                    
                    if candidates:
                        try:
                            with open(candidates[0], 'r') as f:
                                cm_data = json.load(f)
                            wt_acc = cm_data['class_accuracy'].get(self.classes[0], 0)
                            ko_acc = cm_data['class_accuracy'].get(self.classes[1], 0)
                            overall_acc = cm_data.get('overall_accuracy', 0)
                            
                            if self.ckpt_select_method == 'composite_score':
                                # Calculate composite score for display
                                class_accuracies = {self.classes[0]: wt_acc, self.classes[1]: ko_acc}
                                comp_score, comp_std, min_acc = self._calculate_composite_score(
                                    class_accuracies, overall_acc, penalty_weight=self.penalty_weight
                                )
                                tqdm.write(f"  {checkpoint_file}: WT={wt_acc:.2%}, KO={ko_acc:.2%}, overall={overall_acc:.2%}, composite={comp_score:.4f}")
                            else:
                                tqdm.write(f"  {checkpoint_file}: WT={wt_acc:.2%}, KO={ko_acc:.2%}, overall={overall_acc:.2%}")
                        except:
                            tqdm.write(f"  {checkpoint_file}")
        
        # Process checkpoints
        with tqdm(checkpoint_files, desc=f"Dataset {dataset_num}/{total_datasets} - Checkpoints", position=1, leave=False) as pbar:
            for checkpoint_file in pbar:
                checkpoint_path = checkpoints_path / checkpoint_file
                try:
                    # Load checkpoint
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    
                    # Handle the checkpoint dictionary format from train.py
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        # Format from train.py
                        self.cnn.load_state_dict(checkpoint['model_state_dict'])
                        epoch_info = checkpoint.get('epoch', 'unknown')
                        if epoch_info != 'unknown':
                            epoch_info = epoch_info + 1
                        tqdm.write(f"  Loaded checkpoint (epoch {epoch_info}, acc={checkpoint.get('accuracy', 0):.2%})")
                    elif isinstance(checkpoint, dict):
                        # Maybe it's a direct state_dict wrapped in dict
                        self.cnn.load_state_dict(checkpoint)
                        tqdm.write(f"  Loaded checkpoint as dictionary")
                    else:
                        # Direct state_dict
                        self.cnn.load_state_dict(checkpoint)
                        tqdm.write(f"  Loaded checkpoint as direct state_dict")
                    
                    self.cnn.eval()
                    confidences = self._get_predictions_with_confidence(dataset_num)
                    dataset_results[checkpoint_file] = self._organize_prediction_results(confidences)
                except Exception as e:
                    tqdm.write(f"\nError processing {checkpoint_file}: {str(e)}")
                    if isinstance(e, RuntimeError) and "CUDA out of memory" in str(e):
                        raise
                    continue
                    
        return dataset_results

    # Get predictions with confidence scores
    def _get_predictions_with_confidence(self, dataset_num):
        test_loader = self._create_filtered_dataset(dataset_num)
        confidences = {}
        total_images = len(test_loader.dataset)
        
        if total_images == 0:
            tqdm.write(f"  WARNING: No images to predict for dataset {dataset_num}!")
            return confidences
        
        with torch.no_grad():
            with autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                with tqdm(test_loader, desc="Predicting images", total=total_images, position=0, leave=False) as img_pbar:
                    for batch_idx, (images, labels) in enumerate(img_pbar):
                        img_path = test_loader.dataset.dataset.samples[test_loader.dataset.indices[batch_idx]][0]
                        images = images.to(self.device)
                        if images.dim() == 3:
                            images = images.unsqueeze(0)
                        outputs = self.cnn(images)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        max_prob, pred_idx = torch.max(probs, 1)
                        confidences[img_path] = (
                            self.classes[labels.item()],
                            self.classes[pred_idx.item()],
                            max_prob.item()
                        )
        return confidences
    
    # Organize prediction results into structured format
    def _organize_prediction_results(self, confidences):
        per_image = {}
        aggregated = {cls: {'all_confidences': [], 'correct': [], 'incorrect': []} 
                     for cls in self.classes}
        for img_path, (true_class, pred_class, confidence) in confidences.items():
            img_key = Path(img_path).name
            per_image[img_key] = {
                'true_class': true_class,
                'pred_class': pred_class,
                'confidence': confidence
            }
            aggregated[true_class]['all_confidences'].append(confidence)
            if pred_class == true_class:
                aggregated[true_class]['correct'].append(confidence)
            else:
                aggregated[true_class]['incorrect'].append(confidence)
        return {
            'per_image_results': per_image,
            'aggregated_stats': aggregated
        }
    
    # Find images matching the specified confidence filter criteria
    def find_filtered_images(self, results):
        self._build_image_history(results)
        filtered_images = {cls: [] for cls in self.classes}
        
        for img_path, predictions in self.image_history.items():
            img_key = Path(img_path).name
            true_class = None
            
            source_dir = self._get_source_directory_for_search()
            
            for wt_line in self.wt_lines:
                if (source_dir / wt_line / img_key).exists():
                    true_class = 'WT'
                    break
                    
            if true_class is None:
                for ko_line in self.ko_lines:
                    if (source_dir / ko_line / img_key).exists():
                        true_class = 'KO'
                        break
            
            if true_class is None:
                continue
                
            confidences = [p['confidence'] for p in predictions]
            correct = [p['pred_class'] == true_class for p in predictions]
            avg_confidence = sum(confidences) / len(confidences)
            correctness_rate = sum(correct) / len(correct)
            
            if self.filter_type.lower() == 'correct':
                if (all(correct) and 
                    all(self.min_conf <= c <= self.max_conf for c in confidences)):
                    pred_classes = {p['pred_class'] for p in predictions}
                    if len(pred_classes) == 1:
                        filtered_images[pred_classes.pop()].append(
                            (img_path, avg_confidence, correctness_rate)
                        )
            elif self.filter_type.lower() == 'incorrect':
                if (not any(correct) and 
                    all(self.min_conf <= c <= self.max_conf for c in confidences)):
                    filtered_images[true_class].append(
                        (img_path, avg_confidence, correctness_rate))
            elif self.filter_type.lower() == 'low_confidence':
                if all(c < self.min_conf for c in confidences):
                    filtered_images[true_class].append(
                        (img_path, avg_confidence, correctness_rate))
            elif self.filter_type.lower() == 'unsure':
                if all(self.min_conf <= c <= self.max_conf for c in confidences):
                    filtered_images[true_class].append(
                        (img_path, avg_confidence, correctness_rate))
        
        return filtered_images
    
    def _get_source_directory_for_search(self):
        if self.training_data_source in ['synthetic_only', 'real_only'] and self.pth_ds_gen_input_real:
            return self.pth_ds_gen_input_real
        return self.pth_ds_gen_input
    
    def _build_image_history(self, results):
        for dataset_num, checkpoints in results.items():
            split_images, (split_count, other_count) = self._get_split_images(int(dataset_num))
            processed_images = 0
            
            for checkpoint_name, pred_data in checkpoints.items():
                for img_path, pred in pred_data['per_image_results'].items():
                    img_name = Path(img_path).name
                    if split_images is not None and img_name not in split_images:
                        continue
                    processed_images += 1
                    try:
                        original_path = self._find_original_image_path(img_name)
                        self.image_history[original_path].append({
                            'dataset': dataset_num,
                            'checkpoint': checkpoint_name,
                            **pred
                        })
                    except FileNotFoundError as e:
                        tqdm.write(f"Warning: {str(e)}")
                        continue
            
            if split_images is not None:
                tqdm.write(f"Dataset {dataset_num}: Processed {processed_images} {self.split_to_use} images, skipped {other_count} other images")

    def _get_split_images(self, dataset_num):
        """Get images from the appropriate split based on ca_split_to_use setting"""
        split_file = self.pth_acv_results / f"dataset_{dataset_num}" / "split_info.json"
        if not split_file.exists():
            tqdm.write(f"Dataset {dataset_num}: No split info found - will use all available images")
            return None, (0, 0)
        
        try:
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            
            if self.split_to_use == 'validation':
                images = set(split_info['validation']['WT'] + split_info['validation']['KO'])
                other_images = set(split_info['test']['WT'] + split_info['test']['KO'])
                tqdm.write(f"Dataset {dataset_num}: Using {len(images)} validation images")
                return images, (len(images), len(other_images))
                
            elif self.split_to_use == 'test':
                images = set(split_info['test']['WT'] + split_info['test']['KO'])
                other_images = set(split_info['validation']['WT'] + split_info['validation']['KO'])
                if len(images) == 0:
                    tqdm.write(f"Dataset {dataset_num}: WARNING - No test images found! (ds_val_from_test_split may be 1.0)")
                    tqdm.write(f"Dataset {dataset_num}: Falling back to validation images")
                    images = set(split_info['validation']['WT'] + split_info['validation']['KO'])
                    other_images = set()
                else:
                    tqdm.write(f"Dataset {dataset_num}: Using {len(images)} test-only images (excluding {len(other_images)} validation)")
                return images, (len(images), len(other_images))
                
            else:  # 'all'
                all_images = set(split_info['test']['WT'] + split_info['test']['KO'] + 
                                split_info['validation']['WT'] + split_info['validation']['KO'])
                tqdm.write(f"Dataset {dataset_num}: Using {len(all_images)} total images (test + validation)")
                return all_images, (len(all_images), 0)
                
        except Exception as e:
            tqdm.write(f"Error loading split info for dataset {dataset_num}: {str(e)}")
            return None, (0, 0)

    def _find_original_image_path(self, img_key):
        source_dir = self._get_source_directory_for_search()
        for line in self.wt_lines + self.ko_lines:
            candidate = source_dir / line / img_key
            if candidate.exists():
                return str(candidate)
        for line in self.wt_lines + self.ko_lines:
            candidate = self.pth_ds_gen_input / line / img_key
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"Original image not found for {img_key}")

    def organize_filtered_images(self, filtered_images):
        output_subdir = {
            'correct': "high_confidence_correct",
            'incorrect': "high_confidence_incorrect", 
            'low_confidence': "low_confidence",
            'unsure': "medium_confidence_unsure"
        }.get(self.filter_type.lower(), "filtered_images")
        
        output_dir = self.pth_conf_analizer_results / output_subdir
        copied_files = set()
        
        for class_name in self.classes:
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path, avg_confidence, correctness_rate in filtered_images[class_name]:
                if img_path in copied_files:
                    continue
                try:
                    original_name = Path(img_path).name
                    base, ext = os.path.splitext(original_name)
                    confidence_pct = int(round(avg_confidence * 100))
                    correctness_pct = int(round(correctness_rate * 100))
                    new_filename = f"{base}_conf{confidence_pct}_corr{correctness_pct}{ext}"
                    dest_path = class_dir / new_filename
                    shutil.copy2(img_path, dest_path)
                    copied_files.add(img_path)
                except Exception as e:
                    print(f"Error copying {img_path}: {str(e)}")
        
        self._create_filter_readme(output_dir, filtered_images)
        return output_dir

    def _save_results_to_csv(self, results, output_path):
        rows = []
        for dataset_num, checkpoints in results.items():
            for ckpt_name, pred_data in checkpoints.items():
                for cls, metrics in pred_data['aggregated_stats'].items():
                    if not metrics['all_confidences']:
                        continue
                    rows.append({
                        'dataset': dataset_num,
                        'checkpoint': ckpt_name,
                        'class': cls,
                        'mean_confidence': sum(metrics['all_confidences'])/len(metrics['all_confidences']),
                        'accuracy': len(metrics['correct'])/len(metrics['all_confidences']),
                        'total_samples': len(metrics['all_confidences']),
                        'correct_predictions': len(metrics['correct'])
                    })
        if rows:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            return True
        return False
    
    def _export_used_checkpoints(self, results):
        """Export information about which checkpoints were used for analysis"""
        
        print("\n>> Exporting used checkpoints information...")
        
        if not results:
            print("  WARNING: No results to export!")
            return False
        
        rows = []
        available_datasets = self._get_available_datasets()
        
        for dataset_num, checkpoints in results.items():
            dataset_num_int = int(dataset_num)
            if dataset_num_int not in available_datasets:
                print(f"  WARNING: Dataset {dataset_num_int} not found in available_datasets")
                continue
                
            config = available_datasets[dataset_num_int]
            checkpoint_files = list(checkpoints.keys())
            was_filtered = self.max_ckpts is not None and len(checkpoint_files) > self.max_ckpts
            
            for ckpt_name in checkpoint_files:
                epoch_num = self._extract_epoch_from_filename(ckpt_name)
                if epoch_num is None:
                    continue
                    
                plots_path = self.pth_acv_results / f"dataset_{dataset_num}" / 'plots'
                
                # FIXED: Match your actual file naming pattern
                # Your files: *_e10_*_val_cm.json (val comes AFTER the metrics)
                json_candidates = list(plots_path.glob(f"*_e{epoch_num:02d}_*_val_cm.json"))
                if not json_candidates:
                    json_candidates = list(plots_path.glob(f"*_e{epoch_num}_*_val_cm.json"))
                if not json_candidates:
                    # More flexible fallback
                    json_candidates = list(plots_path.glob(f"*e{epoch_num:02d}*val_cm.json"))
                if not json_candidates:
                    json_candidates = list(plots_path.glob(f"*e{epoch_num}*val_cm.json"))
                
                if not json_candidates:
                    print(f"  WARNING: No JSON found for {ckpt_name} (epoch {epoch_num})")
                    continue
                
                try:
                    with open(json_candidates[0], 'r') as f:
                        cm_data = json.load(f)
                    
                    wt_acc = cm_data['class_accuracy'].get(self.classes[0], 0)
                    ko_acc = cm_data['class_accuracy'].get(self.classes[1], 0)
                    overall_acc = cm_data.get('overall_accuracy', 0)
                    
                    # Calculate composite score if method is composite_score
                    composite_score = None
                    composite_std = None
                    min_class_acc = None
                    if self.ckpt_select_method == 'composite_score':
                        class_accuracies = {self.classes[0]: wt_acc, self.classes[1]: ko_acc}
                        composite_score, composite_std, min_class_acc = self._calculate_composite_score(
                            class_accuracies, overall_acc, penalty_weight=self.penalty_weight
                        )
                    
                    rows.append({
                        'dataset': dataset_num,
                        'test_wt': config['test_wt'],
                        'test_ko': config['test_ko'],
                        'checkpoint': ckpt_name,
                        'epoch': epoch_num,
                        'wt_accuracy': wt_acc,
                        'ko_accuracy': ko_acc,
                        'overall_accuracy': overall_acc,
                        'composite_score': composite_score,
                        'composite_std': composite_std,
                        'min_class_acc': min_class_acc,
                        'selection_method': self.ckpt_select_method if was_filtered else 'none',
                        'max_checkpoints': self.max_ckpts if was_filtered else len(checkpoint_files),
                        'was_filtered': was_filtered,
                        'cm_source': self.cm_source,
                        'cm_file_pattern': self.cm_file_pattern,
                        'split_used': self.split_to_use,
                        'penalty_weight': self.penalty_weight if self.ckpt_select_method == 'composite_score' else None
                    })
                    
                except Exception as e:
                    print(f"  ERROR processing {ckpt_name}: {str(e)}")
                    continue
        
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values(['dataset', 'overall_accuracy'], ascending=[True, False])
            output_path = self.pth_conf_analizer_results / 'used_checkpoints.csv'
            df.to_csv(output_path, index=False)
            print(f"  ✓ Saved used checkpoints report to: {output_path}")
            print(f"  ✓ Exported {len(rows)} checkpoint entries across {len(results)} datasets")
            return True
        else:
            print("  WARNING: No rows to export! No checkpoint data was collected.")
            return False

    def _create_filter_readme(self, output_dir, filtered_images):
        filter_descriptions = {
            'correct': f"Correctly classified in all cases with confidence between {self.min_conf:.0%}-{self.max_conf:.0%}",
            'incorrect': f"Incorrectly classified in all cases with confidence between {self.min_conf:.0%}-{self.max_conf:.0%}",
            'low_confidence': f"Low confidence (below {self.min_conf:.0%}) in all cases",
            'unsure': f"Medium confidence ({self.min_conf:.0%}-{self.max_conf:.0%}) in all cases"
        }
        
        with open(output_dir / "README.txt", 'w') as f:
            f.write("Filtered Image Classification Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Filter type: {self.filter_type}\n")
            f.write(f"CM source: {self.cm_source} (file pattern: {self.cm_file_pattern})\n")
            f.write(f"Split used: {self.split_to_use}\n")
            f.write(f"Description: {filter_descriptions.get(self.filter_type.lower(), 'Custom filter')}\n\n")
            f.write(f"Images meeting criteria: {sum(len(v) for v in filtered_images.values())}\n")
            for class_name, images in filtered_images.items():
                f.write(f"{class_name}: {len(images)} images\n")

    def cleanup_test_folder(self):
        if self.pth_test.exists():
            shutil.rmtree(self.pth_test, ignore_errors=True)

    def analyze_all_datasets(self):
        all_results = {}
        available_datasets = self._get_available_datasets()
        total_datasets = len(available_datasets)
        output_csv = self.pth_conf_analizer_results / 'confidence_analysis.csv'
        
        tqdm.write("Starting confidence analysis...")
        tqdm.write(f"Found {total_datasets} datasets to analyze")
        tqdm.write(f"Training data source: {self.training_data_source}")
        tqdm.write(f"Using {self.cm_source.upper()} confusion matrices for checkpoint selection")
        tqdm.write(f"Using '{self.split_to_use.upper()}' split for analysis")
        tqdm.write(f"Checkpoint selection method: {self.ckpt_select_method}")
        if self.ckpt_select_method == 'composite_score':
            tqdm.write(f"  Composite score penalty weight: {self.penalty_weight}")
            tqdm.write(f"  Min class accuracy threshold: {self.min_class_acc_threshold:.0%}")
        tqdm.write(f"Results will be saved to: {output_csv}")
        
        with tqdm(available_datasets.items(), desc="Processing datasets", position=0, leave=True) as main_pbar:
            for dataset_num, config in main_pbar:
                self._generate_test_set(config['test_wt'], config['test_ko'], current_dataset=dataset_num, total_datasets=total_datasets)
                dataset_results = self._analyze_single_dataset(dataset_num, total_datasets)
                if dataset_results:
                    all_results[dataset_num] = dataset_results
                    if self._save_results_to_csv(all_results, output_csv):
                        tqdm.write(f"Progress saved to: {output_csv}")
                else:
                    tqdm.write(f"Warning: No results for dataset {dataset_num}")
        
        return all_results
    
    def _create_filtered_dataset(self, dataset_num):
        """Create a test dataset filtered to only include images from the specified split"""
        ds = Dataset()
        ds.load_test_dataset()
        
        split_file = self.pth_acv_results / f"dataset_{dataset_num}" / "split_info.json"
        if not split_file.exists():
            return ds.ds_test
        
        with open(split_file, 'r') as f:
            split_info = json.load(f)
        
        # Get images from the appropriate split
        if self.split_to_use == 'validation':
            split_images = set(split_info['validation']['WT'] + split_info['validation']['KO'])
        elif self.split_to_use == 'test':
            split_images = set(split_info['test']['WT'] + split_info['test']['KO'])
            if len(split_images) == 0:
                tqdm.write(f"\n  WARNING: No test images found! Falling back to validation images.")
                split_images = set(split_info['validation']['WT'] + split_info['validation']['KO'])
        else:  # 'all'
            split_images = set(split_info['test']['WT'] + split_info['test']['KO'] + 
                              split_info['validation']['WT'] + split_info['validation']['KO'])
        
        # Filter the dataset samples
        filtered_indices = [
            i for i, sample in enumerate(ds.ds_test.dataset.samples)
            if Path(sample[0]).name in split_images
        ]
        
        if len(filtered_indices) == 0:
            tqdm.write(f"\n  WARNING: No images found for split '{self.split_to_use}'!")
            return ds.ds_test  # Return original as fallback
        
        # Create new dataset with filtered samples
        filtered_dataset = torch.utils.data.dataset.Subset(
            ds.ds_test.dataset,
            indices=filtered_indices
        )
        
        # Create new DataLoader
        filtered_loader = torch.utils.data.DataLoader(
            filtered_dataset,
            batch_size=ds.ds_test.batch_size,
            num_workers=ds.ds_test.num_workers,
            pin_memory=ds.ds_test.pin_memory
        )
        
        tqdm.write(f"\nFiltered dataset: {len(filtered_indices)} images from '{self.split_to_use}' split (originally {len(ds.ds_test.dataset)})")
        return filtered_loader
    
    #############################################################################################################
    # CALL:

    def __call__(self):
        results = self.analyze_all_datasets()
        
        if not results:
            print("No results generated. Analysis failed.")
            return None
        
        tqdm.write(f"\n>> SAVING {self.filter_type.upper()} IMAGES:")
        filtered_images = self.find_filtered_images(results)
        
        total_filtered = sum(len(v) for v in filtered_images.values())
        if total_filtered == 0:
            tqdm.write(f"No images found matching {self.filter_type} filter criteria.")
            tqdm.write(f"Consider adjusting min_conf={self.min_conf:.2f}, max_conf={self.max_conf:.2f}")
            tqdm.write(f"Or check if the split '{self.split_to_use}' contains images.")
        else:
            tqdm.write(f"Found {total_filtered} images matching criteria. Organizing...")
            output_dir = self.organize_filtered_images(filtered_images)
            tqdm.write(f"Filtered images saved to: {output_dir}")

        self._export_used_checkpoints(results)
        self.cleanup_test_folder()
        
        tqdm.write("\nAnalysis complete!\n")
        return results