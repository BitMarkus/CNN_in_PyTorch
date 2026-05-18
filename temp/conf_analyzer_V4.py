"""
CONFIDENCE ANALYZER FOR CROSS-VALIDATION RESULTS

================================================================================
PURPOSE:
Analyzes predictions across ALL cross-validation folds to identify images with
specific confidence patterns. Filters images based on their confidence scores
and correctness across multiple models/datasets.

KEY CONCEPT:
Analyze how EACH IMAGE is predicted across ALL 20 cross-validation folds and
multiple checkpoints within each fold, then filter images based on consistent
patterns.

================================================================================
WHAT THIS SCRIPT DOES:

1. For each of the 20 cross-validation folds:
   - Loads the test set (real images from left-out cell lines)
   - Loads multiple model checkpoints from that fold's training
   - Runs predictions on ALL test images with EACH checkpoint

2. For each individual image across ALL folds and checkpoints:
   - Tracks: true class, predicted class, confidence score
   - Calculates: average confidence, correctness rate (% of times correctly predicted)

3. Filters images based on consistent patterns:
   - HIGH-CONFIDENCE CORRECT: Always correct with high confidence
   - HIGH-CONFIDENCE INCORRECT: Always wrong but model is confident
   - LOW-CONFIDENCE: Model is never confident about these
   - MEDIUM-CONFIDENCE UNSURE: Intermediate confidence (needs review)

4. Organizes filtered images into folders for downstream use.

================================================================================
WORKFLOW EXAMPLE:

For a single image "cell_001.jpg" that appears in:
- Fold 1: WT_1618-02 + KO_1096-01 test set
- Fold 2: WT_JG + KO_1096-01 test set
- ... etc. (whenever its cell line is in test set)

The script tracks predictions from:
- Fold 1: Checkpoint epoch_10.model → 95% confidence, correct
- Fold 1: Checkpoint epoch_20.model → 98% confidence, correct
- Fold 2: Checkpoint epoch_15.model → 92% confidence, correct
- ... etc.

If this image is CORRECT in ALL predictions with confidence >80%,
it gets saved to: ca_results/high_confidence_correct/WT/cell_001_conf95_corr100.jpg

================================================================================
INPUT REQUIREMENTS:

1. COMPLETED CROSS-VALIDATION:
   acv_results/
   ├── dataset_1/
   │   ├── checkpoints/        # Model weights (epoch_XX.model)
   │   ├── plots/
   │   │   └── epoch_XX_cm.json  # Confusion matrix results
   │   └── split_info.json     # Which images were test vs validation
   ├── dataset_2/
   └── ... (20 total)

2. SOURCE IMAGES (based on train_data_source setting):
   
   MODE: "synthetic_only" or "real_only":
   dataset_gen/input_real/      # REAL images for testing
   ├── WT_1618-02/
   │   ├── cell_001.jpg
   │   └── ...
   ├── WT_JG/
   └── ...
   
   MODE: "mixed" (backward compatibility):
   dataset_gen/input/           # MIXED synthetic+real images
   ├── WT_1618-02/
   └── ...

================================================================================
FILTER TYPES AND THEIR USES:

1. "correct" (High-Confidence Correct):
   - Images that are CORRECTLY classified in ALL predictions
   - Confidence ALWAYS between min_conf-max_conf (e.g., 80-100%)
   - USES: Most reliable images for validation sets, publication figures

2. "incorrect" (High-Confidence Errors):
   - Images that are WRONGLY classified in ALL predictions  
   - Confidence ALWAYS between min_conf-max_conf (e.g., 80-100%)
   - USES: Identify systematic errors, problematic cases to fix

3. "low_confidence" (Low-Confidence):
   - Confidence ALWAYS below min_conf (e.g., <80%)
   - Ignores correctness (could be right or wrong)
   - USES: Ambiguous cases, poor-quality images, needs manual review

4. "unsure" (Medium-Confidence):
   - Confidence ALWAYS between min_conf-max_conf (e.g., 50-80%)
   - Ignores correctness
   - USES: Borderline cases, interesting for further analysis

================================================================================
SETTINGS CONFIGURATION (settings.py):

# REQUIRED SETTINGS:
"train_data_source": "synthetic_only",  # or "real_only", "mixed"
"pth_ds_gen_input_real": BASE_DIR / "dataset_gen/input_real/",
"pth_acv_results": BASE_DIR / "acv_results/",
"pth_conf_analizer_results": BASE_DIR / "ca_results/",

# CONFIDENCE THRESHOLDS:
"ca_min_conf": 0.8,     # Minimum confidence (e.g., 80%)
"ca_max_conf": 1.0,     # Maximum confidence (e.g., 100%)
"ca_filter_type": "correct",  # "correct", "incorrect", "low_confidence", "unsure"

# CHECKPOINT SELECTION:
"ca_max_ckpts": 1,      # Max checkpoints per fold to analyze (None = all)
"ca_ckpt_select_method": "balanced_accuracy",  # How to select "best" checkpoints

================================================================================
CHECKPOINT SELECTION METHODS:

When ca_max_ckpts = 3 and a fold has 30 checkpoints, selects top 3 using:

1. "balanced_sum": (WT_acc + KO_acc) - |WT_acc - KO_acc|
   - Rewards balanced performance
   - Penalizes large differences between classes

2. "f1_score": 2 * (WT_acc * KO_acc) / (WT_acc + KO_acc)
   - Harmonic mean of class accuracies
   - Strongly penalizes one poor class

3. "min_difference": min(WT_acc, KO_acc)
   - Most conservative
   - Ensures both classes perform reasonably

4. "balanced_accuracy": sklearn.metrics.balanced_accuracy_score
   - Gold standard for imbalanced data
   - Requires true/predicted labels in JSON

================================================================================
OUTPUT STRUCTURE:

ca_results/
├── confidence_analysis.csv          # Main analysis results
├── used_checkpoints.csv             # Which checkpoints were analyzed
├── high_confidence_correct/         # Filtered images (if filter_type="correct")
│   ├── WT/
│   │   ├── cell_001_conf95_corr100.jpg
│   │   └── ...
│   └── KO/
│       ├── cell_045_conf88_corr100.jpg
│       └── ...
├── high_confidence_incorrect/       # (if filter_type="incorrect")
├── low_confidence/                  # (if filter_type="low_confidence")  
└── medium_confidence_unsure/        # (if filter_type="unsure")

EACH FILTERED IMAGE IS RENAMED WITH:
   OriginalName_confXX_corrYY.jpg
   Where: XX = average confidence percentage
          YY = correctness rate percentage

================================================================================
ANALYSIS DETAILS:

For EACH IMAGE across ALL folds where it appears:

1. HISTORY TRACKING:
   - Which folds included this image in test set
   - Which checkpoints predicted it
   - Prediction results for each checkpoint

2. STATISTICS CALCULATION:
   - Average confidence across all predictions
   - Correctness rate (% of times correctly predicted)
   - Consistency of predictions (all agree on class?)

3. FILTERING LOGIC:
   Image is included if it meets ALL criteria:
   - Appears in at least one test set
   - Has predictions from selected checkpoints
   - Meets confidence/correctness criteria for ALL predictions

================================================================================
SPECIAL HANDLING OF VALIDATION/TEST SPLITS:

If split_info.json exists (when ds_val_from_test_split < 1.0):
- Only uses images marked as "test" in split_info.json
- Excludes images marked as "validation"
- This ensures pure test-set evaluation

If NO split_info.json (when ds_val_from_test_split = 1.0):
- Uses ALL images in test folder
- Since 100% used for validation, all are considered "test"

================================================================================
INTERPRETATION:

- 50%: The model is completely uncertain (equivalent to random guessing).
- <50%: The model is less confident than a coin flip (rare, but possible if predictions are miscalibrated).
- >50%: The model favors one class over the other. Higher values indicate stronger certainty.
- ~100%: The model is extremely confident (may indicate overfitting or easy examples).
"""

import os
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
        
        # NEW: Training data source configuration
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

    # Get only dataset folders that exist in acv_results
    # This way a confidence analysis can be done with only a subset of datasets from a cross validation
    def _get_available_datasets(self):
        datasets = {}
        for idx, (wt, ko) in enumerate(product(self.wt_lines, self.ko_lines), 1):
            dataset_path = self.pth_acv_results / f"dataset_{idx}"
            if dataset_path.exists():
                datasets[idx] = {'test_wt': wt, 'test_ko': ko, 'dataset_idx': idx}
        return datasets 

    # Generates test set directly from input files
    # and copies the images to the data/test folder
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
            source_dir = self.pth_ds_gen_input  # Mixed mode or real folder not available
        
        # Copy images using single shutil.copytree call per class
        if current_dataset is not None and total_datasets is not None:
            tqdm.write(f"\n>> PROCESSING DATASET {current_dataset} OF {total_datasets}:")
        
        tqdm.write(f"Generating test set with WT: {test_wt}, KO: {test_ko}")
        tqdm.write(f"Source directory: {source_dir}")
        
        # Copy WT images
        wt_source = source_dir / test_wt
        if wt_source.exists():
            shutil.copytree(
                wt_source,
                self.pth_test / 'WT',
                dirs_exist_ok=True
            )
            wt_count = len(list(wt_source.glob("*.*")))
            tqdm.write(f"  Copied {wt_count} WT images from {wt_source}")
        else:
            tqdm.write(f"  ⚠️  WARNING: WT source folder not found: {wt_source}")
        
        # Copy KO images
        ko_source = source_dir / test_ko
        if ko_source.exists():
            shutil.copytree(
                ko_source,
                self.pth_test / 'KO',
                dirs_exist_ok=True
            )
            ko_count = len(list(ko_source.glob("*.*")))
            tqdm.write(f"  Copied {ko_count} KO images from {ko_source}")
        else:
            tqdm.write(f"  ⚠️  WARNING: KO source folder not found: {ko_source}")
        
        # Check if any images were copied
        total_images = len(list((self.pth_test / 'WT').glob("*.*"))) + \
                      len(list((self.pth_test / 'KO').glob("*.*")))
        if total_images == 0:
            tqdm.write(f"  ⚠️  CRITICAL: No images copied to test folder!")

    # Analyze a single datasets checkpoints with optional checkpoint selection
    def _analyze_single_dataset(self, dataset_num, total_datasets):
        dataset_results = {}
        checkpoints_path = self.pth_acv_results / f"dataset_{dataset_num}" / 'checkpoints'
        plots_path = self.pth_acv_results / f"dataset_{dataset_num}" / 'plots'
        # Get all checkpoint files that have corresponding confusion matrix JSON files
        checkpoint_files = []
        for f in checkpoints_path.iterdir():
            if f.suffix == '.model' or f.suffix == '.pt':
                # Fixed line - replaced with_stem with compatible version
                json_file = f.parent / f"{f.stem}_cm.json"  # Changed to explicitly use .json
                json_path = plots_path / json_file.name
                if json_path.exists():  # Only consider checkpoints with CM data
                    checkpoint_files.append(f.name)
        # Select checkpoints if needed
        if self.max_ckpts is not None and len(checkpoint_files) > self.max_ckpts:
            checkpoint_files = self._select_checkpoints_by_metric(
                checkpoint_files,
                plots_path,
                top_n=self.max_ckpts,
                method=self.ckpt_select_method
            )
        # Print checkpoint selection info
        tqdm.write(f"Selected {len(checkpoint_files)} checkpoint(s):")
        for checkpoint_file in checkpoint_files:
            base_name = Path(checkpoint_file).stem
            json_file = f"{base_name}_cm.json"
            json_path = plots_path / json_file
            
            try:
                with open(json_path, 'r') as f:
                    cm_data = json.load(f)
                
                if 'class_accuracy' not in cm_data or 'overall_accuracy' not in cm_data:
                    raise ValueError("JSON missing required keys")
                
                wt_acc = cm_data['class_accuracy'].get(self.classes[0], 0)
                ko_acc = cm_data['class_accuracy'].get(self.classes[1], 0)
                overall_acc = cm_data['overall_accuracy']
                
                tqdm.write(f"> {checkpoint_file}:")
                tqdm.write(f"  WT test accuracy: {wt_acc:.2%}")
                tqdm.write(f"  KO test accuracy: {ko_acc:.2%}")
                tqdm.write(f"  Overall test accuracy: {overall_acc:.2%}")
                
            except Exception as e:
                tqdm.write(f"- {checkpoint_file} (Error: {str(e)})")
                tqdm.write(f"  Attempted path: {json_path}")
                if json_path.exists():
                    tqdm.write("  File exists but has unexpected content")
                else:
                    tqdm.write("  File does not exist")
                continue
        
        # Process checkpoints
        with tqdm(
            checkpoint_files,
            desc=f"Dataset {dataset_num}/{total_datasets} - Checkpoints",
            position=1,
            leave=False
        ) as pbar:
            for checkpoint_file in pbar:
                checkpoint_path = checkpoints_path / checkpoint_file
                try:
                    # Load checkpoint
                    tqdm.write(f"Loading checkpoint: {checkpoint_file}")
                    self.cnn.load_state_dict(
                        torch.load(checkpoint_path, map_location=self.device)
                    )
                    self.cnn.eval()
                    # Get predictions
                    confidences = self._get_predictions_with_confidence(dataset_num)
                    dataset_results[checkpoint_file] = self._organize_prediction_results(confidences)
                except Exception as e:
                    tqdm.write(f"\nError processing {checkpoint_file}: {str(e)}")
                    if isinstance(e, RuntimeError) and "CUDA out of memory" in str(e):
                        raise  # Critical error - stop execution
                    continue
                    
        return dataset_results
    
    # Select top checkpoints based on specified metric from confusion matrix JSON files
    def _select_checkpoints_by_metric(self, checkpoint_files, plots_path, top_n=3, method='balanced_sum'):
        scores = []
        
        for checkpoint_file in checkpoint_files:
            # Construct JSON filename by inserting '_cm' before '.json'
            base_name = Path(checkpoint_file).stem  # removes .model
            json_file = f"{base_name}_cm.json"
            json_path = plots_path / json_file
            
            try:
                with open(json_path, 'r') as f:
                    cm_data = json.load(f)
                
                # Verify the expected keys exist
                if 'class_accuracy' not in cm_data or not isinstance(cm_data['class_accuracy'], dict):
                    raise ValueError("Invalid or missing 'class_accuracy' in JSON")  
                if 'overall_accuracy' not in cm_data:
                    raise ValueError("Missing 'overall_accuracy' in JSON")
                
                # Get class accuracies
                wt_acc = cm_data['class_accuracy'].get(self.classes[0], 0)
                ko_acc = cm_data['class_accuracy'].get(self.classes[1], 0)
                overall_acc = cm_data['overall_accuracy']
                
                # Calculate score based on selection method
                # balanced_sum:
                # This rewards models with both high accuracy and balanced performance across classes
                # Effectively gives more weight to the lower of the two accuracies
                # Simple calculation without needing true/predicted labels
                if method == 'balanced_sum':
                    score = (wt_acc + ko_acc) - abs(wt_acc - ko_acc)
                # f1_score:
                # Harmonic mean of the two class accuracies
                # Strongly penalizes large disparities between class performances
                # Can be unstable when one accuracy is very low             
                elif method == 'f1_score':
                    score = 2 * (wt_acc * ko_acc) / (wt_acc + ko_acc) if (wt_acc + ko_acc) > 0 else 0
                # min_difference:
                # Most conservative approach - selects based on worst-performing class
                # Ensures no single class is neglected
                # May reject models with one strong and one moderate class performance   
                elif method == 'min_difference':
                    score = min(wt_acc, ko_acc)
                # balanced_accuracy:
                # Gold standard for imbalanced classification
                # Requires true/predicted labels (not just class accuracies)
                # Calculates the average of recall obtained on each class                    
                elif method == 'balanced_accuracy':
                    # Get true and predicted labels if available
                    if 'true_labels' not in cm_data or 'predicted_labels' not in cm_data:
                        raise ValueError("Missing 'true_labels' or 'predicted_labels' for balanced accuracy calculation")
                    y_true = cm_data['true_labels']
                    y_pred = cm_data['predicted_labels']
                    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
                    score = balanced_accuracy_score(y_true, y_pred)
                else:
                    raise ValueError(f"Unknown selection method: {method}")
                
                scores.append((checkpoint_file, score, wt_acc, ko_acc, overall_acc))
                
            except Exception as e:
                tqdm.write(f"Error processing {json_file}: {str(e)}")
                continue
        
        # Sort by score in descending order and select top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in scores[:top_n]]

    # Get predictions with confidence scores
    def _get_predictions_with_confidence(self, dataset_num):
        # Get filtered test loader
        test_loader = self._create_filtered_dataset(dataset_num)
        
        # Iterate over test images
        confidences = {}
        total_images = len(test_loader.dataset)
        
        with torch.no_grad():
            with autocast(device_type='cuda', enabled=self.device.type == 'cuda'):
                # Create progress bar with proper positioning
                with tqdm(
                    test_loader,
                    desc="Predicting images",
                    total=total_images,
                    position=0,  # Changed to position 0
                    leave=False
                ) as img_pbar:
                    for batch_idx, (images, labels) in enumerate(img_pbar):
                        img_path = test_loader.dataset.dataset.samples[
                            test_loader.dataset.indices[batch_idx]
                        ][0]
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
        # Iteate over confidences
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
    
    # Find images matching the specified confidence filter criteria.
    # Returns dict: {class_name: [(img_path, avg_confidence, correctness_rate)]}
    def find_filtered_images(self, results):

        self._build_image_history(results)
        filtered_images = {cls: [] for cls in self.classes}
        
        for img_path, predictions in self.image_history.items():
            # Get the true class from the original image path
            img_key = Path(img_path).name
            true_class = None
            
            # Determine true class by checking original locations
            # Use appropriate source directory based on training_data_source
            source_dir = self._get_source_directory_for_search()
            
            for wt_line in self.wt_lines:
                if (source_dir / wt_line / img_key).exists():
                    true_class = 'WT'
                    break
                    
            if true_class is None:  # If not found in WT, check KO lines
                for ko_line in self.ko_lines:
                    if (source_dir / ko_line / img_key).exists():
                        true_class = 'KO'
                        break
            
            if true_class is None:
                continue  # Skip if we can't determine origin
                
            # Calculate statistics
            confidences = [p['confidence'] for p in predictions]
            correct = [p['pred_class'] == true_class for p in predictions]  # Compare to true_class
            avg_confidence = sum(confidences) / len(confidences)
            correctness_rate = sum(correct) / len(correct)
            
            # Apply filters
            if self.filter_type.lower() == 'correct':
                # Must be: correct in all cases AND confidence in range AND consistent prediction
                if (all(correct) and 
                    all(self.min_conf <= c <= self.max_conf for c in confidences)):
                    # All predictions should agree on the class
                    pred_classes = {p['pred_class'] for p in predictions}
                    if len(pred_classes) == 1:  # All predictions agree
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
    
    # Helper method to get the appropriate source directory for searching
    def _get_source_directory_for_search(self):
        """Get the source directory to search for original images"""
        if self.training_data_source in ['synthetic_only', 'real_only'] and self.pth_ds_gen_input_real:
            return self.pth_ds_gen_input_real
        return self.pth_ds_gen_input
    
    # Build history of all image predictions
    def _build_image_history(self, results):
        for dataset_num, checkpoints in results.items():
            test_only_images, (test_count, val_count) = self._get_test_only_images(int(dataset_num))
            
            processed_images = 0
            
            for checkpoint_name, pred_data in checkpoints.items():
                for img_path, pred in pred_data['per_image_results'].items():
                    img_name = Path(img_path).name
                    
                    # Skip validation images when split info exists
                    if test_only_images is not None and img_name not in test_only_images:
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
            
            if test_only_images is not None:
                tqdm.write(
                    f"Dataset {dataset_num}: Processed {processed_images} test images, "
                    f"skipped {val_count} validation images"
                )

    # Find original image path by checking all possible locations
    def _find_original_image_path(self, img_key):
        """Find original image in the appropriate source directory"""
        
        # Use appropriate source directory based on training_data_source
        source_dir = self._get_source_directory_for_search()
        
        for line in self.wt_lines + self.ko_lines:
            candidate = source_dir / line / img_key
            if candidate.exists():
                return str(candidate)
        
        # Fallback: check mixed folder
        for line in self.wt_lines + self.ko_lines:
            candidate = self.pth_ds_gen_input / line / img_key
            if candidate.exists():
                return str(candidate)
        
        raise FileNotFoundError(f"Original image not found for {img_key}")

    # Organize filtered images into appropriate folder structure
    def organize_filtered_images(self, filtered_images):

        # Create appropriate output subdirectory based on filter type
        output_subdir = {
            'correct': "high_confidence_correct",
            'incorrect': "high_confidence_incorrect", 
            'low_confidence': "low_confidence",
            'unsure': "medium_confidence_unsure"
        }.get(self.filter_type.lower(), "filtered_images")
        
        output_dir = self.pth_conf_analizer_results / output_subdir
        
        # Track copied files to prevent duplicates
        copied_files = set()
        
        for class_name in self.classes:
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path, avg_confidence, correctness_rate in filtered_images[class_name]:
                if img_path in copied_files:
                    continue
                    
                try:
                    # Create new filename with confidence and correctness info
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

    # Save analysis results to CSV
    def _save_results_to_csv(self, results, output_path):
        rows = []
        for dataset_num, checkpoints in results.items():
            for ckpt_name, pred_data in checkpoints.items():
                for cls, metrics in pred_data['aggregated_stats'].items():
                    if not metrics['all_confidences']:
                        continue
                    # Add rows    
                    rows.append({
                        'dataset': dataset_num,
                        'checkpoint': ckpt_name,
                        'class': cls,
                        'mean_confidence': sum(metrics['all_confidences'])/len(metrics['all_confidences']),
                        'accuracy': len(metrics['correct'])/len(metrics['all_confidences']),
                        'total_samples': len(metrics['all_confidences']),
                        'correct_predictions': len(metrics['correct'])
                    })
        # Write rows to csv. file
        if rows:
            pd.DataFrame(rows).to_csv(output_path, index=False)
            return True
        return False
    
    # Export a CSV file listing all used checkpoints with their accuracies
    def _export_used_checkpoints(self, results):
        rows = []
        for dataset_num, checkpoints in results.items():
            config = self._get_available_datasets()[int(dataset_num)]
            checkpoint_files = list(checkpoints.keys())
            was_filtered = self.max_ckpts is not None and len(checkpoint_files) > self.max_ckpts
            
            for ckpt_name in checkpoint_files:
                # Get the corresponding JSON file for accuracy metrics
                base_name = Path(ckpt_name).stem
                json_file = f"{base_name}_cm.json"
                json_path = (
                    self.pth_acv_results / 
                    f"dataset_{dataset_num}" / 
                    'plots' / 
                    json_file
                )
                
                try:
                    with open(json_path, 'r') as f:
                        cm_data = json.load(f)
                    
                    wt_acc = cm_data['class_accuracy'].get(self.classes[0], 0)
                    ko_acc = cm_data['class_accuracy'].get(self.classes[1], 0)
                    overall_acc = cm_data['overall_accuracy']
                    
                    rows.append({
                        'dataset': dataset_num,
                        'test_wt': config['test_wt'],
                        'test_ko': config['test_ko'],
                        'checkpoint': ckpt_name,
                        'wt_accuracy': wt_acc,
                        'ko_accuracy': ko_acc,
                        'overall_accuracy': overall_acc,
                        'selection_method': self.ckpt_select_method if was_filtered else 'none',
                        'max_checkpoints': self.max_ckpts if was_filtered else len(checkpoint_files),
                        'was_filtered': was_filtered
                    })
                    
                except Exception as e:
                    tqdm.write(f"Error loading metrics for {ckpt_name}: {str(e)}")
                    continue
        
        if rows:
            df = pd.DataFrame(rows)
            # Sort by dataset and overall accuracy
            df = df.sort_values(['dataset', 'overall_accuracy'], ascending=[True, False])
            output_path = self.pth_conf_analizer_results / 'used_checkpoints.csv'
            df.to_csv(output_path, index=False)
            tqdm.write(f"Saved used checkpoints report to: {output_path}")
            return True
        return False

    # Create README file explaining the filtering criteria
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
            f.write(f"Description: {filter_descriptions.get(self.filter_type.lower(), 'Custom filter')}\n\n")
            f.write(f"Images meeting criteria: {sum(len(v) for v in filtered_images.values())}\n")
            for class_name, images in filtered_images.items():
                f.write(f"{class_name}: {len(images)} images\n")

    # Clean up the test folder after analysis
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
        
        # Check if any datasets use test-only filtering
        test_only_count = sum(
            1 for dataset_num in available_datasets 
            if (self.pth_acv_results / f"dataset_{dataset_num}" / "split_info.json").exists()
        )
        if test_only_count > 0:
            tqdm.write(f"Note: {test_only_count} datasets will use test-only images (excluding validation)")
        tqdm.write(f"Results will be saved to: {output_csv}")
        
        # Use context manager for main progress bar
        with tqdm(
            available_datasets.items(),
            desc="Processing datasets",
            position=0,
            leave=True
        ) as main_pbar:
            for dataset_num, config in main_pbar:
                # Generate test set for current dataset
                self._generate_test_set(
                    config['test_wt'], 
                    config['test_ko'],
                    current_dataset=dataset_num,
                    total_datasets=total_datasets
                )
                # Analyze dataset with optional checkpoint selection
                dataset_results = self._analyze_single_dataset(dataset_num, total_datasets)
                all_results[dataset_num] = dataset_results
                # Save interim results
                if self._save_results_to_csv(all_results, output_csv):
                    tqdm.write(f"Progress saved to: {output_csv}")
                else:
                    tqdm.write("Warning: No results to save in current batch")
        return all_results
    
    # Returns set of test-only image names if split info exists, else None
    def _get_test_only_images(self, dataset_num):
        split_file = self.pth_acv_results / f"dataset_{dataset_num}" / "split_info.json"
        if not split_file.exists():
            tqdm.write(f"Dataset {dataset_num}: No split info found - will use all available images")
            return None, (0, 0)  # Return None and (0,0) for counts
        
        try:
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            
            test_images = set(split_info['test']['WT'] + split_info['test']['KO'])
            val_images = set(split_info['validation']['WT'] + split_info['validation']['KO'])
            
            test_count = len(test_images)
            val_count = len(val_images)
            
            tqdm.write(f"Dataset {dataset_num}: Using {test_count} test-only images (excluding {val_count} validation)")
            return test_images, (test_count, val_count)
            
        except Exception as e:
            tqdm.write(f"Error loading split info for dataset {dataset_num}: {str(e)}")
            return None, (0, 0)
        
    def _create_filtered_dataset(self, dataset_num):
        """Create a test dataset filtered to only include images in the test split"""
        # Load original test dataset
        ds = Dataset()
        ds.load_test_dataset()
        
        # Get test-only images from split info
        split_file = self.pth_acv_results / f"dataset_{dataset_num}" / "split_info.json"
        if not split_file.exists():
            return ds.ds_test  # Return original if no split info
        
        with open(split_file, 'r') as f:
            split_info = json.load(f)
        test_images = set(split_info['test']['WT'] + split_info['test']['KO'])
        
        # Filter the dataset samples
        filtered_indices = [
            i for i, sample in enumerate(ds.ds_test.dataset.samples)
            if Path(sample[0]).name in test_images
        ]
        
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
        
        # Use tqdm.write() for cleaner output
        tqdm.write(f"\nFiltered dataset: {len(test_images)} test images (originally {len(ds.ds_test.dataset)})", end=' ')
        return filtered_loader
    
    #############################################################################################################
    # CALL:

    # Main analysis method
    def __call__(self):

        # Get results from all datasets
        results = self.analyze_all_datasets()
        
        if not results:
            print("No results generated. Analysis failed.")
            return None
        
        # Find and organize filtered images
        tqdm.write(f"\n>> SAVING {self.filter_type.upper()} IMAGES:")
        filtered_images = self.find_filtered_images(results)
        
        if not any(filtered_images.values()):
            tqdm.write(f"No images found matching {self.filter_type} filter criteria.")
            tqdm.write(f"Consider adjusting min_conf={self.min_conf:.2f}, max_conf={self.max_conf:.2f}")
        else:
            tqdm.write(f"Organizing {self.filter_type} images...")
            output_dir = self.organize_filtered_images(filtered_images)
            tqdm.write(f"Filtered images saved to: {output_dir}")

        # Export checkpoint usage report
        self._export_used_checkpoints(results)
        
        # Cleanup test folder when done
        self.cleanup_test_folder()
        
        tqdm.write("\nAnalysis complete!\n")
        return results