
# Filter types in settings file:
# Filter Type	    Confidence Range	    Considers Correctness?	        Typical Use Case
# correct	        [min_conf, max_conf]	Must be correct in all folds	Reliable predictions
# incorrect	        [min_conf, max_conf]	Must be wrong in all folds	    High-confidence errors
# low_confidence	Below min_conf	        Ignores correctness	            Ambiguous/poor-quality images
# unsure	        [min_conf, max_conf]	Ignores correctness	            Intermediate-confidence cases

# Confidence interpretation
# 50%:	The model is completely uncertain (equivalent to random guessing).
# <50%:	The model is less confident than a coin flip (rare, but possible if predictions are miscalibrated).
# >50%:	The model favors one class over the other. Higher values indicate stronger certainty.
# ~100%:The model is extremely confident (may indicate overfitting or easy examples).

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
        self.pth_ds_gen_input = Path(setting['pth_ds_gen_input']).absolute()
        self.pth_test = Path(setting['pth_test']).absolute()
        self.pth_conf_analizer_results = Path(setting['pth_conf_analizer_results']).absolute()
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
        # Clear and recreate test folder
        shutil.rmtree(self.pth_test, ignore_errors=True)
        for cls in self.classes:
            (self.pth_test / cls).mkdir(parents=True)
        # Copy images using single shutil.copytree call per class
        if current_dataset is not None and total_datasets is not None:
            tqdm.write(f"\n>> PROCESSING DATASET {current_dataset} OF {total_datasets}:")
        tqdm.write(f"Generating test set with WT: {test_wt}, KO: {test_ko}")
        shutil.copytree(
            self.pth_ds_gen_input / test_wt,
            self.pth_test / 'WT',
            dirs_exist_ok=True
        )
        shutil.copytree(
            self.pth_ds_gen_input / test_ko,
            self.pth_test / 'KO',
            dirs_exist_ok=True
        )

    # Analyze a single datasets checkpoints with optional checkpoint selection
    def _analyze_single_dataset(self, dataset_num, total_datasets):
        dataset_results = {}
        checkpoints_path = self.pth_acv_results / f"dataset_{dataset_num}" / 'checkpoints'
        plots_path = self.pth_acv_results / f"dataset_{dataset_num}" / 'plots'
        # Get all checkpoint files that have corresponding confusion matrix JSON files
        checkpoint_files = []
        for f in checkpoints_path.iterdir():
            if f.suffix == '.model':
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
                    confidences = self._get_predictions_with_confidence()
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
    def _get_predictions_with_confidence(self):
        # Load test dataset
        ds = Dataset()
        ds.load_test_dataset()
        # Iterate over test images
        confidences = {}
        total_images = len(ds.ds_test.dataset)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Mixed precision if using CUDA
                # Add progress bar for image predictions
                with tqdm(
                    ds.ds_test,
                    desc="Predicting images",
                    total=total_images,
                    position=2,
                    leave=False
                ) as img_pbar:
                    for batch_idx, (images, labels) in enumerate(img_pbar):
                        img_path = ds.ds_test.dataset.samples[batch_idx][0]
                        images = images.to(self.device)
                        if images.dim() == 3:
                            images = images.unsqueeze(0)
                        # Make prediction - fixed model access
                        outputs = self.cnn(images)
                        # Use softmax to get confidence
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        max_prob, pred_idx = torch.max(probs, 1)
                        # Add data to confidences dict
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
    
    def find_filtered_images(self, results):
        """Find images matching the specified confidence filter criteria.
        
        Returns:
            dict: {class_name: [(img_path, avg_confidence, correctness_rate)]}
        """
        self._build_image_history(results)
        filtered_images = {cls: [] for cls in self.classes}
        
        for img_path, predictions in self.image_history.items():
            # Calculate average confidence and correctness rate
            confidences = [p['confidence'] for p in predictions]
            correct = [p['pred_class'] == p['true_class'] for p in predictions]
            avg_confidence = sum(confidences) / len(confidences)
            correctness_rate = sum(correct) / len(correct)
            true_class = predictions[0]['true_class']
            
            # Apply filters based on filter type
            if self.filter_type.lower() == 'correct':
                # Correct in all cases AND within confidence range
                if all(correct) and all(self.min_conf <= c <= self.max_conf for c in confidences):
                    filtered_images[true_class].append((img_path, avg_confidence, correctness_rate))
                    
            elif self.filter_type.lower() == 'incorrect':
                # Incorrect in all cases AND within confidence range
                if not any(correct) and all(self.min_conf <= c <= self.max_conf for c in confidences):
                    filtered_images[true_class].append((img_path, avg_confidence, correctness_rate))
                    
            elif self.filter_type.lower() == 'low_confidence':
                # Below min confidence in all cases
                if all(c < self.min_conf for c in confidences):
                    filtered_images[true_class].append((img_path, avg_confidence, correctness_rate))
                    
            elif self.filter_type.lower() == 'unsure':
                # In middle confidence range (could be correct or incorrect)
                if all(self.min_conf <= c <= self.max_conf for c in confidences):
                    filtered_images[true_class].append((img_path, avg_confidence, correctness_rate))
        
        return filtered_images
    
    # Build history of all image predictions
    def _build_image_history(self, results):
        for dataset_num, checkpoints in results.items():
            config = self._get_available_datasets()[int(dataset_num)]
            for checkpoint_name, pred_data in checkpoints.items():
                for img_path, pred in pred_data['per_image_results'].items():
                    try:
                        original_path = self._find_original_image_path(Path(img_path).name)
                        self.image_history[original_path].append({
                            'dataset': dataset_num,
                            'checkpoint': checkpoint_name,
                            **pred
                        })
                    except FileNotFoundError as e:
                        tqdm.write(f"Warning: {str(e)}")
                        continue

    # Find original image path by checking all possible locations
    def _find_original_image_path(self, img_key):
        for line in self.wt_lines + self.ko_lines:
            candidate = self.pth_ds_gen_input / line / img_key
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(f"Original image not found for {img_key}")


    def organize_filtered_images(self, filtered_images):
        """Organize filtered images into appropriate folder structure."""
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

    # Create README file for output folder
    def _create_filter_readme(self, output_dir, filtered_images):
        """Create README file explaining the filtering criteria."""
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
        # Path and name of output csv file
        output_csv = self.pth_conf_analizer_results / 'confidence_analysis.csv'
        
        tqdm.write("Starting confidence analysis...")
        tqdm.write(f"Found {total_datasets} datasets to analyze")
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
    
    #############################################################################################################
    # CALL:

    # Main analysis method
    def __call__(self):
        # Get results from all datasets
        results = self.analyze_all_datasets()
        
        # Find and organize filtered images
        tqdm.write(f"\n>> SAVING {self.filter_type.upper()} IMAGES:")
        filtered_images = self.find_filtered_images(results)
        
        tqdm.write(f"Organizing {self.filter_type} images...")
        output_dir = self.organize_filtered_images(filtered_images)
        tqdm.write(f"Filtered images saved to: {output_dir}")

        # Export checkpoint usage report
        self._export_used_checkpoints(results)
        
        # Cleanup test folder when done
        self.cleanup_test_folder()
        
        tqdm.write("\nAnalysis complete!\n")
        return results