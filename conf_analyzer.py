from custom_model import Custom_CNN_Model
from dataset import Dataset
import os
import shutil
import torch
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from itertools import product
import json
# Own modules
from model import CNN_Model
from custom_model import Custom_CNN_Model
import functions as fn
from settings import setting

class ConfidenceAnalyzer:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device):

        # Passed parameters
        self.device = device
        # Settings parameters
        # Paths
        self.pth_acv_results = os.path.abspath(setting['pth_acv_results'])
        self.pth_ds_gen_input = os.path.abspath(setting['pth_ds_gen_input'])
        self.pth_test = os.path.abspath(setting['pth_test'])
        self.pth_conf_analizer_results = os.path.abspath(setting['pth_conf_analizer_results'])
        # Classes and cell lines
        self.classes = setting['classes']
        self.wt_lines = setting['wt_lines']
        self.ko_lines = setting['ko_lines']
        # Min conf for sorting
        self.min_conf = setting['ca_min_conf']
        # Maximum number of checkpoints which for a dataset
        self.max_ckpts = setting['ca_max_ckpts']
        # Selection method for selecting the "best" checkpoints
        self.ckpt_select_method = setting['ca_ckpt_select_method']

        # Objects
        # Create model wrapper
        if setting["cnn_type"] == "custom":
            self.cnn = Custom_CNN_Model()  # Custom_CNN_Model is both wrapper AND model
            self.cnn.model = self.cnn      # Make model reference point to itself
        else:
            self.cnn = CNN_Model()         # Standard wrapper
            self.cnn.model = self.cnn.load_model(self.device)  # Load actual model

        # Ensure everything is on the right device
        self.cnn.model = self.cnn.model.to(self.device)
        if hasattr(self.cnn, 'to'):  # Move wrapper if it's a nn.Module
            self.cnn = self.cnn.to(self.device)

        # Create required directories
        os.makedirs(self.pth_test, exist_ok=True)
        os.makedirs(self.pth_conf_analizer_results, exist_ok=True)
        # List of confidences for each testing for each image
        self.image_history = defaultdict(list)

    #############################################################################################################
    # METHODS:

    # Get only dataset folders that exist in acv_results
    # This way a confidence analysis can be done with only a subset of datasets from a cross validation
    def _get_available_datasets(self):
        datasets = {}
        for idx, (wt, ko) in enumerate(product(self.wt_lines, self.ko_lines), 1):
            dataset_path = os.path.join(self.pth_acv_results, f"dataset_{idx}")
            if os.path.exists(dataset_path):
                datasets[idx] = {'test_wt': wt, 'test_ko': ko, 'dataset_idx': idx}
        return datasets 

    # Generates test set directly from input files
    # and copies the images to the data/test folder
    def _generate_test_set(self, test_wt, test_ko, current_dataset=None, total_datasets=None):

        # Clear and recreate test folder
        shutil.rmtree(self.pth_test, ignore_errors=True)
        for cls in self.classes:
            os.makedirs(os.path.join(self.pth_test, cls))
        # Copy images using single shutil.copytree call per class
        if current_dataset is not None and total_datasets is not None:
            tqdm.write(f"\n>> PROCESSING DATASET {current_dataset} OF {total_datasets}:")
        tqdm.write(f"Generating test set with WT: {test_wt}, KO: {test_ko}")
        shutil.copytree(
            os.path.join(self.pth_ds_gen_input, test_wt),
            os.path.join(self.pth_test, 'WT'),
            dirs_exist_ok=True
        )
        shutil.copytree(
            os.path.join(self.pth_ds_gen_input, test_ko),
            os.path.join(self.pth_test, 'KO'),
            dirs_exist_ok=True
        )

    # Analyze a single datasets checkpoints with optional checkpoint selection
    def _analyze_single_dataset(self, dataset_num, total_datasets):
        dataset_results = {}
        checkpoints_path = os.path.join(
            self.pth_acv_results, 
            f"dataset_{dataset_num}", 
            'checkpoints'
        )
        plots_path = os.path.join(self.pth_acv_results, f"dataset_{dataset_num}", 'plots')
        
        # Get all checkpoint files that have corresponding confusion matrix JSON files
        checkpoint_files = []
        for f in os.listdir(checkpoints_path):
            if f.endswith('.model'):
                json_file = f.replace('.model', '_cm.json')
                json_path = os.path.join(plots_path, json_file)
                if os.path.exists(json_path):  # Only consider checkpoints with CM data
                    checkpoint_files.append(f)
        
        # Store checkpoint info for reporting
        selected_checkpoints_info = []
        
        # Select checkpoints based on the specified method if max_checkpoints is set
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
            base_name = os.path.splitext(checkpoint_file)[0]
            json_file = f"{base_name}_cm.json"
            json_path = os.path.join(plots_path, json_file)
            
            try:
                with open(json_path, 'r') as f:
                    cm_data = json.load(f)
                
                # Verify structure
                if 'class_accuracy' not in cm_data or 'overall_accuracy' not in cm_data:
                    raise ValueError("JSON missing required keys")
                
                # Get accuracies
                wt_acc = cm_data['class_accuracy'].get(self.classes[0], 0)
                ko_acc = cm_data['class_accuracy'].get(self.classes[1], 0)
                overall_acc = cm_data['overall_accuracy']
                
                tqdm.write(f"> {checkpoint_file}:")
                tqdm.write(f"  WT test accuracy: {wt_acc:.2%}")
                tqdm.write(f"  KO test accuracy: {ko_acc:.2%}")
                tqdm.write(f"  Overall test accuracy: {overall_acc:.2%}")
                
            except Exception as e:
                tqdm.write(f"- {checkpoint_file} (Error: {str(e)})")
                # Print the full path that was attempted
                tqdm.write(f"  Attempted path: {json_path}")
                if os.path.exists(json_path):
                    tqdm.write("  File exists but has unexpected content")
                else:
                    tqdm.write("  File does not exist")
                continue
        
        # Use context manager for clean progress bar handling
        with tqdm(
            checkpoint_files,
            desc=f"Dataset {dataset_num}/{total_datasets} - Checkpoints",
            position=1,  # Nested position
            leave=False  # Auto-remove when done
        ) as pbar:
            for checkpoint_file in pbar:
                checkpoint_path = os.path.join(checkpoints_path, checkpoint_file)
                try:
                    # Load checkpoint
                    tqdm.write(f"Loading checkpoint: {checkpoint_file}")
                    self.cnn.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                    # Set network to evaluation mode
                    self.cnn.model.eval()
                    # Make predictions on test images and get confidences
                    confidences = self._get_predictions_with_confidence()
                    # Save all results for one dataset
                    dataset_results[checkpoint_file] = self._organize_prediction_results(confidences)
                except Exception as e:
                    tqdm.write(f"\nError loading {checkpoint_path}: {str(e)}")
                    continue                
        return dataset_results
    
    # Select top checkpoints based on specified metric from confusion matrix JSON files
    def _select_checkpoints_by_metric(self, checkpoint_files, plots_path, top_n=3, method='balanced_sum'):
        scores = []
        
        for checkpoint_file in checkpoint_files:
            # Construct JSON filename by inserting '_cm' before '.json'
            base_name = os.path.splitext(checkpoint_file)[0]  # removes .model
            json_file = f"{base_name}_cm.json"
            json_path = os.path.join(plots_path, json_file)
            
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
                if method == 'balanced_sum':
                    score = (wt_acc + ko_acc) - abs(wt_acc - ko_acc)
                elif method == 'f1_score':
                    score = 2 * (wt_acc * ko_acc) / (wt_acc + ko_acc) if (wt_acc + ko_acc) > 0 else 0
                elif method == 'min_difference':
                    score = min(wt_acc, ko_acc)
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
        total_images = len(ds.ds_pred.dataset)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Mixed precision if using CUDA
                # Add progress bar for image predictions
                with tqdm(
                    ds.ds_pred,
                    desc="Predicting images",
                    total=total_images,
                    position=2,
                    leave=False
                ) as img_pbar:
                    for batch_idx, (images, labels) in enumerate(img_pbar):
                        img_path = ds.ds_pred.dataset.samples[batch_idx][0]
                        images = images.to(self.device)
                        if images.dim() == 3:
                            images = images.unsqueeze(0)
                        # Make prediction - fixed model access
                        outputs = self.cnn.model(images)
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
            img_key = os.path.basename(img_path)
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
    
    # Find consistently correctly classified high-confidence images
    def find_high_confidence_images(self, results):
        self._build_image_history(results)
        consistent_images = {cls: [] for cls in self.classes}
        # Iterate over image history and sort out images, which
        # 1) are correctly classified
        # 2) have a confidence > min_conf
        # in all datasets and checkpoints they were part of
        for img_key, predictions in self.image_history.items():
            if all((p['pred_class'] == p['true_class']) and 
                  (p['confidence'] >= self.min_conf) 
                  for p in predictions):
                true_class = predictions[0]['true_class']
                avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
                consistent_images[true_class].append((self._find_original_image_path(img_key), avg_confidence))
        return consistent_images
    
    # Build history of all image predictions
    def _build_image_history(self, results):
        for dataset_num, checkpoints in results.items():
            config = self._get_available_datasets()[int(dataset_num)]
            for checkpoint_name, pred_data in checkpoints.items():
                for img_path, pred in pred_data['per_image_results'].items():
                    self.image_history[os.path.basename(img_path)].append({
                        'dataset': dataset_num,
                        'checkpoint': checkpoint_name,
                        **pred
                    })

    # Find original image path by checking all possible locations
    def _find_original_image_path(self, img_key):
        for line in self.wt_lines + self.ko_lines:
            candidate = os.path.join(self.pth_ds_gen_input, line, img_key)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Original image not found for {img_key}")

    # Organize high-confidence images into folder structure
    def organize_high_confidence_images(self, consistent_images):
        # Folder for high confidence images, separated by class
        output_subdir="high_confidence_examples"
        output_dir = os.path.join(self.pth_conf_analizer_results, output_subdir)
        # Iterate over each class and make directories
        for class_name in self.classes:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            # Iterate over images
            for img_path, confidence in tqdm(
                consistent_images[class_name],
                desc=f"Copying {class_name}",
                leave=True
            ):
                try:
                    base, ext = os.path.splitext(os.path.basename(img_path))
                    confidence_pct = int(round(confidence * 100))
                    new_filename = f"{base}_{class_name}_{confidence_pct}{ext}"
                    dest_path = os.path.join(class_dir, new_filename)
                    # Handle duplicates (not really necessary as all images suppose to have unique names)
                    counter = 1
                    while os.path.exists(dest_path):
                        new_filename = f"{base}_{class_name}_{confidence_pct}_{counter}{ext}"
                        dest_path = os.path.join(class_dir, new_filename)
                        counter += 1
                    # Copy images to high conf folder
                    shutil.copy2(img_path, dest_path)
                except Exception as e:
                    tqdm.write(f"\nError copying {img_path}: {str(e)}")
        # Create readme
        self._create_readme(output_dir, consistent_images)
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
            
            for ckpt_name, pred_data in checkpoints.items():
                # Get the corresponding JSON file for accuracy metrics
                base_name = os.path.splitext(ckpt_name)[0]
                json_file = f"{base_name}_cm.json"
                json_path = os.path.join(
                    self.pth_acv_results,
                    f"dataset_{dataset_num}",
                    'plots',
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
                        'selection_method': 'top3_by_balanced_sum'  # or make this dynamic
                    })
                    
                except Exception as e:
                    tqdm.write(f"Error loading metrics for {ckpt_name}: {str(e)}")
                    continue
        
        if rows:
            df = pd.DataFrame(rows)
            # Sort by dataset and overall accuracy
            df = df.sort_values(['dataset', 'overall_accuracy'], ascending=[True, False])
            output_path = os.path.join(self.pth_conf_analizer_results, 'used_checkpoints.csv')
            df.to_csv(output_path, index=False)
            tqdm.write(f"\nSaved used checkpoints report to: {output_path}")
            return True
        return False    

    # Create README file for output folder
    def _create_readme(self, output_dir, consistent_images):
        with open(os.path.join(output_dir, "README.txt"), 'w') as f:
            f.write("High-Confidence Image Classification Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Images meeting confidence threshold: {sum(len(v) for v in consistent_images.values())}\n")
            for class_name, images in consistent_images.items():
                f.write(f"{class_name}: {len(images)} images\n")

    # Clean up the test folder after analysis
    def cleanup_test_folder(self):
        if os.path.exists(self.pth_test):
            shutil.rmtree(self.pth_test, ignore_errors=True)

    def analyze_all_datasets(self):
        all_results = {}
        available_datasets = self._get_available_datasets()
        total_datasets = len(available_datasets)
        # Path and name of output csv file
        output_csv = os.path.join(self.pth_conf_analizer_results, 'confidence_analysis.csv')
        
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
        
        # Find and organize high-confidence images
        tqdm.write("\n>> Finding high-confidence images...")
        consistent_images = self.find_high_confidence_images(results)
        
        tqdm.write("Organizing high-confidence images...")
        output_dir = self.organize_high_confidence_images(consistent_images)
        tqdm.write(f"High-confidence images saved to: {output_dir}")

        # Export checkpoint usage report
        self._export_used_checkpoints(results)
        
        # Cleanup test folder when done
        self.cleanup_test_folder()
        
        tqdm.write("\nAnalysis complete!\n")
        return results