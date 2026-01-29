"""
ClassAnalyzer - Image Classification Analysis and Diagnostic Tool

This script analyzes images using a trained CNN model to predict their classes and
provides detailed statistics about the predictions. It includes comprehensive logit
analysis to understand model confidence beyond softmax probabilities.

Key Features:
1. Batch prediction on entire directories of images organized by class folders
2. Comprehensive per-folder statistics including:
   - Total image count per folder
   - Distribution of predictions across all classes
   - Percentage breakdown of predictions
   - Most likely class for each folder
3. Advanced logit analysis:
   - Raw logit values for each prediction (not just softmax probabilities)
   - Maximum logit per image (indicates model's raw confidence)
   - Logit range (shows how distinct the prediction is)
   - Count of uncertain predictions (max_logit < 0)
   - Full logit distribution statistics
4. Optional image renaming with confidence scores and logit values
5. Supports both 2-class (WT/KO) and 9-class (cell lines) configurations
6. Interactive checkpoint selection for model loading
7. CSV export with comprehensive logit statistics
8. JSON export of detailed logit distributions

Why Logit Analysis Matters:
Softmax can be misleading! Example scenario:
• Logits: [-2.0, -1.8, -1.9] → all negative (model is uncertain)
• Softmax: [0.25, 0.45, 0.3] → shows "45% confidence" (misleading!)
• Max logit: -1.8 < 0 → reveals true uncertainty

Logit Interpretation:
• max_logit > 0: Model thinks image belongs to this class (positive evidence)
• max_logit > 2: Model is quite confident
• max_logit > 5: Model is very confident
• max_logit < 0: Model is uncertain (negative evidence)
• max_logit < -2: Strong evidence AGAINST this class

Usage Workflow:
1. Organize images in folders by their true classes (e.g., WT_JG, KO_1618-01)
2. Configure settings in settings.py:
   - Set pth_prediction to your image directory
   - Set classes to your classification scheme
   - Set analyze_rename_with_confidence (True/False)
   - Set analyze_include_logits_in_rename (True/False) - NEW
3. Run the script to analyze predictions
4. Review the generated CSV and JSON files with detailed statistics
5. Optionally use renamed images for downstream analysis

Output Files:
1. results_{checkpoint_name}.csv - Main analysis results with per-folder statistics
   - Now includes logit statistics: Avg_Max_Logit, Avg_Logit_Range, etc.
2. logit_statistics_{checkpoint_name}.json - Detailed logit distribution analysis
3. Images optionally renamed with confidence and logit information

Integration with Fibroblast Project:
• Used after synthetic image generation to evaluate quality at the logit level
• Helps identify which synthetic images are most biologically plausible
• Provides critical data for the dual-CNN filtering pipeline
• Supports leave-one-pair-out cross-validation experiments
• Essential for validating the mother-child relationship preservation
• Helps compare different training strategies (Step 7000 vs Step 8000 checkpoints)

Settings Configuration:
In settings.py, ensure these settings are configured:
    'pth_prediction': Path('path/to/images'),  # Directory with class folders
    'pth_checkpoint': Path('path/to/checkpoints'),  # Directory with model checkpoints
    'classes': ['WT_1618-02', 'WT_JG', ..., 'KO_BR3075'],  # Your classes
    'analyze_rename_with_confidence': False,  # Enable/disable image renaming
    'analyze_include_logits_in_rename': False,  # NEW: Include logits in filenames
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Subset
import json
# Own modules
from dataset import Dataset
from settings import setting
from model import CNN_Model

class ClassAnalyzer:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device):
        self.device = device
        self.pth_prediction = setting['pth_prediction'].resolve()
        self.pth_checkpoint = setting['pth_checkpoint'].resolve()
        self.classes = setting['classes']

        # File naming options - UPDATED to match class_sorter.py
        self.rename_with_confidence = setting['analyze_rename_with_confidence']
        # NEW: Option to include logits in renamed filenames
        self.include_logits_in_rename = setting.get('analyze_include_logits_in_rename', False)
        
        # Validate setting types
        if not isinstance(self.rename_with_confidence, bool):
            raise ValueError(f"analyze_rename_with_confidence must be boolean, got {type(self.rename_with_confidence)}")
        
        if not isinstance(self.include_logits_in_rename, bool):
            raise ValueError(f"analyze_include_logits_in_rename must be boolean, got {type(self.include_logits_in_rename)}")
        
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

        self.checkpoint_loaded = False
        self.loaded_checkpoint_name = None

    #############################################################################################################
    # METHODS:

    # Load model weights from a selected checkpoint
    def load_checkpoint(self):
        # First get checkpoints without printing table
        silent_checkpoints = self.cnn_wrapper.print_checkpoints_table(self.pth_checkpoint, print_table=False)
        
        if not silent_checkpoints:
            print("The checkpoint folder is empty!")
            return False
        
        # If only one checkpoint exists
        if len(silent_checkpoints) == 1:
            # Extract filename from the tuple
            checkpoint_file = silent_checkpoints[0][1]  # (id, name) -> get name
            print(f"\nFound single checkpoint: {checkpoint_file}")
            print("Loading automatically...")
        else:
            # Show interactive table for multiple checkpoints
            self.cnn_wrapper.print_checkpoints_table(self.pth_checkpoint)  # prints table
            checkpoint_file = self.cnn_wrapper.select_checkpoint(silent_checkpoints, "Select a checkpoint: ")
            if not checkpoint_file:
                return False
        
        try:
            full_path = self.pth_checkpoint / checkpoint_file
            self.cnn.load_state_dict(torch.load(full_path))
            self.checkpoint_loaded = True
            self.loaded_checkpoint_name = full_path.stem
            print(f"Successfully loaded weights from {checkpoint_file}")
            return True
        except FileNotFoundError as e:
            print(f"\nError loading checkpoint: {str(e)}")
            print(f"Full path attempted: {full_path}")
            return False
        except Exception as e:
            print(f"\nError loading checkpoint: {str(e)}")
            return False
        
    # Return indices of samples belonging to a specific folder
    def get_folder_indices(self, dataset, folder_name):
        return [i for i, (path, _) in enumerate(dataset.samples) 
                if Path(path).parent.name == folder_name]
    
    # Predict images from a specific folder with comprehensive logit analysis
    def predict_folder(self, folder_name):
        folder_indices = self.get_folder_indices(self.ds.ds_pred.dataset, folder_name)
        if not folder_indices:
            print(f"No images found for folder: {folder_name}")
            return None

        subset = Subset(self.ds.ds_pred.dataset, folder_indices)
        loader = DataLoader(subset, batch_size=self.ds.batch_size_pred)

        class_counts = {class_name: 0 for class_name in self.classes}
        
        # For individual image tracking and renaming
        image_predictions = []
        
        # NEW: Collect logit statistics
        all_max_logits = []
        all_logit_ranges = []
        all_logits_data = []  # Store all logits for detailed analysis
        
        self.cnn.eval()
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Predicting {folder_name}", unit="img")
            for batch_idx, (images, _) in enumerate(pbar):
                outputs = self.cnn(images.to(self.device))
                
                # Get raw logits for analysis
                raw_logits = outputs.cpu().numpy()
                
                # Calculate softmax probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                # Process each image in the batch
                for i in range(len(images)):
                    pred_class_idx = predicted[i].item()
                    confidence = confidences[i].item()
                    class_name = self.classes[pred_class_idx]
                    
                    # Get the original image path
                    sample_idx = folder_indices[batch_idx * self.ds.batch_size_pred + i]
                    original_path = self.ds.ds_pred.dataset.samples[sample_idx][0]
                    
                    # Calculate logit statistics
                    current_logits = raw_logits[i]
                    max_logit = current_logits.max()
                    min_logit = current_logits.min()
                    logit_range = max_logit - min_logit
                    
                    # Store logit statistics
                    all_max_logits.append(max_logit)
                    all_logit_ranges.append(logit_range)
                    all_logits_data.append({
                        'logits': current_logits.tolist(),
                        'max_logit': float(max_logit),
                        'predicted_class': class_name
                    })
                    
                    # Store prediction info with logits
                    image_predictions.append({
                        'original_path': original_path,
                        'predicted_class': class_name,
                        'confidence': confidence,
                        'confidence_percentage': int(confidence * 100),
                        # NEW: Logit information
                        'max_logit': float(max_logit),
                        'min_logit': float(min_logit),
                        'logit_range': float(logit_range),
                        'all_logits': current_logits.tolist(),
                        'is_uncertain': max_logit < 0  # Flag uncertain predictions
                    })
                    
                    # Count for overall statistics
                    class_counts[class_name] += 1
                
                pbar.set_postfix({
                    'total': len(folder_indices),
                    **{k: v for k, v in class_counts.items() if v > 0}
                })

        # Rename images if requested - UPDATED variable name
        if self.rename_with_confidence:
            self._rename_images_with_confidence(image_predictions, folder_name)

        total = len(folder_indices)
        
        # Calculate logit statistics
        uncertain_count = sum(1 for ml in all_max_logits if ml < 0)
        uncertain_percentage = (uncertain_count / total * 100) if total > 0 else 0
        
        # Calculate confidence statistics
        confidence_values = [img['confidence'] for img in image_predictions]
        avg_confidence = np.mean(confidence_values) if confidence_values else 0
        
        # Create comprehensive results dictionary
        results = {
            'Folder': folder_name,
            'Total Images': total,
            # Class prediction statistics
            **{f"{k}_Count": v for k, v in class_counts.items()},
            **{f"{k}_Percentage": (v/total)*100 for k, v in class_counts.items()},
            'Most Likely Class': max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else "None",
            # Confidence statistics
            'Avg_Confidence': avg_confidence,
            # NEW: Logit statistics
            'Avg_Max_Logit': float(np.mean(all_max_logits)) if all_max_logits else 0,
            'Std_Max_Logit': float(np.std(all_max_logits)) if all_max_logits else 0,
            'Min_Max_Logit': float(np.min(all_max_logits)) if all_max_logits else 0,
            'Max_Max_Logit': float(np.max(all_max_logits)) if all_max_logits else 0,
            'Avg_Logit_Range': float(np.mean(all_logit_ranges)) if all_logit_ranges else 0,
            'Uncertain_Images_Count': uncertain_count,
            'Uncertain_Images_Percentage': uncertain_percentage,
            'Logit_Health_Score': self._calculate_logit_health_score(all_max_logits)
        }
        
        # Store detailed logit data for JSON export
        self.folder_logit_data = {
            'folder_name': folder_name,
            'all_max_logits': [float(ml) for ml in all_max_logits],
            'all_logit_ranges': [float(lr) for lr in all_logit_ranges],
            'detailed_logits': all_logits_data,
            'image_predictions': image_predictions
        }
        
        return results
    
    # Calculate a health score for logit distribution
    def _calculate_logit_health_score(self, max_logits):
        """Calculate a score (0-100) indicating health of logit distribution"""
        if not max_logits:
            return 0
        
        max_logits_array = np.array(max_logits)
        
        # Penalize negative logits (uncertain predictions)
        negative_penalty = np.sum(max_logits_array < 0) / len(max_logits_array) * 50
        
        # Reward positive, moderate logits (2-10 is ideal)
        positive_reward = np.sum((max_logits_array >= 2) & (max_logits_array <= 10)) / len(max_logits_array) * 30
        
        # Penalize extreme logits (>20 can indicate overconfidence)
        extreme_penalty = np.sum(max_logits_array > 20) / len(max_logits_array) * 20
        
        # Base score
        base_score = 50
        
        # Calculate final score
        health_score = base_score - negative_penalty + positive_reward - extreme_penalty
        
        return max(0, min(100, health_score))

    # Rename images by adding confidence percentage and optionally logits to filename
    def _rename_images_with_confidence(self, image_predictions, folder_name):
        # Early return if renaming is disabled
        if not self.rename_with_confidence:
            print(f"Renaming disabled for folder '{folder_name}'")
            return
        
        renamed_count = 0
        
        for pred_info in image_predictions:
            original_path = Path(pred_info['original_path'])
            confidence_pct = pred_info['confidence_percentage']
            predicted_class = pred_info['predicted_class']
            max_logit = pred_info['max_logit']
            
            # Create new filename
            original_stem = original_path.stem.rstrip('_') # remove eventual underscores from the filename
            
            if self.include_logits_in_rename:
                # Include both confidence and logit in filename
                new_stem = f"{original_stem}_conf{confidence_pct}_logit{max_logit:.1f}-{predicted_class}"
            else:
                # Include only confidence in filename
                new_stem = f"{original_stem}_conf{confidence_pct}-{predicted_class}"
            
            new_filename = f"{new_stem}{original_path.suffix}"
            new_path = original_path.parent / new_filename
            
            try:
                # Rename the file
                original_path.rename(new_path)
                renamed_count += 1
            except Exception as e:
                # Silent fail on rename errors
                pass
        
        print(f"Successfully renamed {renamed_count}/{len(image_predictions)} images in folder '{folder_name}'")
        if self.include_logits_in_rename:
            print(f"  (logits included in filenames)")

    # Save detailed logit statistics to JSON file
    def _save_logit_statistics(self, results_df):
        """Save comprehensive logit statistics to JSON file"""
        
        if not hasattr(self, 'all_folder_logit_data'):
            return
        
        logit_stats = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'checkpoint_used': self.loaded_checkpoint_name,
            'total_folders_analyzed': len(self.all_folder_logit_data),
            'classes': self.classes,
            'logit_interpretation': {
                'max_logit_greater_than_0': 'Model thinks image belongs to this class',
                'max_logit_less_than_0': 'Model is uncertain or thinks it does NOT belong',
                'max_logit_greater_than_2': 'Model is quite confident',
                'max_logit_greater_than_5': 'Model is very confident'
            },
            'per_folder_statistics': {}
        }
        
        # Add per-folder statistics
        for folder_data in self.all_folder_logit_data:
            folder_name = folder_data['folder_name']
            
            if folder_data['all_max_logits']:
                max_logits = np.array(folder_data['all_max_logits'])
                
                logit_stats['per_folder_statistics'][folder_name] = {
                    'total_images': len(max_logits),
                    'max_logit_statistics': {
                        'mean': float(np.mean(max_logits)),
                        'median': float(np.median(max_logits)),
                        'std': float(np.std(max_logits)),
                        'min': float(np.min(max_logits)),
                        'max': float(np.max(max_logits)),
                        'percentile_25': float(np.percentile(max_logits, 25)),
                        'percentile_75': float(np.percentile(max_logits, 75))
                    },
                    'uncertainty_analysis': {
                        'images_with_negative_max_logit': int(np.sum(max_logits < 0)),
                        'percentage_negative': float(np.sum(max_logits < 0) / len(max_logits) * 100),
                        'images_with_max_logit_between_0_and_2': int(np.sum((max_logits >= 0) & (max_logits < 2))),
                        'images_with_max_logit_above_5': int(np.sum(max_logits >= 5))
                    },
                    'confidence_vs_logit_correlation': {
                        'avg_softmax_confidence': float(results_df.loc[results_df['Folder'] == folder_name, 'Avg_Confidence'].iloc[0] 
                                                        if not results_df.empty and 'Avg_Confidence' in results_df.columns else 0),
                        'avg_max_logit': float(np.mean(max_logits))
                    }
                }
        
        # Add cross-folder comparisons
        all_max_logits_combined = []
        for folder_data in self.all_folder_logit_data:
            all_max_logits_combined.extend(folder_data['all_max_logits'])
        
        if all_max_logits_combined:
            all_max_logits_array = np.array(all_max_logits_combined)
            logit_stats['overall_statistics'] = {
                'total_images_analyzed': len(all_max_logits_array),
                'overall_max_logit_mean': float(np.mean(all_max_logits_array)),
                'overall_uncertain_images_percentage': float(np.sum(all_max_logits_array < 0) / len(all_max_logits_array) * 100),
                'health_assessment': {
                    'excellent': f"{(np.sum(all_max_logits_array >= 2) / len(all_max_logits_array) * 100):.1f}% images with good confidence (max_logit ≥ 2)",
                    'concerning': f"{(np.sum(all_max_logits_array < 0) / len(all_max_logits_array) * 100):.1f}% uncertain images (max_logit < 0)",
                    'overconfident': f"{(np.sum(all_max_logits_array > 20) / len(all_max_logits_array) * 100):.1f}% possibly overconfident (max_logit > 20)"
                }
            }
        
        # Save to JSON file
        json_path = self.pth_prediction / f"logit_statistics_{self.loaded_checkpoint_name}.json"
        with open(json_path, 'w') as f:
            json.dump(logit_stats, f, indent=2)
        
        print(f"Saved detailed logit statistics to: {json_path}")
        return json_path

    # Analyze all folders in prediction directory with comprehensive logit analysis
    def analyze_prediction_folder(self):
        if not self.load_checkpoint():
            print("WARNING: Using untrained weights!")
            self.loaded_checkpoint_name = "untrained"

        # Get ALL folders in prediction directory
        all_folders = [d.name for d in self.pth_prediction.iterdir() if d.is_dir()]
        
        if not all_folders:
            print(f"No folders found in {self.pth_prediction}")
            return None

        print(f"\nAnalyzing {len(all_folders)} folders...")
        print(f"Model checkpoint: {self.loaded_checkpoint_name}")
        if self.rename_with_confidence:
            print(f"Image renaming: ENABLED (logits in names: {'YES' if self.include_logits_in_rename else 'NO'})")
        else:
            print("Image renaming: DISABLED")
        
        results = []
        self.all_folder_logit_data = []  # Store logit data for all folders
        
        for folder in all_folders:
            try:
                print(f"\n{'='*60}")
                print(f"> Processing folder: {folder}")
                print(f"{'='*60}")
                
                result = self.predict_folder(folder)
                if result:
                    results.append(result)
                    
                    # Store logit data for this folder
                    if hasattr(self, 'folder_logit_data'):
                        self.all_folder_logit_data.append(self.folder_logit_data)
                    
                    # Print folder results with logit info
                    print(f"\nRESULTS for {folder}:")
                    print(f"  Total images: {result['Total Images']}")
                    print(f"  Most likely class: {result['Most Likely Class']}")
                    print(f"  Avg confidence: {result['Avg_Confidence']:.3f}")
                    print(f"  Logit statistics:")
                    print(f"    • Avg max logit: {result['Avg_Max_Logit']:.2f}")
                    print(f"    • Logit range: {result['Avg_Logit_Range']:.2f}")
                    print(f"    • Uncertain images: {result['Uncertain_Images_Count']} ({result['Uncertain_Images_Percentage']:.1f}%)")
                    print(f"    • Logit health score: {result['Logit_Health_Score']:.1f}/100")
                    
                    # Print class distribution for top classes
                    print(f"  Class distribution (top 3):")
                    class_percentages = [(cls, result[f'{cls}_Percentage']) for cls in self.classes]
                    class_percentages.sort(key=lambda x: x[1], reverse=True)
                    for cls, pct in class_percentages[:3]:
                        if pct > 0:
                            print(f"    • {cls}: {result[f'{cls}_Count']} images ({pct:.1f}%)")
                    
            except Exception as e:
                print(f"Error processing {folder}: {str(e)}")
                import traceback
                traceback.print_exc()

        if not results:
            print("No valid results generated")
            return None

        # Save results to CSV
        df = pd.DataFrame(results)
        output_path = self.pth_prediction / f"results_{self.loaded_checkpoint_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"Saved results to: {output_path}")
        
        # Save detailed logit statistics to JSON
        logit_json_path = self._save_logit_statistics(df)
        
        # Print overall summary
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE - SUMMARY")
        print(f"{'='*60}")
        print(f"Total folders analyzed: {len(results)}")
        print(f"Total images analyzed: {df['Total Images'].sum()}")
        print(f"Overall avg confidence: {df['Avg_Confidence'].mean():.3f}")
        print(f"Overall avg max logit: {df['Avg_Max_Logit'].mean():.2f}")
        print(f"Overall uncertain images: {df['Uncertain_Images_Count'].sum()} ({(df['Uncertain_Images_Count'].sum() / df['Total Images'].sum() * 100):.1f}%)")
        print(f"Average logit health score: {df['Logit_Health_Score'].mean():.1f}/100")
        
        # Identify folders with potential issues
        concerning_folders = df[df['Logit_Health_Score'] < 50]
        if not concerning_folders.empty:
            print(f"\n⚠️  FOLDERS WITH POTENTIAL ISSUES (health score < 50):")
            for _, row in concerning_folders.iterrows():
                print(f"  • {row['Folder']}: score={row['Logit_Health_Score']:.1f}, "
                      f"avg_logit={row['Avg_Max_Logit']:.2f}, "
                      f"{row['Uncertain_Images_Percentage']:.1f}% uncertain")
        
        print(f"\nDetailed logit analysis saved to: {logit_json_path}")
        
        return df