from custom_model import Custom_CNN_Model
from dataset import Dataset
import os
import shutil
import torch
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from itertools import product
import functions as fn

class ConfidenceAnalyzer:
    def __init__(self, device, model, base_paths, wt_lines, ko_lines):
        """
        Initialize analyzer with on-the-fly dataset generation capability.
        Only processes datasets that exist in acv_results folder.
        """
        self.device = device
        self.model = model
        self.base_paths = base_paths
        self.classes = ['KO', 'WT']
        self.wt_lines = wt_lines
        self.ko_lines = ko_lines
        os.makedirs(base_paths['test_folder'], exist_ok=True)
        os.makedirs(base_paths['output_root'], exist_ok=True)
        self.image_history = defaultdict(list)

    # --------------------------
    # Main Analysis Methods
    # --------------------------

    def _get_available_datasets(self):
        """Get only dataset folders that exist in acv_results with their original indices."""
        available = {}
        
        # First get all possible configs
        all_configs = self._get_dataset_configs()
        
        # Check which ones actually exist
        for config in all_configs:
            dataset_num = config['dataset_idx']
            dataset_path = os.path.join(self.base_paths['acv_results'], f"dataset_{dataset_num}")
            if os.path.exists(dataset_path):
                available[dataset_num] = config
                
        return available
    
    def analyze_all_datasets(self, output_csv=None):
        """Main method that only processes available datasets."""
        all_results = {}
        available_datasets = self._get_available_datasets()
        
        for dataset_num, config in tqdm(available_datasets.items(), desc="Processing datasets"):
            test_wt = config['test_wt']
            test_ko = config['test_ko']
            
            # Generate test set on-the-fly
            self._generate_test_set(test_wt, test_ko)
            dataset_results = self._analyze_single_dataset(dataset_num, test_wt, test_ko)
            all_results[dataset_num] = dataset_results
            
        if output_csv:
            self._save_results_to_csv(all_results, output_csv)

        self._print_results_summary(all_results)
        return all_results
    
    def _generate_test_set(self, test_wt, test_ko):
        """Generate test set directly from input files."""
        # Clear existing test folder
        if os.path.exists(self.base_paths['test_folder']):
            shutil.rmtree(self.base_paths['test_folder'])
        
        # Create fresh test folders
        os.makedirs(os.path.join(self.base_paths['test_folder'], 'WT'), exist_ok=True)
        os.makedirs(os.path.join(self.base_paths['test_folder'], 'KO'), exist_ok=True)

        # Copy WT test images
        wt_src = os.path.join(self.base_paths['dataset_gen_input'], test_wt)
        wt_dest = os.path.join(self.base_paths['test_folder'], 'WT')
        for img in os.listdir(wt_src):
            shutil.copy2(os.path.join(wt_src, img), wt_dest)

        # Copy KO test images
        ko_src = os.path.join(self.base_paths['dataset_gen_input'], test_ko)
        ko_dest = os.path.join(self.base_paths['test_folder'], 'KO')
        for img in os.listdir(ko_src):
            shutil.copy2(os.path.join(ko_src, img), ko_dest)
    
    def find_high_confidence_images(self, results, min_confidence=0.95):
        """
        Find images that were consistently correctly classified with high confidence
        in ALL checkpoints OF THE DATASETS WHERE THEY APPEARED.
        """
        # First build complete image history from results
        self._build_image_history(results)
        
        consistent_images = {cls: [] for cls in self.classes}
        
        for img_key, predictions in self.image_history.items():
            # All predictions must be correct and high-confidence
            all_good = all(
                (p['pred_class'] == p['true_class']) and 
                (p['confidence'] >= min_confidence)
                for p in predictions
            )
            
            if all_good and predictions:
                true_class = predictions[0]['true_class']
                avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
                original_path = self._find_original_image_path(img_key)
                consistent_images[true_class].append((original_path, avg_confidence))
        
        return consistent_images
    
    def _build_image_history(self, results):
        """Modified version using dataset_gen_input"""
        for dataset_num, checkpoints in results.items():
            # Get the config for this dataset to find which cell lines were used
            config = self._get_available_datasets()[int(dataset_num)]
            test_wt = config['test_wt']
            test_ko = config['test_ko']
            
            for checkpoint_name, pred_data in checkpoints.items():
                for img_path, pred in pred_data['per_image_results'].items():
                    img_key = os.path.basename(img_path)
                    self.image_history[img_key].append({
                        'dataset': dataset_num,
                        'checkpoint': checkpoint_name,
                        'true_class': pred['true_class'],
                        'pred_class': pred['pred_class'],
                        'confidence': pred['confidence'],
                        'test_wt': test_wt,  # Store which cell line was used
                        'test_ko': test_ko
                    })

    def _get_all_test_image_paths(self, test_data_path):
        """Get paths of all test images in a dataset."""
        paths = []
        for class_name in self.classes:
            class_dir = os.path.join(test_data_path, class_name)
            if os.path.exists(class_dir):
                paths.extend([
                    os.path.join(class_dir, fname) 
                    for fname in os.listdir(class_dir)
                ])
        return paths

    def _find_original_image_path(self, img_key):
        """Find original in input folder"""
        for wt in self.wt_lines:
            candidate = os.path.join(self.base_paths['dataset_gen_input'], wt, img_key)
            if os.path.exists(candidate):
                return candidate
        for ko in self.ko_lines:
            candidate = os.path.join(self.base_paths['dataset_gen_input'], ko, img_key)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Original image not found for {img_key}")
    
    def organize_high_confidence_images(self, consistent_images, output_subdir="high_confidence_examples"):
        """
        Copy high-confidence images to organized folder structure with prediction info in filenames.
        
        Creates folder structure:
        output_root/
            high_confidence_examples/
                WT/
                    image1_WT_95.jpg
                    image2_WT_97.jpg
                KO/
                    image3_KO_96.jpg
                    image4_KO_98.jpg
        """
        output_dir = os.path.join(self.base_paths['output_root'], output_subdir)
        
        # Create class folders
        for class_name in self.classes:
            os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
        
        # Copy images with prediction info in filenames
        for class_name, image_info in consistent_images.items():
            print(f"\nCopying {len(image_info)} high-confidence {class_name} images")
            
            for img_path, confidence in tqdm(image_info, desc=f"Copying {class_name}"):
                try:
                    # Get base filename without extension
                    base, ext = os.path.splitext(os.path.basename(img_path))
                    
                    # Format confidence as percentage without decimals
                    confidence_pct = int(round(confidence * 100))
                    
                    # Create new filename: originalname_predclass_confidence.ext
                    new_filename = f"{base}_{class_name}_{confidence_pct}{ext}"
                    dest_path = os.path.join(output_dir, class_name, new_filename)
                    
                    # Handle potential duplicates
                    counter = 1
                    while os.path.exists(dest_path):
                        new_filename = f"{base}_{class_name}_{confidence_pct}_{counter}{ext}"
                        dest_path = os.path.join(output_dir, class_name, new_filename)
                        counter += 1
                    
                    shutil.copy2(img_path, dest_path)
                except Exception as e:
                    print(f"Error copying {img_path}: {str(e)}")
        
        print(f"\nAll high-confidence images organized in: {output_dir}")
        self._create_readme(output_dir, consistent_images)
        
        return output_dir

    # --------------------------
    # Core Processing Methods
    # --------------------------
    
    def _analyze_single_dataset(self, dataset_num, test_wt, test_ko):
        """Analyze a single dynamically generated dataset."""
        dataset_results = {}
        checkpoints_path = os.path.join(
            self.base_paths['acv_results'], 
            f"dataset_{dataset_num}", 
            'checkpoints'
        )
        
        checkpoint_files = [f for f in os.listdir(checkpoints_path) 
                          if f.endswith('.model')]
        
        for checkpoint_file in tqdm(checkpoint_files, 
                                 desc=f"Dataset {dataset_num} - Checkpoints"):
            checkpoint_path = os.path.join(checkpoints_path, checkpoint_file)
            
            try:
                self.model.load_state_dict(torch.load(checkpoint_path, 
                                                   map_location=self.device))
                self.model.eval()
                confidences = self._get_predictions_with_confidence()
                dataset_results[checkpoint_file] = self._organize_prediction_results(
                    confidences, test_wt, test_ko)
            except Exception as e:
                print(f"Error loading {checkpoint_path}: {str(e)}")
                continue
        
        return dataset_results
    
    def _get_predictions_with_confidence(self, test_image_paths=None):
        """Get predictions with optional pre-loaded test paths."""
        ds = Dataset()
        ds.pth_test = self.base_paths['test_folder']
        ds.load_prediction_dataset()

        confidences = {}
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(ds.ds_pred):
                img_path = ds.ds_pred.dataset.samples[batch_idx][0]
                
                images = images.to(self.device)
                # Ensure proper input dimensions
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                
                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                max_prob, pred_idx = torch.max(probs, 1)
                
                true_class = self.classes[labels.item()]
                pred_class = self.classes[pred_idx.item()]
                confidence = max_prob.item()
                
                confidences[img_path] = (true_class, pred_class, confidence)
        
        return confidences

    # --------------------------
    # Helper Methods
    # --------------------------
 
    def _get_dataset_folders(self):
        """Get sorted list of dataset folders."""
        return sorted(
            [d for d in os.listdir(self.base_paths['acv_results']) 
             if d.startswith('dataset') and os.path.isdir(os.path.join(self.base_paths['acv_results'], d))],
            key=lambda x: int(x.split('_')[-1])
        )
    
    def _prepare_test_data(self, test_data_path):
        """Clear and prepare test folder with new data."""
        # Remove existing test folder if it exists
        if os.path.exists(self.base_paths['test_folder']):
            shutil.rmtree(self.base_paths['test_folder'])
        
        # Create fresh directory and copy data
        os.makedirs(self.base_paths['test_folder'], exist_ok=True)
        for class_name in self.classes:
            src = os.path.join(test_data_path, class_name)
            dst = os.path.join(self.base_paths['test_folder'], class_name)
            if os.path.exists(src):
                shutil.copytree(src, dst)
    
    def _organize_prediction_results(self, confidences, test_wt, test_ko):
        """Organize results with test line metadata."""
        per_image = {}
        aggregated = {cls: {'all_confidences': [], 'correct': [], 'incorrect': [],
                          'test_line': test_wt if cls == 'WT' else test_ko}
                     for cls in self.classes}

        for img_path, (true_class, pred_class, confidence) in confidences.items():
            img_key = os.path.basename(img_path)
            per_image[img_key] = {
                'true_class': true_class,
                'pred_class': pred_class,
                'confidence': confidence,
                'test_line': test_wt if true_class == 'WT' else test_ko
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
    
    def cleanup_test_folder(self):
        """Clears the test folder after analysis is complete."""
        test_folder = self.base_paths['test_folder']
        
        if os.path.exists(test_folder):
            print(f"\nCleaning up test folder: {test_folder}")
            try:
                # Remove all contents but keep the folder itself
                for filename in os.listdir(test_folder):
                    file_path = os.path.join(test_folder, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
                        
                print("Test folder cleanup complete")
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
        else:
            print(f"Test folder {test_folder} does not exist")

    # --------------------------
    # Output Methods
    # --------------------------

    def _get_dataset_configs(self):
        """Generate all possible configs (same as DatasetGenerator)."""
        return [
            {"test_wt": wt, "test_ko": ko, "dataset_idx": idx}
            for idx, (wt, ko) in enumerate(product(self.wt_lines, self.ko_lines), 1)
        ]
    
    def _print_results_summary(self, results):
        """Print summary of analysis results."""
        print("\nResults summary:")
        for dataset_num, checkpoints in results.items():
            print(f"Dataset {dataset_num}: {len(checkpoints)} checkpoints")
            for ckpt, data in checkpoints.items():
                wt_count = len(data['aggregated_stats']['WT']['all_confidences'])
                ko_count = len(data['aggregated_stats']['KO']['all_confidences'])
                print(f"  {ckpt}: WT={wt_count} KO={ko_count}")
    
    def _print_prediction_stats(self, confidences):
        """Print prediction statistics."""
        if not confidences:
            return
            
        print("\nFirst 5 predictions:")
        for i, (path, (true, pred, conf)) in enumerate(list(confidences.items())[:5]):
            print(f"{i+1}. {true}->{pred} (conf: {conf:.2f}) | {path}")

        print("\nPrediction statistics:")
        all_confs = [c[2] for c in confidences.values()]
        print(f"Avg confidence: {sum(all_confs)/len(all_confs):.2f}")
        print(f"Max confidence: {max(all_confs):.2f}")
        print(f"Min confidence: {min(all_confs):.2f}")
    
    def _save_results_to_csv(self, results, output_path):
        """Save analysis results to CSV file."""
        rows = []
        
        for dataset_num, checkpoints in results.items():
            for ckpt_name, pred_data in checkpoints.items():
                # Access the aggregated stats
                aggregated_stats = pred_data['aggregated_stats']
                for cls, metrics in aggregated_stats.items():
                    row = {
                        'dataset': dataset_num,
                        'checkpoint': ckpt_name,
                        'class': cls,
                        'mean_confidence': sum(metrics['all_confidences'])/len(metrics['all_confidences']) if metrics['all_confidences'] else 0,
                        'min_confidence': min(metrics['all_confidences']) if metrics['all_confidences'] else 0,
                        'max_confidence': max(metrics['all_confidences']) if metrics['all_confidences'] else 0,
                        'accuracy': len(metrics.get('correct', [])) / len(metrics['all_confidences']) if metrics['all_confidences'] else 0,
                        'total_samples': len(metrics['all_confidences']),
                        'correct_predictions': len(metrics.get('correct', [])),
                        'incorrect_predictions': len(metrics.get('incorrect', []))
                    }
                    rows.append(row)
        
        if not rows:
            print("WARNING: No results to save!")
            return

        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    def _create_readme(self, output_dir, consistent_images):
        """Create README file explaining the folder structure."""
        readme_path = os.path.join(output_dir, "README.txt")
        with open(readme_path, 'w') as f:
            f.write("High-Confidence Image Classification Results\n")
            f.write("="*50 + "\n\n")
            f.write("Folder contains images that were consistently classified:\n")
            f.write("- With correct class prediction\n")
            f.write("- With high confidence (>95%)\n")
            f.write("- Across ALL checkpoints and ALL datasets\n\n")
            f.write("Folder structure:\n")
            f.write("WT/: Wildtype cells meeting all criteria\n")
            f.write("KO/: Knockout cells meeting all criteria\n\n")
            f.write("Counts:\n")
            for class_name, images in consistent_images.items():
                f.write(f"{class_name}: {len(images)} images\n")

# Example usage:
if __name__ == "__main__":
    # Configuration
    base_paths = {
        'acv_results': './acv_results',
        'dataset_gen_input': './dataset_gen/input',  # New required path
        'test_folder': './data/test',
        'output_root': './analysis_results'
    }
    
    # Convert to absolute paths
    base_paths = {k: os.path.abspath(v) for k,v in base_paths.items()}
    print("Resolved paths:", base_paths)
    
    # Initialize device and model
    device = fn.show_cuda_and_versions()
    cnn = Custom_CNN_Model()
    cnn.model = cnn.load_model(device)
    
    # Verify model
    """
    print("\nModel validation:")
    test_tensor = torch.randn(1, 1, 512, 512).to(device)
    try:
        out = cnn.model(test_tensor)
        print(f"Model test output shape: {out.shape}")
    except Exception as e:
        print(f"Model validation failed: {str(e)}")
        exit()
    """

    analyzer = ConfidenceAnalyzer(
        device=device,
        model=cnn.model,
        base_paths=base_paths,
        wt_lines=["WT_1618-02", "WT_JG", "WT_JT", "WT_KM", "WT_MS"], 
        ko_lines=["KO_1096-01", "KO_1618-01", "KO_BR2986", "KO_BR3075"]         
    )

    results = analyzer.analyze_all_datasets(
        output_csv=os.path.join(base_paths['output_root'], 'confidence_analysis.csv')
    )
    
    # Find and organize high-confidence images
    consistent_images = analyzer.find_high_confidence_images(results, min_confidence=0.90)
    analyzer.organize_high_confidence_images(consistent_images)

    # Cleanuptest folder when done
    analyzer.cleanup_test_folder()