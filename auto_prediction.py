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
        self.device = device
        self.model = model
        self.base_paths = {k: os.path.abspath(v) for k,v in base_paths.items()}
        self.classes = ['KO', 'WT']
        self.wt_lines = wt_lines
        self.ko_lines = ko_lines
        
        # Create required directories
        os.makedirs(self.base_paths['test_folder'], exist_ok=True)
        os.makedirs(self.base_paths['output_root'], exist_ok=True)
        self.image_history = defaultdict(list)

    def _get_available_datasets(self):
        """Get only dataset folders that exist in acv_results"""
        available = {}
        for idx, (wt, ko) in enumerate(product(self.wt_lines, self.ko_lines), 1):
            dataset_path = os.path.join(self.base_paths['acv_results'], f"dataset_{idx}")
            if os.path.exists(dataset_path):
                available[idx] = {'test_wt': wt, 'test_ko': ko, 'dataset_idx': idx}
        return available
    
    def analyze_all_datasets(self, output_csv=None):
        """Main analysis method processing only available datasets"""
        all_results = {}
        available_datasets = self._get_available_datasets()
        
        for dataset_num, config in tqdm(available_datasets.items(), desc="Processing datasets"):
            self._generate_test_set(config['test_wt'], config['test_ko'])
            dataset_results = self._analyze_single_dataset(dataset_num)
            all_results[dataset_num] = dataset_results
            
        if output_csv:
            self._save_results_to_csv(all_results, output_csv)

        return all_results
    
    def _generate_test_set(self, test_wt, test_ko):
        """Generate test set directly from input files"""
        test_folder = self.base_paths['test_folder']
        
        # Clear and recreate test folder
        shutil.rmtree(test_folder, ignore_errors=True)
        for cls in self.classes:
            os.makedirs(os.path.join(test_folder, cls))
            
        # Copy images using single shutil.copytree call per class
        shutil.copytree(
            os.path.join(self.base_paths['dataset_gen_input'], test_wt),
            os.path.join(test_folder, 'WT'),
            dirs_exist_ok=True
        )
        shutil.copytree(
            os.path.join(self.base_paths['dataset_gen_input'], test_ko),
            os.path.join(test_folder, 'KO'),
            dirs_exist_ok=True
        )
    
    def _analyze_single_dataset(self, dataset_num):
        """Analyze a single dataset's checkpoints"""
        dataset_results = {}
        checkpoints_path = os.path.join(
            self.base_paths['acv_results'], 
            f"dataset_{dataset_num}", 
            'checkpoints'
        )
        
        for checkpoint_file in tqdm(
            [f for f in os.listdir(checkpoints_path) if f.endswith('.model')],
            desc=f"Dataset {dataset_num} - Checkpoints"
        ):
            checkpoint_path = os.path.join(checkpoints_path, checkpoint_file)
            try:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                self.model.eval()
                confidences = self._get_predictions_with_confidence()
                dataset_results[checkpoint_file] = self._organize_prediction_results(confidences)
            except Exception as e:
                print(f"Error loading {checkpoint_path}: {str(e)}")
                continue
        
        return dataset_results
    
    def _get_predictions_with_confidence(self):
        """Get predictions with confidence scores"""
        ds = Dataset()
        ds.pth_test = self.base_paths['test_folder']
        ds.load_prediction_dataset()

        confidences = {}
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(ds.ds_pred):
                img_path = ds.ds_pred.dataset.samples[batch_idx][0]
                images = images.to(self.device)
                
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                
                outputs = self.model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                max_prob, pred_idx = torch.max(probs, 1)
                
                confidences[img_path] = (
                    self.classes[labels.item()],
                    self.classes[pred_idx.item()],
                    max_prob.item()
                )
        
        return confidences

    def _organize_prediction_results(self, confidences):
        """Organize prediction results into structured format"""
        per_image = {}
        aggregated = {cls: {'all_confidences': [], 'correct': [], 'incorrect': []} 
                     for cls in self.classes}

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

    def find_high_confidence_images(self, results, min_confidence=0.95):
        """Find consistently correctly classified high-confidence images"""
        self._build_image_history(results)
        consistent_images = {cls: [] for cls in self.classes}
        
        for img_key, predictions in self.image_history.items():
            if all((p['pred_class'] == p['true_class']) and 
                  (p['confidence'] >= min_confidence) 
                  for p in predictions):
                true_class = predictions[0]['true_class']
                avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
                consistent_images[true_class].append((self._find_original_image_path(img_key), avg_confidence))
        
        return consistent_images
    
    def _build_image_history(self, results):
        """Build history of all image predictions"""
        for dataset_num, checkpoints in results.items():
            config = self._get_available_datasets()[int(dataset_num)]
            for checkpoint_name, pred_data in checkpoints.items():
                for img_path, pred in pred_data['per_image_results'].items():
                    self.image_history[os.path.basename(img_path)].append({
                        'dataset': dataset_num,
                        'checkpoint': checkpoint_name,
                        **pred
                    })

    def _find_original_image_path(self, img_key):
        """Find original image path by checking all possible locations"""
        for line in self.wt_lines + self.ko_lines:
            candidate = os.path.join(self.base_paths['dataset_gen_input'], line, img_key)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Original image not found for {img_key}")
    
    def organize_high_confidence_images(self, consistent_images, output_subdir="high_confidence_examples"):
        """Organize high-confidence images into folder structure"""
        output_dir = os.path.join(self.base_paths['output_root'], output_subdir)
        
        for class_name in self.classes:
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for img_path, confidence in tqdm(
                consistent_images[class_name],
                desc=f"Copying {class_name}"
            ):
                try:
                    base, ext = os.path.splitext(os.path.basename(img_path))
                    confidence_pct = int(round(confidence * 100))
                    new_filename = f"{base}_{class_name}_{confidence_pct}{ext}"
                    dest_path = os.path.join(class_dir, new_filename)
                    
                    # Handle duplicates
                    counter = 1
                    while os.path.exists(dest_path):
                        new_filename = f"{base}_{class_name}_{confidence_pct}_{counter}{ext}"
                        dest_path = os.path.join(class_dir, new_filename)
                        counter += 1
                    
                    shutil.copy2(img_path, dest_path)
                except Exception as e:
                    print(f"Error copying {img_path}: {str(e)}")
        
        self._create_readme(output_dir, consistent_images)
        return output_dir

    def _save_results_to_csv(self, results, output_path):
        """Save analysis results to CSV"""
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
            print(f"Results saved to {output_path}")
        else:
            print("WARNING: No results to save!")

    def _create_readme(self, output_dir, consistent_images):
        """Create README file for output folder"""
        with open(os.path.join(output_dir, "README.txt"), 'w') as f:
            f.write("High-Confidence Image Classification Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Images meeting confidence threshold: {sum(len(v) for v in consistent_images.values())}\n")
            for class_name, images in consistent_images.items():
                f.write(f"{class_name}: {len(images)} images\n")

    def cleanup_test_folder(self):
        """Clean up the test folder after analysis"""
        test_folder = self.base_paths['test_folder']
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder, ignore_errors=True)




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
    # print("Resolved paths:", base_paths)
    
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
    consistent_images = analyzer.find_high_confidence_images(results, min_confidence=0.85)
    analyzer.organize_high_confidence_images(consistent_images)

    # Cleanuptest folder when done
    analyzer.cleanup_test_folder()