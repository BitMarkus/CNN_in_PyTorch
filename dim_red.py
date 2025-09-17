import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
from sklearn.manifold import TSNE
import umap
import trimap
from pacmap import PaCMAP
from sklearn.preprocessing import StandardScaler
import gc
# Own modules
from settings import setting
from dataset import Dataset

class DimRed:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device):
         
        self.device = device

        # Settings parameters
        self.classes = setting['classes']
        self.pth_prediction = Path(setting['pth_prediction'])
        self.pth_checkpoint = Path(setting['pth_checkpoint'])
        
        # Group settings
        self.mode = setting['dimred_mode']  # "train", "test", or "groups"
        self.group_mode = setting['dimred_group_mode'] # 'auto' or 'manual'
        self.group_mapping = setting['dimred_group_mapping'] 

        # Set up group mapping based on mode
        if self.mode == "groups":
            if self.group_mode == "manual" and isinstance(self.group_mapping, dict) and self.group_mapping:
                # Convert manual mapping format
                manual_mapping = {}
                for idx, (folder_name, display_name) in enumerate(self.group_mapping.items()):
                    manual_mapping[folder_name] = (display_name, idx)
                self.group_mapping = manual_mapping
                print(f"Manual mapping set up: {list(self.group_mapping.keys())}")
            else:
                # Auto-detect groups
                self._setup_auto_group_mapping()
            # Check to ensure groups were actually set up
            if not self.group_mapping:
                raise ValueError("No groups found for dimensionality reduction!")
        
        # Method activation flags
        self.use_umap = setting['dimred_use_umap'] 
        self.use_tsne = setting['dimred_use_tsne'] 
        self.use_trimap = setting['dimred_use_trimap'] 
        self.use_pacmap = setting['dimred_use_pacmap']      
        
        # Store parameters for each method
        # UMAP parameters
        self.umap_params = {
            'n_neighbors': setting['dimred_umap_n_neighbors'],
            'min_dist': setting['dimred_umap_min_dist'],
            'random_state': 42
        }
        # t-SNE parameters
        self.tsne_params = {
            'perplexity': setting['dimred_tsne_perplexity'],
            'learning_rate': setting['dimred_tsne_learning_rate'],
            'random_state': 42,
            'n_iter': 1000
        }
        # TriMAP parameters
        self.trimap_params = {
            'n_inliers':  setting['dimred_trimap_n_inliers'],
            'n_outliers': setting['dimred_trimap_n_outliers'],
            'n_random': 5
        }
        # PaCMAP parameters
        self.pacmap_params = {
            'n_neighbors': setting['dimred_pacmap_n_neighbors'],
            'MN_ratio': setting['dimred_pacmap_MN_ratio'],
            'FP_ratio': setting['dimred_pacmap_FP_ratio'],
            'random_state': 42
        }

        # Initialize components
        self.scaler = StandardScaler()
        self.checkpoint_name = "untrained"
        self.checkpoint_loaded = False
        
        # Initialize model
        from model import CNN_Model
        self.cnn_wrapper = CNN_Model()
        self.cnn = self.cnn_wrapper.load_model(self.device).to(self.device)
        torch.backends.cudnn.benchmark = True

    #############################################################################################################
    # METHODS:

    def _setup_auto_group_mapping(self):
        """Automatically set up group mapping based on folder structure in predictions/ folder"""
        
        if not self.pth_prediction.exists():
            print(f"Warning: Predictions directory {self.pth_prediction} does not exist for auto group detection")
            self.group_mapping = {}
            return
        
        # Find which folders actually exist and create mapping
        self.group_mapping = {}
        
        for folder_name in os.listdir(self.pth_prediction):
            folder_path = self.pth_prediction / folder_name
            if folder_path.is_dir():
                # Use folder name as display name
                label_val = len(self.group_mapping)
                self.group_mapping[folder_name] = (folder_name, label_val)
        
        print(f"Auto-detected {len(self.group_mapping)} groups in predictions folder: {list(self.group_mapping.keys())}")

    def load_checkpoint(self):
        silent_checkpoints = self.cnn_wrapper.print_checkpoints_table(self.pth_checkpoint, print_table=False)
        if not silent_checkpoints:
            print("No checkpoints found!")
            return False
        
        if len(silent_checkpoints) == 1:
            checkpoint_file = silent_checkpoints[0][1]
            print(f"\nFound single checkpoint: {checkpoint_file}")
        else:
            self.cnn_wrapper.print_checkpoints_table(self.pth_checkpoint)
            checkpoint_file = self.cnn_wrapper.select_checkpoint(silent_checkpoints, "Select checkpoint: ")
            if not checkpoint_file:
                return False
        
        try:
            original_params = list(self.cnn.parameters())[0].clone()
            full_path = self.pth_checkpoint / checkpoint_file
            
            # Load checkpoint weights
            checkpoint_weights = torch.load(full_path)
            model_dict = self.cnn.state_dict()
            
            # Filter out incompatible weights (like the classifier)
            compatible_weights = {k: v for k, v in checkpoint_weights.items() 
                                if k in model_dict and model_dict[k].shape == v.shape}
            
            # Load only compatible weights
            model_dict.update(compatible_weights)
            self.cnn.load_state_dict(model_dict)
            
            # Report what was loaded
            print(f"Loaded {len(compatible_weights)}/{len(checkpoint_weights)} layers from checkpoint")
            if len(compatible_weights) < len(checkpoint_weights):
                print("Note: Skipped incompatible classifier layer - using feature extractor only")
            
            # Verify weights changed
            new_params = list(self.cnn.parameters())[0]
            if torch.equal(original_params, new_params):
                print("Warning: Model weights unchanged after loading!")
            
            self.checkpoint_loaded = True
            self.checkpoint_name = full_path.stem
            print(f"Successfully loaded compatible weights from {checkpoint_file}")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def extract_features(self):
        """
        Modified to handle both original modes (train/test) and the new groups mode
        Uses existing dataset loading functionality for all modes
        """
        features, labels = [], []
        
        if self.mode in ["train", "test", "groups"]:

            self.ds = Dataset()
            
            if(self.mode == "train"):
                if not self.ds.load_training_dataset():
                    raise ValueError("Failed to load training data")
                dataloader = self.ds.ds_train
            elif(self.mode == "test"):
                if not self.ds.load_test_dataset():
                    raise ValueError("Failed to load test data")
                dataloader = self.ds.ds_test
            elif(self.mode == "groups"):
                if not self.ds.load_pred_dataset():
                    raise ValueError("Failed to load prediction data")
                dataloader = self.ds.ds_pred
            
            self.cnn.eval()
            with torch.no_grad():
                for images, batch_labels in tqdm(dataloader, desc="Extracting features"):
                    images = images.to(self.device)
                    batch_features = self.cnn.features(images)
                    pooled = torch.nn.functional.adaptive_avg_pool2d(batch_features, (1, 1))
                    flattened = pooled.view(images.size(0), -1)
                    features.append(flattened.cpu().numpy())
                    labels.append(batch_labels.numpy())
                    
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'train', 'test', or 'groups'")
                
        return np.concatenate(features), np.concatenate(labels)

    def run_reduction(self, method, reducer, features, labels):
        print(f"\nRunning {method}...")
        start_time = time()
        scaled_features = self.scaler.fit_transform(features)
        embedding = reducer.fit_transform(scaled_features)
        print(f"{method} completed in {time()-start_time:.2f} seconds")
        
        plt.figure(figsize=(12, 8))
        
        if self.mode in ["train", "test"]:
            # Original 2-class plotting
            for class_idx, class_name in enumerate(self.classes):
                mask = labels == class_idx
                if np.sum(mask) == 0:
                    continue
                plt.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    label=class_name, alpha=0.7, s=40
                )
        else:
            # NEW: Flexible group plotting
            unique_labels = np.unique(labels)
            for label_val in unique_labels:
                mask = labels == label_val
                if np.sum(mask) == 0:
                    continue
                
                # Find the display name for this label
                display_name = f"Group {label_val}"
                for folder_name, (name, val) in self.group_mapping.items():
                    if val == label_val:
                        display_name = name
                        break
                
                plt.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    label=display_name, alpha=0.7, s=40
                )
        
        # Update title based on mode
        if self.mode == "groups":
            title = f"{method} Projection ({len(self.group_mapping)} Groups)\nCheckpoint: {self.checkpoint_name}"
        else:
            title = f"{method} Projection ({self.mode} set)\nCheckpoint: {self.checkpoint_name}"
        
        plt.title(title)
        plt.xlabel(f"{method} 1")
        plt.ylabel(f"{method} 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.gca().set_facecolor('#f5f5f5')
        plt.tight_layout()
        
        output_dir = self.pth_prediction / "dim_red"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Update filename based on mode
        if self.mode == "groups":
            output_path = output_dir / f"{method.lower()}_{len(self.group_mapping)}_groups_{self.checkpoint_name}.png"
        else:
            output_path = output_dir / f"{method.lower()}_{self.mode}_{self.checkpoint_name}.png"
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {method} plot to {output_path}")

    #############################################################################################################
    # CALL

    def __call__(self):

        if not any([self.use_umap, self.use_tsne, self.use_trimap, self.use_pacmap]):
            print("Warning: No dimensionality reduction methods enabled!")
            return

        print(f"\nStarting dimensionality reduction on {self.mode} set...")
        if self.pth_checkpoint.exists():
            self.load_checkpoint()
        if not self.checkpoint_loaded:
            print("Warning: Using untrained weights")
        
        features, labels = self.extract_features()
        
        if self.use_umap:
            reducer = umap.UMAP(
                n_components=2,
                **self.umap_params,
                verbose=True
            )
            self.run_reduction("UMAP", reducer, features, labels)
            gc.collect()
        
        if self.use_tsne:
            reducer = TSNE(
                n_components=2,
                **self.tsne_params,
                verbose=1
            )
            self.run_reduction("t-SNE", reducer, features, labels)
            gc.collect()
        
        if self.use_trimap:
            reducer = trimap.TRIMAP(
                n_dims=2,
                **self.trimap_params,
                verbose=True
            )
            self.run_reduction("TriMAP", reducer, features, labels)
            gc.collect()
        
        if self.use_pacmap:
            reducer = PaCMAP(
                n_components=2,
                **self.pacmap_params,
                verbose=True
            )
            self.run_reduction("PaCMAP", reducer, features, labels)
            gc.collect()
        
        torch.cuda.empty_cache()
        print("\nAll reductions completed!")