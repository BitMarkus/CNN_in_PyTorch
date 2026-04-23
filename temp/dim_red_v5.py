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

        # NEW: Color palette setting
        self.color_palette = setting['dimred_color_palette']  # 'default' or any matplotlib colormap name, like 'rainbow', 'jet', etc.

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

    # Automatically set up group mapping based on folder structure in predictions/ folder
    def _setup_auto_group_mapping(self):
        
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
            print(f"Successfully loaded compatible weights from {checkpoint_file}\n")
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
    
    def extract_features(self):

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
                for batch_idx, (images, batch_labels) in enumerate(tqdm(dataloader, desc="Extracting features")):
                    images = images.to(self.device)
                    batch_features = self.cnn.features(images)
                    pooled = torch.nn.functional.adaptive_avg_pool2d(batch_features, (1, 1))
                    flattened = pooled.view(images.size(0), -1)
                    features.append(flattened.cpu().numpy())
                    
                    # For groups mode, we need to REMAP the labels to match our group_mapping
                    if self.mode == "groups":
                        # The dataloader gives us ImageFolder's automatic labels
                        # We need to map them to our desired group_mapping labels
                        numeric_labels = []
                        for label_val in batch_labels.numpy():
                            # Find which folder name corresponds to this numeric label
                            folder_name = None
                            if hasattr(dataloader.dataset, 'classes') and label_val < len(dataloader.dataset.classes):
                                folder_name = dataloader.dataset.classes[label_val]
                            
                            # Now map to our group_mapping numeric value
                            if folder_name and folder_name in self.group_mapping:
                                _, desired_label = self.group_mapping[folder_name]
                                numeric_labels.append(desired_label)
                            else:
                                # Keep original if not found
                                numeric_labels.append(label_val)
                        
                        labels.append(np.array(numeric_labels))
                    else:
                        labels.append(batch_labels.numpy())
                    
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'train', 'test', or 'groups'")
                
        all_features = np.concatenate(features)
        all_labels = np.concatenate(labels)

        return all_features, all_labels

    def _get_color_map(self, num_groups):
        """Get colormap with comprehensive error handling - supports any matplotlib colormap"""
        # List of safe fallback colormaps in order of preference
        FALLBACK_MAPS = ['viridis', 'plasma', 'tab10', 'rainbow']
        
        try:
            if self.color_palette == 'default':
                # Keep the original smart default behavior
                if num_groups <= 10:
                    return plt.cm.tab10
                elif num_groups <= 20:
                    return plt.cm.tab20
                else:
                    return plt.cm.viridis
            else:
                # Try to get the requested colormap
                cmap = plt.cm.get_cmap(self.color_palette)
                
                # Warn if using discrete colormap with many groups
                discrete_maps = ['tab10', 'Set1', 'Set2', 'Set3', 'Dark2', 'Paired']
                if self.color_palette in discrete_maps and num_groups > 10:
                    print(f"Warning: Discrete colormap '{self.color_palette}' used with {num_groups} groups. Colors will repeat.")
                
                return cmap
                
        except (ValueError, AttributeError) as e:
            # Try fallbacks
            for fallback in FALLBACK_MAPS:
                try:
                    print(f"Colormap '{self.color_palette}' not found. Using '{fallback}' instead.")
                    return plt.cm.get_cmap(fallback)
                except:
                    continue
            
            # Ultimate fallback
            print("All fallbacks failed. Using basic rainbow.")
            return plt.cm.rainbow

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
            # FIXED: Plot by sample count (largest first = background, smallest last = foreground)
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Create list of (label, count) pairs and sort by count (descending)
            label_counts = list(zip(unique_labels, counts))
            label_counts.sort(key=lambda x: x[1], reverse=True)  # Sort by count descending
            
            num_groups = len(unique_labels)
            
            # Get appropriate colormap with error handling
            try:
                cmap = self._get_color_map(num_groups)
                print(f"Using colormap: {self.color_palette}")
            except Exception as e:
                print(f"Error loading colormap '{self.color_palette}': {e}. Using viridis instead.")
                cmap = plt.cm.viridis
            
            # Create scatter plots
            scatter_objects = []
            legend_labels = []
            
            # Plot in order: largest groups first (background), smallest last (foreground)
            for i, (label_val, count) in enumerate(label_counts):
                mask = labels == label_val
                if np.sum(mask) == 0:
                    continue
                
                # Find display name
                display_name = f"Label_{label_val}"
                for folder_name, (name, val) in self.group_mapping.items():
                    if val == label_val:
                        display_name = name
                        break
                
                # Enhanced color assignment for any colormap
                if self.color_palette == 'default':
                    # Use discrete color selection for default behavior
                    color = cmap(i % cmap.N)
                else:
                    # For named colormaps, distribute colors evenly across the colormap
                    # This works for both continuous and discrete colormaps
                    color = cmap(i / max(1, num_groups - 1))
                
                # Create scatter plot
                scatter = plt.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    color=color,
                    alpha=0.7, 
                    s=40
                )
                scatter_objects.append(scatter)
                legend_labels.append(display_name)

        # Update title based on mode and color palette
        if self.mode == "groups":
            title = f"{method} Projection ({len(self.group_mapping)} Groups)\nCheckpoint: {self.checkpoint_name} | Palette: {self.color_palette}"
        else:
            title = f"{method} Projection ({self.mode} set)\nCheckpoint: {self.checkpoint_name} | Palette: {self.color_palette}"
        
        plt.title(title)
        plt.xlabel(f"{method} 1")
        plt.ylabel(f"{method} 2")
        
        # Create legend using the actual scatter objects
        if self.mode == "groups":
            n_groups = len(scatter_objects)
            if n_groups > 10:
                plt.legend(scatter_objects, legend_labels, 
                        bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3)
            else:
                plt.legend(scatter_objects, legend_labels, 
                        bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.grid(alpha=0.3)
        plt.gca().set_facecolor('#f5f5f5')
        plt.tight_layout()
        
        output_dir = self.pth_prediction / "dim_red"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Update filename based on mode and palette
        if self.mode == "groups":
            output_path = output_dir / f"{method.lower()}_{len(self.group_mapping)}_groups_{self.checkpoint_name}_{self.color_palette}.png"
        else:
            output_path = output_dir / f"{method.lower()}_{self.mode}_{self.checkpoint_name}_{self.color_palette}.png"
            
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
        print(f"Using color palette: {self.color_palette}")
        if self.pth_checkpoint.exists():
            self.load_checkpoint()
        if not self.checkpoint_loaded:
            print("Warning: Using untrained weights")
        
        features, labels = self.extract_features()
        
        # Debug info
        print(f"Extracted features: {features.shape}")
        print(f"Labels: {np.unique(labels, return_counts=True)}")
        
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