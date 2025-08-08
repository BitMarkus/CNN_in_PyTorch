import torch
import numpy as np
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

class DimRed:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device):
         
        self.device = device

        # Settings parameters
        self.classes = setting['classes']
        self.pth_prediction = Path(setting['pth_prediction'])
        self.pth_checkpoint = Path(setting['pth_checkpoint'])
        # Mode ("train" or "test")
        self.mode = setting['dimred_mode']  
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
            self.cnn.load_state_dict(torch.load(full_path))
            new_params = list(self.cnn.parameters())[0]
            if torch.equal(original_params, new_params):
                print("Warning: Model weights unchanged after loading!")
            self.checkpoint_loaded = True
            self.checkpoint_name = full_path.stem
            print(f"Loaded weights from {checkpoint_file}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def extract_features(self):

        from dataset import Dataset
        self.ds = Dataset()
        
        # Modified to use mode parameter
        if self.mode == "train":
            if not self.ds.load_training_dataset():
                raise ValueError("Failed to load training data")
            dataloader = self.ds.ds_train
        else:
            if not self.ds.load_pred_dataset():
                raise ValueError("Failed to load test data")
            dataloader = self.ds.ds_pred
        
        self.cnn.eval()
        features, labels = [], []
        
        with torch.no_grad():
            for images, batch_labels in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)
                batch_features = self.cnn.features(images)
                pooled = torch.nn.functional.adaptive_avg_pool2d(batch_features, (1, 1))
                flattened = pooled.view(images.size(0), -1)
                features.append(flattened.cpu().numpy())
                labels.append(batch_labels.numpy())
                
        return np.concatenate(features), np.concatenate(labels)

    def run_reduction(self, method, reducer, features, labels):

        print(f"\nRunning {method}...")
        start_time = time()
        scaled_features = self.scaler.fit_transform(features)
        embedding = reducer.fit_transform(scaled_features)
        print(f"{method} completed in {time()-start_time:.2f} seconds")
        
        plt.figure(figsize=(12, 8))
        for class_idx, class_name in enumerate(self.classes):
            mask = labels == class_idx
            if np.sum(mask) == 0:
                continue
            plt.scatter(
                embedding[mask, 0], embedding[mask, 1],
                label=class_name, alpha=0.7, s=40
            )
        
        plt.title(f"{method} Projection ({self.mode} set)\nCheckpoint: {self.checkpoint_name}")
        plt.xlabel(f"{method} 1")
        plt.ylabel(f"{method} 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.gca().set_facecolor('#f5f5f5')
        plt.tight_layout()
        
        output_dir = self.pth_prediction / "dim_red"
        output_dir.mkdir(exist_ok=True, parents=True)
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