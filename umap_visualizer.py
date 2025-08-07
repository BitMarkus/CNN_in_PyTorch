import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import umap
from pacmap import PaCMAP
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from time import time
from settings import setting
from dataset import Dataset
from model import CNN_Model

class UMAP_Visualizer:
    
    # https://towardsdatascience.com/why-you-should-not-rely-on-t-sne-umap-or-trimap-f8f5dc333e59/

    def __init__(self, device):
        # self.device = torch.device(device) if isinstance(device, str) else device
        self.device = device
        self.mode = "test"
        
        # Load settings
        self.classes = setting['classes']
        self.pth_prediction = setting['pth_prediction']
        self.pth_checkpoint = setting['pth_checkpoint']
        
        # Initialize components
        self.ds = Dataset()
        self.cnn_wrapper = CNN_Model()
        self.scaler = StandardScaler()
        self.checkpoint_name = "untrained"
        self.checkpoint_loaded = False

        # UMAP:
        self.reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
            verbose=True
        )

        # PaCMAP:
        """
        self.reducer = PaCMAP(
            n_components=2,      # 2D projection
            n_neighbors=15,      # Similar to UMAP
            MN_ratio=0.5,        # Balance local/global (default=0.5)
            FP_ratio=2.0,        # Control far points (default=2.0)
            random_state=42,
            verbose=True
        )        
        """

        # Load model
        print(f"Creating new {self.cnn_wrapper.cnn_type} network...")
        self.cnn = self.cnn_wrapper.load_model(self.device).to(self.device)
        print("Network created successfully.")
        torch.backends.cudnn.benchmark = True

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
        # assert next(self.cnn.parameters()).device == self.device, "Model-device mismatch!"
        
        if self.mode == "train":
            if not hasattr(self.ds, 'load_training_dataset'):
                raise NotImplementedError("Training dataset loader not implemented")
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
                pooled = F.adaptive_avg_pool2d(batch_features, (1, 1))
                flattened = pooled.view(images.size(0), -1)
                features.append(flattened.cpu().numpy())
                labels.append(batch_labels.numpy())
                
        return np.concatenate(features), np.concatenate(labels)

    def visualize(self, features, labels):
        start_time = time()
        scaled_features = self.scaler.fit_transform(features)
        embedding = self.reducer.fit_transform(scaled_features)
        print(f"UMAP completed in {time()-start_time:.2f} seconds")
        
        plt.figure(figsize=(12, 8))
        for class_idx, class_name in enumerate(self.classes):
            mask = labels == class_idx
            if np.sum(mask) == 0:
                print(f"Warning: Class '{class_name}' has no samples")
                continue
            plt.scatter(
                embedding[mask, 0], embedding[mask, 1],
                label=class_name, alpha=0.7, s=40
            )
        
        plt.title(f"UMAP Projection\nCheckpoint: {self.checkpoint_name}")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.3)
        plt.gca().set_facecolor('#f5f5f5')
        plt.tight_layout()
        
        output_dir = self.pth_prediction / "umap_results"
        try:
            output_dir.mkdir(exist_ok=True, parents=True)
        except PermissionError:
            print(f"Error: Cannot create directory {output_dir}")
            return
        
        output_path = output_dir / f"umap_{self.checkpoint_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved UMAP to {output_path}")

    def run(self):
        print("\nStarting UMAP visualization...")
        if Path(self.pth_checkpoint).exists():
            self.load_checkpoint()
        if not self.checkpoint_loaded:
            print("Warning: Using untrained weights")
        features, labels = self.extract_features()
        self.visualize(features, labels)
        torch.cuda.empty_cache()
        print("Done!")

"""
# For test data (default):
visualizer = UMAP_Visualizer(device="cuda")  # Auto-detects GPU
visualizer.run()

For training data (if implemented):
visualizer = UMAP_Visualizer(device="cuda")
visualizer.mode = "train"  # Switch to training mode
visualizer.run()
"""