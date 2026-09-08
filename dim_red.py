# ===== Standard Library Imports =====
import os
import json
import time
import gc
from pathlib import Path
# ===== Third-Party Imports =====
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import trimap
from pacmap import PaCMAP
import sklearn
# ===== Own Modules =====
from settings import setting
from dataset import Dataset


class DimRed:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the dimensionality reduction handler.
    # Supports UMAP, t-SNE, TriMAP, and PaCMAP for visualizing high-dimensional
    # feature embeddings from CNN models.
    # Args:
    #   device (torch.device): Device to run feature extraction on
    def __init__(self, device: torch.device) -> None:

        self.device = device

        # Settings parameters
        self.classes = setting['classes']
        self.pth_prediction = Path(setting['pth_prediction'])
        self.pth_checkpoint = Path(setting['pth_checkpoint'])

        # Group settings
        self.mode = setting['dimred_mode']  # "train", "test", or "groups"
        self.group_mode = setting['dimred_group_mode']  # 'auto' or 'manual'
        self.group_mapping = setting['dimred_group_mapping']

        # Color palette setting
        self.color_palette = setting['dimred_color_palette']

        # Export format setting
        self.export_format = setting.get('dimred_export_format', 'csv')

        # Set up group mapping based on mode
        if self.mode == "groups":
            if self.group_mode == "manual" and isinstance(self.group_mapping, dict) and self.group_mapping:
                manual_mapping = {}
                for idx, (folder_name, display_name) in enumerate(self.group_mapping.items()):
                    manual_mapping[folder_name] = (display_name, idx)
                self.group_mapping = manual_mapping
                print(f"Manual mapping set up: {list(self.group_mapping.keys())}")
            else:
                self._setup_auto_group_mapping()

            if not self.group_mapping:
                raise ValueError("No groups found for dimensionality reduction!")

        # Method activation flags
        self.use_umap = setting['dimred_use_umap']
        self.use_tsne = setting['dimred_use_tsne']
        self.use_trimap = setting['dimred_use_trimap']
        self.use_pacmap = setting['dimred_use_pacmap']

        # UMAP parameters
        self.umap_params = {
            'n_neighbors': setting['dimred_umap_n_neighbors'],
            'min_dist': setting['dimred_umap_min_dist'],
            'random_state': 42
        }

        # t-SNE parameters (check scikit-learn version for API compatibility)
        sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
        if sklearn_version >= (1, 2):
            self.tsne_params = {
                'perplexity': setting['dimred_tsne_perplexity'],
                'learning_rate': setting['dimred_tsne_learning_rate'],
                'random_state': 42,
                'max_iter': 1000
            }
        else:
            self.tsne_params = {
                'perplexity': setting['dimred_tsne_perplexity'],
                'learning_rate': setting['dimred_tsne_learning_rate'],
                'random_state': 42,
                'n_iter': 1000
            }

        # TriMAP parameters
        self.trimap_params = {
            'n_inliers': setting['dimred_trimap_n_inliers'],
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
        self.sample_ids = []
        self.n_features = 0

        # Initialize model
        from model import CNN_Model
        self.cnn_wrapper = CNN_Model()
        self.cnn = self.cnn_wrapper.load_model(self.device).to(self.device)
        torch.backends.cudnn.benchmark = True

    #############################################################################################################
    # METHODS

    # Automatically set up group mapping based on folder structure in predictions folder.
    def _setup_auto_group_mapping(self) -> None:
        if not self.pth_prediction.exists():
            print(f"Warning: Predictions directory {self.pth_prediction} does not exist for auto group detection")
            self.group_mapping = {}
            return

        self.group_mapping = {}

        for folder_name in os.listdir(self.pth_prediction):
            folder_path = self.pth_prediction / folder_name
            if folder_path.is_dir():
                label_val = len(self.group_mapping)
                self.group_mapping[folder_name] = (folder_name, label_val)

        print(f"Auto-detected {len(self.group_mapping)} groups in predictions folder: {list(self.group_mapping.keys())}")

    # Load model weights from a selected checkpoint.
    # Handles both direct state dict and wrapped format from train.py.
    # Returns:
    #   bool: True if loading succeeded, False otherwise
    def load_checkpoint(self) -> bool:
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

            checkpoint = torch.load(full_path, map_location=self.device)

            if 'model_state_dict' in checkpoint:
                checkpoint_weights = checkpoint['model_state_dict']
                print(f"Loaded from training checkpoint (epoch {checkpoint.get('epoch', '?')}, acc={checkpoint.get('accuracy', 0):.2%})")
            else:
                checkpoint_weights = checkpoint
                print("Loaded direct state dict format")

            # Clean layer names (remove 'model.' or 'module.' prefixes)
            cleaned_weights = {}
            for k, v in checkpoint_weights.items():
                new_k = k
                if new_k.startswith('model.'):
                    new_k = new_k[6:]
                if new_k.startswith('module.'):
                    new_k = new_k[7:]
                cleaned_weights[new_k] = v

            model_dict = self.cnn.state_dict()

            compatible_weights = {}
            skipped_layers = []

            for k, v in cleaned_weights.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        compatible_weights[k] = v
                    else:
                        skipped_layers.append(f"{k} (shape mismatch)")
                else:
                    skipped_layers.append(f"{k} (not in model)")

            feature_layers_loaded = [k for k in compatible_weights.keys() if 'classifier' not in k]
            print(f"Loaded {len(feature_layers_loaded)} feature extraction layers")

            if skipped_layers:
                classifier_skipped = [s for s in skipped_layers if 'classifier' in s]
                if classifier_skipped:
                    print(f"Skipped {len(classifier_skipped)} classifier layers (expected)")

            model_dict.update(compatible_weights)
            self.cnn.load_state_dict(model_dict)

            new_params = list(self.cnn.parameters())[0]
            if torch.equal(original_params, new_params):
                print("ERROR: No weights loaded! Checkpoint format may be incompatible.")
                return False

            self.checkpoint_loaded = True
            self.checkpoint_name = full_path.stem
            print(f"Successfully loaded weights from {checkpoint_file}\n")
            return True

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    # Extract features from images using the CNN model.
    # Returns:
    #   tuple: (features, labels) as numpy arrays
    def extract_features(self) -> tuple:
        features, labels = [], []
        self.sample_ids = []

        if self.mode in ["train", "test", "groups"]:
            self.ds = Dataset()

            if self.mode == "train":
                if not self.ds.load_training_dataset():
                    raise ValueError("Failed to load training data")
                dataloader = self.ds.ds_train
            elif self.mode == "test":
                if not self.ds.load_test_dataset():
                    raise ValueError("Failed to load test data")
                dataloader = self.ds.ds_test
            elif self.mode == "groups":
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

                    if self.mode == "groups" and hasattr(dataloader.dataset, 'samples'):
                        for idx_in_batch in range(len(batch_labels)):
                            sample_idx = batch_idx * dataloader.batch_size + idx_in_batch
                            if sample_idx < len(dataloader.dataset.samples):
                                img_path = dataloader.dataset.samples[sample_idx][0]
                                self.sample_ids.append(Path(img_path).name)

                    if self.mode == "groups":
                        numeric_labels = []
                        for label_val in batch_labels.numpy():
                            folder_name = None
                            if hasattr(dataloader.dataset, 'classes') and label_val < len(dataloader.dataset.classes):
                                folder_name = dataloader.dataset.classes[label_val]

                            if folder_name and folder_name in self.group_mapping:
                                _, desired_label = self.group_mapping[folder_name]
                                numeric_labels.append(desired_label)
                            else:
                                numeric_labels.append(label_val)

                        labels.append(np.array(numeric_labels))
                    else:
                        labels.append(batch_labels.numpy())

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'train', 'test', or 'groups'")

        all_features = np.concatenate(features)
        all_labels = np.concatenate(labels)

        if len(self.sample_ids) != len(all_labels):
            self.sample_ids = [f"sample_{i}" for i in range(len(all_labels))]

        return all_features, all_labels

    # Get a colormap with comprehensive error handling.
    # Supports any matplotlib colormap name.
    # Args:
    #   num_groups (int): Number of groups for discrete colormaps
    # Returns:
    #   matplotlib.colors.Colormap: The selected colormap
    def _get_color_map(self, num_groups: int):
        FALLBACK_MAPS = ['viridis', 'plasma', 'tab10', 'rainbow']

        try:
            if self.color_palette == 'default':
                if num_groups <= 10:
                    return plt.cm.tab10
                elif num_groups <= 20:
                    return plt.cm.tab20
                else:
                    return plt.cm.viridis
            else:
                cmap = plt.cm.get_cmap(self.color_palette)
                discrete_maps = ['tab10', 'Set1', 'Set2', 'Set3', 'Dark2', 'Paired']
                if self.color_palette in discrete_maps and num_groups > 10:
                    print(f"Warning: Discrete colormap '{self.color_palette}' used with {num_groups} groups. Colors will repeat.")
                return cmap

        except (ValueError, AttributeError) as e:
            for fallback in FALLBACK_MAPS:
                try:
                    print(f"Colormap '{self.color_palette}' not found. Using '{fallback}' instead.")
                    return plt.cm.get_cmap(fallback)
                except:
                    continue

            print("All fallbacks failed. Using basic rainbow.")
            return plt.cm.rainbow

    # Export embedding data to CSV or JSON format.
    # Args:
    #   method (str): Name of the reduction method
    #   embedding (np.ndarray): Embedding coordinates (n_samples, 2)
    #   labels (np.ndarray): Labels (n_samples,)
    #   output_dir (Path): Output directory
    def _export_embedding_data(self, method: str, embedding: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
        if self.mode in ["train", "test"]:
            label_names_dict = {idx: name for idx, name in enumerate(self.classes)}
        else:
            label_names_dict = {}
            for folder_name, (display_name, label_val) in self.group_mapping.items():
                label_names_dict[label_val] = display_name

        export_rows = []
        for i in range(len(embedding)):
            label_val = int(labels[i])
            label_name = label_names_dict.get(label_val, f"class_{label_val}")

            row = {
                'sample_id': self.sample_ids[i] if i < len(self.sample_ids) else f"sample_{i}",
                f'{method.lower()}_dim1': float(embedding[i, 0]),
                f'{method.lower()}_dim2': float(embedding[i, 1]),
                'label_numeric': label_val,
                'label_name': label_name,
                'dataset': self.mode
            }
            export_rows.append(row)

        if self.export_format == 'csv':
            df = pd.DataFrame(export_rows)
            csv_path = output_dir / f"{method.lower()}_{self.mode}_{self.checkpoint_name}_embedding.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved embedding CSV to {csv_path}")

        elif self.export_format == 'json':
            export_data = {
                'metadata': {
                    'method': method,
                    'mode': self.mode,
                    'checkpoint': self.checkpoint_name,
                    'n_samples': len(embedding),
                    'color_palette': self.color_palette,
                    'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'embeddings': export_rows
            }
            json_path = output_dir / f"{method.lower()}_{self.mode}_{self.checkpoint_name}_embedding.json"
            with open(json_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"  ✓ Saved embedding JSON to {json_path}")

        # Always export parameters as JSON
        params_data = {
            'method': method,
            'parameters': getattr(self, f'{method.lower()}_params', {}),
            'n_samples': len(embedding),
            'n_features_original': self.n_features,
            'checkpoint': self.checkpoint_name,
            'mode': self.mode,
            'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        params_path = output_dir / f"{method.lower()}_{self.mode}_{self.checkpoint_name}_params.json"
        with open(params_path, 'w') as f:
            json.dump(params_data, f, indent=2)
        print(f"  ✓ Saved parameters to {params_path}")

    # Run a single dimensionality reduction method and save results.
    # Args:
    #   method (str): Name of the method
    #   reducer: The reducer object (fit_transform method expected)
    #   features (np.ndarray): Feature matrix
    #   labels (np.ndarray): Labels
    def run_reduction(self, method: str, reducer, features: np.ndarray, labels: np.ndarray) -> None:
        print(f"\nRunning {method}...")

        start_time = time.time()
        scaled_features = self.scaler.fit_transform(features)
        embedding = reducer.fit_transform(scaled_features)
        print(f"{method} completed in {time.time()-start_time:.2f} seconds")

        output_dir = self.pth_prediction / "dim_red"
        output_dir.mkdir(exist_ok=True, parents=True)

        self._export_embedding_data(method, embedding, labels, output_dir)

        plt.figure(figsize=(12, 8))

        if self.mode in ["train", "test"]:
            for class_idx, class_name in enumerate(self.classes):
                mask = labels == class_idx
                if np.sum(mask) == 0:
                    continue
                plt.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    label=class_name, alpha=0.7, s=40
                )
        else:
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_counts = list(zip(unique_labels, counts))
            label_counts.sort(key=lambda x: x[1], reverse=True)

            num_groups = len(unique_labels)

            try:
                cmap = self._get_color_map(num_groups)
                print(f"Using colormap: {self.color_palette}")
            except Exception as e:
                print(f"Error loading colormap '{self.color_palette}': {e}. Using viridis instead.")
                cmap = plt.cm.viridis

            scatter_objects = []
            legend_labels = []

            for i, (label_val, count) in enumerate(label_counts):
                mask = labels == label_val
                if np.sum(mask) == 0:
                    continue

                display_name = f"Label_{label_val}"
                for folder_name, (name, val) in self.group_mapping.items():
                    if val == label_val:
                        display_name = name
                        break

                if self.color_palette == 'default':
                    color = cmap(i % cmap.N)
                else:
                    color = cmap(i / max(1, num_groups - 1))

                scatter = plt.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    color=color,
                    alpha=0.7,
                    s=40
                )
                scatter_objects.append(scatter)
                legend_labels.append(display_name)

        if self.mode == "groups":
            title = f"{method} Projection ({len(self.group_mapping)} Groups)\nCheckpoint: {self.checkpoint_name} | Palette: {self.color_palette}"
        else:
            title = f"{method} Projection ({self.mode} set)\nCheckpoint: {self.checkpoint_name} | Palette: {self.color_palette}"

        plt.title(title)
        plt.xlabel(f"{method} 1")
        plt.ylabel(f"{method} 2")

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

        if self.mode == "groups":
            output_path = output_dir / f"{method.lower()}_{len(self.group_mapping)}_groups_{self.checkpoint_name}_{self.color_palette}.png"
        else:
            output_path = output_dir / f"{method.lower()}_{self.mode}_{self.checkpoint_name}_{self.color_palette}.png"

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved {method} plot to {output_path}")

    #############################################################################################################
    # CALL

    # Run all enabled dimensionality reduction methods.
    def __call__(self) -> None:
        if not any([self.use_umap, self.use_tsne, self.use_trimap, self.use_pacmap]):
            print("Warning: No dimensionality reduction methods enabled!")
            return

        print(f"\n{'='*60}")
        print(f"STARTING DIMENSIONALITY REDUCTION")
        print(f"{'='*60}")
        print(f"Mode: {self.mode}")
        print(f"Color palette: {self.color_palette}")
        print(f"Export format: {self.export_format.upper()}")

        if self.pth_checkpoint.exists():
            self.load_checkpoint()
        if not self.checkpoint_loaded:
            print("Warning: Using untrained weights")

        features, labels = self.extract_features()

        print(f"\nExtracted features: {features.shape}")
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Labels: {dict(zip(unique_labels, counts))}")

        self.n_features = features.shape[1]

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
        print(f"\n{'='*60}")
        print("All reductions completed!")
        print(f"Output directory: {self.pth_prediction / 'dim_red'}")
        print(f"{'='*60}\n")