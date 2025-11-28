import streamlit as st
import torch
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import trimap
from pacmap import PaCMAP
from sklearn.preprocessing import StandardScaler
import pandas as pd
from PIL import Image
import plotly.express as px
import gc

# Import your existing modules
from settings import setting
from dataset import Dataset

# set_page_config MUST be the VERY FIRST Streamlit command
st.set_page_config(layout="wide")

class StreamlitDimRed:
    def __init__(self, device):
        self.device = device
        
        # Settings parameters
        self.classes = setting['classes']
        self.pth_prediction = Path(setting['pth_prediction'])
        self.pth_checkpoint = Path(setting['pth_checkpoint'])
        
        # Group settings
        self.mode = setting['dimred_mode']
        self.group_mode = setting['dimred_group_mode']
        self.group_mapping = setting['dimred_group_mapping']
        
        # Interactive settings
        self.interactive_sample_size = setting.get('dimred_interactive_sample_size', 2000)
        self.interactive_thumbnail_size = setting.get('dimred_interactive_thumbnail_size', 80)
        self.color_palette = setting.get('dimred_color_palette', 'default')
        
        # Method activation flags
        self.use_umap = setting['dimred_use_umap']
        self.use_tsne = setting['dimred_use_tsne']
        self.use_trimap = setting['dimred_use_trimap']
        self.use_pacmap = setting['dimred_use_pacmap']
        
        # Parameters for each method
        self.umap_params = {
            'n_neighbors': setting['dimred_umap_n_neighbors'],
            'min_dist': setting['dimred_umap_min_dist'],
            'random_state': 42
        }
        self.tsne_params = {
            'perplexity': setting['dimred_tsne_perplexity'],
            'learning_rate': setting['dimred_tsne_learning_rate'],
            'random_state': 42,
            'n_iter': 1000
        }
        self.trimap_params = {
            'n_inliers': setting['dimred_trimap_n_inliers'],
            'n_outliers': setting['dimred_trimap_n_outliers'],
            'n_random': 5
        }
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

        # Setup group mapping
        if self.mode == "groups":
            if self.group_mode == "manual" and isinstance(self.group_mapping, dict) and self.group_mapping:
                manual_mapping = {}
                for idx, (folder_name, display_name) in enumerate(self.group_mapping.items()):
                    manual_mapping[folder_name] = (display_name, idx)
                self.group_mapping = manual_mapping
            else:
                self._setup_auto_group_mapping()

    def _setup_auto_group_mapping(self):
        if not self.pth_prediction.exists():
            st.error(f"Predictions directory {self.pth_prediction} does not exist!")
            self.group_mapping = {}
            return
        
        self.group_mapping = {}
        for folder_name in os.listdir(self.pth_prediction):
            folder_path = self.pth_prediction / folder_name
            if folder_path.is_dir():
                label_val = len(self.group_mapping)
                self.group_mapping[folder_name] = (folder_name, label_val)

    def load_checkpoint(self):
        silent_checkpoints = self.cnn_wrapper.print_checkpoints_table(self.pth_checkpoint, print_table=False)
        if not silent_checkpoints:
            st.error("No checkpoints found!")
            return False
        
        if len(silent_checkpoints) == 1:
            checkpoint_file = silent_checkpoints[0][1]
        else:
            checkpoint_names = [f"{name} (Epoch {epoch})" for epoch, name, path in silent_checkpoints]
            selected = st.selectbox("Select checkpoint:", checkpoint_names, key="checkpoint_select")
            if not selected:
                return False
            checkpoint_file = silent_checkpoints[checkpoint_names.index(selected)][1]
        
        try:
            full_path = self.pth_checkpoint / checkpoint_file
            checkpoint_weights = torch.load(full_path)
            model_dict = self.cnn.state_dict()
            
            compatible_weights = {k: v for k, v in checkpoint_weights.items() 
                                if k in model_dict and model_dict[k].shape == v.shape}
            
            model_dict.update(compatible_weights)
            self.cnn.load_state_dict(model_dict)
            
            self.checkpoint_loaded = True
            self.checkpoint_name = full_path.stem
            st.success(f"Loaded {len(compatible_weights)} layers from {checkpoint_file}")
            return True
            
        except Exception as e:
            st.error(f"Error loading checkpoint: {e}")
            return False

    def extract_features(self):
        features, labels, image_paths = [], [], []
        
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
                for batch_idx, (images, batch_labels) in enumerate(dataloader):
                    images = images.to(self.device)
                    batch_features = self.cnn.features(images)
                    pooled = torch.nn.functional.adaptive_avg_pool2d(batch_features, (1, 1))
                    flattened = pooled.view(images.size(0), -1)
                    features.append(flattened.cpu().numpy())
                    
                    # Store image paths
                    if hasattr(dataloader.dataset, 'samples'):
                        batch_start_idx = batch_idx * dataloader.batch_size
                        batch_end_idx = batch_start_idx + len(batch_labels)
                        batch_paths = [dataloader.dataset.samples[i][0] for i in range(batch_start_idx, min(batch_end_idx, len(dataloader.dataset.samples)))]
                        image_paths.extend(batch_paths)
                    
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
                
        all_features = np.concatenate(features)
        all_labels = np.concatenate(labels)
        
        # Apply sampling based on interactive_sample_size setting
        if len(all_features) > self.interactive_sample_size:
            indices = np.random.choice(len(all_features), self.interactive_sample_size, replace=False)
            all_features = all_features[indices]
            all_labels = all_labels[indices]
            if image_paths and len(image_paths) == len(all_features):
                image_paths = [image_paths[i] for i in indices]
            st.info(f"📊 Sampled {self.interactive_sample_size} points from {len(all_features)} total")
        
        return all_features, all_labels, image_paths

    def run_reduction(self, method, reducer, features, labels, image_paths):
        st.write(f"🔄 Running {method}...")
        
        scaled_features = self.scaler.fit_transform(features)
        embedding = reducer.fit_transform(scaled_features)
        
        # Create DataFrame for plotting
        if self.mode == "groups":
            display_names = []
            for label_val in labels:
                found = False
                for folder_name, (name, val) in self.group_mapping.items():
                    if val == label_val:
                        display_names.append(name)
                        found = True
                        break
                if not found:
                    display_names.append(f"Label_{label_val}")
        else:
            display_names = [self.classes[label] for label in labels]
        
        df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'label': labels,
            'display_name': display_names,
            'image_path': image_paths if image_paths else [''] * len(labels)
        })
        
        return df, embedding

    def create_interactive_plot(self, df, method_name):
        """Create interactive Plotly plot with clickable points"""
        
        # Create the scatter plot
        fig = px.scatter(
            df, 
            x='x', 
            y='y',
            color='display_name',
            hover_data=['display_name', 'image_path'],
            title=f'{method_name} Visualization - Click points to see images',
            custom_data=['image_path', 'display_name']  # For click events
        )
        
        fig.update_traces(
            marker=dict(size=8, opacity=0.7),
            hovertemplate="<b>%{customdata[1]}</b><br>Click to view image<extra></extra>"
        )
        
        return fig

    def run(self):
        st.title("🎯 Dimensionality Reduction with Image Visualization")
        
        # Configuration info
        st.sidebar.header("⚙️ Configuration")
        st.sidebar.write(f"**Mode:** {self.mode}")
        st.sidebar.write(f"**Sample size:** {self.interactive_sample_size}")
        st.sidebar.write(f"**Thumbnail size:** {self.interactive_thumbnail_size}px")
        
        # Checkpoint loading
        if st.sidebar.button("📥 Load Checkpoint") and self.pth_checkpoint.exists():
            self.load_checkpoint()
        
        # Method selection
        selected_methods = []
        if self.use_umap:
            selected_methods.append("UMAP")
        if self.use_tsne:
            selected_methods.append("t-SNE") 
        if self.use_trimap:
            selected_methods.append("TriMAP")
        if self.use_pacmap:
            selected_methods.append("PaCMAP")
        
        if not selected_methods:
            st.error("❌ No dimensionality reduction methods enabled in settings!")
            return
        
        method = st.sidebar.selectbox("🎯 Select method:", selected_methods, key="method_select")
        
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            with st.spinner("🔍 Extracting features from CNN..."):
                features, labels, image_paths = self.extract_features()
            
            st.success(f"✅ Extracted {len(features)} features from {len(np.unique(labels))} classes")
            
            # Run selected reduction method
            if method == "UMAP":
                reducer = umap.UMAP(n_components=2, **self.umap_params)
            elif method == "t-SNE":
                reducer = TSNE(n_components=2, **self.tsne_params)
            elif method == "TriMAP":
                reducer = trimap.TRIMAP(n_dims=2, **self.trimap_params)
            elif method == "PaCMAP":
                reducer = PaCMAP(n_components=2, **self.pacmap_params)
            
            with st.spinner(f"🔄 Running {method}..."):
                df, embedding = self.run_reduction(method, reducer, features, labels, image_paths)
            
            # Store in session state for later access
            st.session_state.df = df
            st.session_state.embedding = embedding
            st.session_state.image_paths = image_paths
            st.session_state.method = method
            st.session_state.analysis_done = True
            
            # Display the interactive plot
            st.subheader(f"📊 {method} Visualization")
            fig = self.create_interactive_plot(df, method)
            
            # Use Plotly chart - click events will be handled by Streamlit
            st.plotly_chart(fig, use_container_width=True, key="main_plot")
            
            # Also show static plot
            st.subheader("🖼️ Static Plot")
            self.create_static_plot(embedding, labels, method)
            
            gc.collect()
            torch.cuda.empty_cache()

        # Image display section (appears after plot is created)
        if st.session_state.get('analysis_done', False):
            st.sidebar.header("👁️ Image Inspection")
            
            # Manual selection options
            st.sidebar.subheader("🔍 Manual Selection")
            
            # Option 1: Random sample
            if st.sidebar.button("🎲 Random Sample"):
                random_idx = np.random.randint(len(st.session_state.df))
                img_path = st.session_state.df.iloc[random_idx]['image_path']
                display_name = st.session_state.df.iloc[random_idx]['display_name']
                
                st.sidebar.write(f"**Random Sample: {display_name}**")
                try:
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        img.thumbnail((self.interactive_thumbnail_size, self.interactive_thumbnail_size))
                        st.sidebar.image(img, caption=os.path.basename(img_path))
                        st.sidebar.write(f"*Path:* {img_path}")
                    else:
                        st.sidebar.error(f"❌ Image not found: {img_path}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading image: {e}")
            
            # Option 2: By index
            point_idx = st.sidebar.number_input("📌 Point Index", 
                                              min_value=0, 
                                              max_value=len(st.session_state.df)-1, 
                                              value=0,
                                              key="point_index")
            if st.sidebar.button("🔍 Show Point"):
                img_path = st.session_state.df.iloc[point_idx]['image_path']
                display_name = st.session_state.df.iloc[point_idx]['display_name']
                
                st.sidebar.write(f"**Point {point_idx}: {display_name}**")
                try:
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        img.thumbnail((self.interactive_thumbnail_size, self.interactive_thumbnail_size))
                        st.sidebar.image(img, caption=os.path.basename(img_path))
                        st.sidebar.write(f"*Path:* {img_path}")
                    else:
                        st.sidebar.error(f"❌ Image not found: {img_path}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading image: {e}")

    def create_static_plot(self, embedding, labels, method_name):
        """Create static matplotlib plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.mode in ["train", "test"]:
            for class_idx, class_name in enumerate(self.classes):
                mask = labels == class_idx
                if np.sum(mask) > 0:
                    ax.scatter(
                        embedding[mask, 0], embedding[mask, 1],
                        label=class_name, alpha=0.7, s=40
                    )
        else:
            unique_labels = np.unique(labels)
            cmap = plt.cm.get_cmap('viridis')
            for i, label_val in enumerate(unique_labels):
                mask = labels == label_val
                display_name = f"Label_{label_val}"
                for folder_name, (name, val) in self.group_mapping.items():
                    if val == label_val:
                        display_name = name
                        break
                
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    color=cmap(i / len(unique_labels)),
                    label=display_name, alpha=0.7, s=40
                )
        
        ax.set_title(f"{method_name} Projection")
        ax.set_xlabel(f"{method_name} 1")
        ax.set_ylabel(f"{method_name} 2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        st.pyplot(fig)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        dimred = StreamlitDimRed(device)
        dimred.run()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()