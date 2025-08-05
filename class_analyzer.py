import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Subset
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
        
    def get_folder_indices(self, dataset, folder_name):
        """Return indices of samples belonging to a specific folder"""
        return [i for i, (path, _) in enumerate(dataset.samples) 
                if Path(path).parent.name == folder_name]
    
    def predict_folder(self, folder_name):
        """Predict images from a specific folder"""
        folder_indices = self.get_folder_indices(self.ds.ds_pred.dataset, folder_name)
        if not folder_indices:
            print(f"No images found for folder: {folder_name}")
            return None

        subset = Subset(self.ds.ds_pred.dataset, folder_indices)
        loader = DataLoader(subset, batch_size=self.ds.batch_size_pred)

        class_counts = {class_name: 0 for class_name in self.classes}
        
        self.cnn.eval()
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Predicting {folder_name}", unit="img")
            for images, _ in pbar:
                outputs = self.cnn(images.to(self.device))
                _, predicted = torch.max(outputs, 1)
                
                for pred in predicted:
                    class_name = self.classes[pred.item()]
                    class_counts[class_name] += 1
                
                pbar.set_postfix({
                    'total': len(folder_indices),
                    **{k: v for k, v in class_counts.items() if v > 0}
                })

        total = len(folder_indices)
        return {
            'Folder': folder_name,
            'Total Images': total,
            **{f"{k}_Count": v for k, v in class_counts.items()},
            **{f"{k}_Percentage": (v/total)*100 for k, v in class_counts.items()},
            'Most Likely Class': max(class_counts.items(), key=lambda x: x[1])[0]
        }

    def analyze_prediction_folder(self):
        if not self.load_checkpoint():
            print("WARNING: Using untrained weights!")
            self.loaded_checkpoint_name = "untrained"

        # Debug:
        self.debug_transforms()

        # Get all class folders in prediction directory
        class_folders = [d.name for d in self.pth_prediction.iterdir() 
                        if d.is_dir() and d.name in self.classes]
        
        if not class_folders:
            print(f"No valid class folders found in {self.pth_prediction}")
            return None

        print(f"\nAnalyzing {len(class_folders)} class folders...")
        results = []
        
        for folder in class_folders:
            try:
                print(f"\n> Processing folder: {folder}")
                result = self.predict_folder(folder)
                if result:
                    results.append(result)
                    # Print folder results
                    print(f"\nRESULTS for {folder}:")
                    print(f"Total images: {result['Total Images']}")
                    for cls in self.classes:
                        print(f"{cls}: {result[f'{cls}_Count']} ({result[f'{cls}_Percentage']:.2f}%)")
                    print(f"Most likely class: {result['Most Likely Class']}")
            except Exception as e:
                print(f"Error processing {folder}: {str(e)}")

        if not results:
            print("No valid results generated")
            return None

        # Save results
        df = pd.DataFrame(results)
        output_path = self.pth_prediction / f"results_{self.loaded_checkpoint_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved results to: {output_path}")
        return df