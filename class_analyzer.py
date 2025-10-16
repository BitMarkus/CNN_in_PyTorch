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
        
    # Return indices of samples belonging to a specific folder
    def get_folder_indices(self, dataset, folder_name):
        return [i for i, (path, _) in enumerate(dataset.samples) 
                if Path(path).parent.name == folder_name]
    
    # Predict images from a specific folder
    def predict_folder(self, folder_name, rename_images=False):
        folder_indices = self.get_folder_indices(self.ds.ds_pred.dataset, folder_name)
        if not folder_indices:
            print(f"No images found for folder: {folder_name}")
            return None

        subset = Subset(self.ds.ds_pred.dataset, folder_indices)
        loader = DataLoader(subset, batch_size=self.ds.batch_size_pred)

        class_counts = {class_name: 0 for class_name in self.classes}
        
        # For individual image tracking and renaming
        image_predictions = []
        
        self.cnn.eval()
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"Predicting {folder_name}", unit="img")
            for batch_idx, (images, _) in enumerate(pbar):
                outputs = self.cnn(images.to(self.device))
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
                    
                    # Store prediction info
                    image_predictions.append({
                        'original_path': original_path,
                        'predicted_class': class_name,
                        'confidence': confidence,
                        'confidence_percentage': int(confidence * 100)
                    })
                    
                    # Count for overall statistics
                    class_counts[class_name] += 1
                
                pbar.set_postfix({
                    'total': len(folder_indices),
                    **{k: v for k, v in class_counts.items() if v > 0}
                })

        # Rename images if requested
        if rename_images:
            self._rename_images_with_confidence(image_predictions, folder_name)

        total = len(folder_indices)
        return {
            'Folder': folder_name,
            'Total Images': total,
            **{f"{k}_Count": v for k, v in class_counts.items()},
            **{f"{k}_Percentage": (v/total)*100 for k, v in class_counts.items()},
            'Most Likely Class': max(class_counts.items(), key=lambda x: x[1])[0]
        }

    # Rename images by adding confidence percentage to filename
    def _rename_images_with_confidence(self, image_predictions, folder_name):

        renamed_count = 0
        
        for pred_info in image_predictions:
            original_path = Path(pred_info['original_path'])
            confidence_pct = pred_info['confidence_percentage']
            predicted_class = pred_info['predicted_class']
            
            # Create new filename: original_stem_conf{confidence}_class{predicted_class}.extension
            original_stem = original_path.stem.rstrip('_') # remove eventual underscores from the filename
            new_stem = f"{original_stem}_conf{confidence_pct}-{predicted_class}"
            new_filename = f"{new_stem}{original_path.suffix}"
            new_path = original_path.parent / new_filename
            
            try:
                # Rename the file
                original_path.rename(new_path)
                renamed_count += 1
                # print(f"Renamed: {original_path.name} -> {new_filename}")
            except Exception as e:
                # print(f"Error renaming {original_path.name}: {str(e)}")
                pass
        
        print(f"Successfully renamed {renamed_count}/{len(image_predictions)} images in folder '{folder_name}'")

    # Analyze all folders in prediction directory
    # rename_images (bool): If True, rename images with confidence scores and predicted class
    def analyze_prediction_folder(self, rename_images=False):

        if not self.load_checkpoint():
            print("WARNING: Using untrained weights!")
            self.loaded_checkpoint_name = "untrained"

        # Get ALL folders in prediction directory
        all_folders = [d.name for d in self.pth_prediction.iterdir() if d.is_dir()]
        
        if not all_folders:
            print(f"No folders found in {self.pth_prediction}")
            return None

        print(f"\nAnalyzing {len(all_folders)} folders...")
        if rename_images:
            print("Image renaming with confidence scores is ENABLED")
        else:
            print("Image renaming with confidence scores is DISABLED")
        
        results = []
        
        for folder in all_folders:
            try:
                print(f"\n> Processing folder: {folder}")
                result = self.predict_folder(folder, rename_images=rename_images)
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