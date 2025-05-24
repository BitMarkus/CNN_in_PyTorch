import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
import torchvision
from tqdm import tqdm
# Own modules
from dataset import Dataset
from settings import setting
from custom_model import Custom_CNN_Model
from model import CNN_Model

class ClassAnalyzer:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device):
        self.device = device
        # Settings parameters
        self.pth_prediction = os.path.abspath(setting['pth_prediction'])
        self.pth_checkpoint = os.path.abspath(setting['pth_checkpoint'])
        self.classes = setting['classes']
        
        # Initialize dataset to get transforms
        self.ds = Dataset()
        self.transform = self.ds.get_transformer()
        if not self.transform:
            raise ValueError("Failed to get image transformer")
        
        # Create model wrapper
        print(f"Creating new network...")
        if setting["cnn_type"] == "custom":
            self.cnn = Custom_CNN_Model()
            self.cnn.model = self.cnn
        else:
            self.cnn = CNN_Model()
            self.cnn.model = self.cnn.load_model(self.device)

        # Ensure everything is on the right device
        self.cnn.model = self.cnn.model.to(self.device)
        if hasattr(self.cnn, 'to'):
            self.cnn = self.cnn.to(self.device)
        print(f"Network {self.cnn.cnn_type} was successfully created.")

        # Initialize checkpoint flag
        self.checkpoint_loaded = False
        self.loaded_checkpoint_name = None

    #############################################################################################################
    # METHODS:

    def load_checkpoint(self):
        """Load model weights from a selected checkpoint"""
        # First get checkpoints without printing table
        silent_checkpoints = self.cnn.print_checkpoints_table(self.pth_checkpoint, print_table=False)
        
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
            self.cnn.print_checkpoints_table(self.pth_checkpoint)  # prints table
            checkpoint_file = self.cnn.select_checkpoint(silent_checkpoints, "Select a checkpoint: ")
            if not checkpoint_file:
                return False
        
        try:
            full_path = os.path.join(self.pth_checkpoint, checkpoint_file)
            self.cnn.model.load_state_dict(torch.load(full_path))
            self.checkpoint_loaded = True
            self.loaded_checkpoint_name = os.path.splitext(checkpoint_file)[0]
            print(f"Successfully loaded weights from {checkpoint_file}")
            return True
        except FileNotFoundError as e:
            print(f"\nError loading checkpoint: {str(e)}")
            print(f"Full path attempted: {full_path}")
            return False
        except Exception as e:
            print(f"\nError loading checkpoint: {str(e)}")
            return False
    
    def create_prediction_loader(self, folder_path):
        """Create a DataLoader that properly handles image loading and transformation"""
        class FlatImageFolder(torch.utils.data.Dataset):
            def __init__(self, folder, transform=None):
                self.folder = folder
                self.transform = transform
                self.image_files = [f for f in os.listdir(folder) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not self.image_files:
                    raise ValueError(f"No images found in {folder}")

            def __len__(self):
                return len(self.image_files)

            def __getitem__(self, idx):
                img_path = os.path.join(self.folder, self.image_files[idx])
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, 0  # Dummy label

        dataset = FlatImageFolder(folder_path, transform=self.transform)
        return DataLoader(dataset, batch_size=self.ds.batch_size_pred)
    
    def predict_folder(self, folder_path):
        try:
            prediction_loader = self.create_prediction_loader(folder_path)
            
            class_counts = {class_name: 0 for class_name in self.classes}
            folder_name = os.path.basename(folder_path)
            total_images = 0
            
            self.cnn.model.eval()
            with torch.no_grad():
                # Initialize tqdm progress bar that will stay after completion
                pbar = tqdm(prediction_loader, desc=f"Predicting {folder_name}", 
                           unit="img", dynamic_ncols=True)
                
                for images, _ in pbar:
                    images = images.to(self.device)
                    outputs = self.cnn.model(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    batch_size = images.size(0)
                    for pred in predicted:
                        class_name = self.classes[pred.item()]
                        class_counts[class_name] += 1
                    
                    total_images += batch_size
                    pbar.set_postfix({
                        'total': total_images,
                        **{k: v for k, v in class_counts.items() if v > 0}
                    })
            
            # Calculate percentages
            class_percentages = {k: (v/total_images)*100 for k,v in class_counts.items()}
            most_likely_class = max(class_counts.items(), key=lambda x: x[1])[0]
            
            # Print results to console
            print(f"> RESULTS:")
            print(f"Total images processed: {total_images}")
            print("Class distribution:")
            for class_name, count in class_counts.items():
                print(f"- {class_name}: {count} images ({class_percentages[class_name]:.2f}%)")
            print(f"Most likely class: {most_likely_class}")
            
            return {
                'Folder': folder_name,
                'Most Likely Class': most_likely_class,
                'Total Images': total_images,
                **{f"{class_name}_Count": count for class_name, count in class_counts.items()},
                **{f"{class_name}_Percentage": class_percentages[class_name] for class_name in class_counts}
            }
            
        except Exception as e:
            raise ValueError(f"Error processing folder {folder_path}: {str(e)}")

    def analyze_prediction_folder(self):
        """Analyze all folders in the prediction directory"""

        # Load checkpoint
        if not self.load_checkpoint():
            print("WARNING: Proceeding with untrained weights!\n")
            self.loaded_checkpoint_name = "untrained"

        subfolders = [f.path for f in os.scandir(self.pth_prediction) if f.is_dir()]
        
        if not subfolders:
            print(f"\nNo subfolders found in {self.pth_prediction}")
            return None
        
        print("\n>> Starting image classification analysis")
        print(f"Found {len(subfolders)} folders to process")
        
        detailed_results = []
        
        for folder in subfolders:
            try:
                print(f"\n> PROCESSING FOLDER: {os.path.basename(folder)}")
                folder_result = self.predict_folder(folder)
                detailed_results.append(folder_result)
            except Exception as e:
                print(f"Error processing folder {folder}: {str(e)}")
                continue
        
        if not detailed_results:
            print("\nNo valid folders were processed successfully")
            return None
        
        # Create and save detailed DataFrame
        detailed_df = pd.DataFrame(detailed_results)
        
        # Generate output filename based on checkpoint name
        output_filename = f"result_{self.loaded_checkpoint_name}.csv"
        output_path = os.path.join(self.pth_prediction, output_filename)
        
        detailed_df.to_csv(output_path, index=False)
        
        print(f"\nAnalysis complete. Results saved to: {output_path}")
        
        return detailed_df