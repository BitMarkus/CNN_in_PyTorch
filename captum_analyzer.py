import torch
from tqdm import tqdm
from pathlib import Path
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
# Own modules
from settings import setting
from dataset import Dataset
from custom_model import Custom_CNN_Model
from model import CNN_Model

class CaptumAnalyzer:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device):
        self.device = device
        # Settings parameters
        self.pth_prediction = setting['pth_prediction']
        self.pth_checkpoint = setting['pth_checkpoint']
        self.classes = setting['classes']
        self.img_channels = setting['img_channels']
        self.pred_batch_size = 1
        
        # Initialize dataset to get transforms
        self.ds = Dataset()
        self.transform = self.ds.get_transformer_test()
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

    #############################################################################################################
    # METHODS

    def predict_captum(self, dataset, device):
        # Set non-interactive backend at the start
        import matplotlib
        matplotlib.use('Agg')  # Set before other matplotlib imports
        import matplotlib.pyplot as plt

        self.cnn.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataset)):
                batch_paths = [Path(dataset.dataset.samples[i][0])  # Convert to Path objects
                            for i in range(batch_idx * dataset.batch_size,
                                        min((batch_idx + 1) * dataset.batch_size, len(dataset.dataset)))]
                
                for img_idx in range(len(images)):
                    try:
                        torch.cuda.empty_cache()
                        
                        img = images[img_idx].unsqueeze(0).to(device)
                        label = labels[img_idx].to(device)
                        img_path = batch_paths[img_idx]
                        
                        # Create output folder matching original subfolder structure
                        relative_path = img_path.relative_to(self.pth_prediction)
                        output_folder = self.pth_prediction / f"{relative_path.parent.name}_captum"
                        output_folder.mkdir(exist_ok=True, parents=True)
                        
                        # Compute attributions
                        ig = IntegratedGradients(self.cnn.model)
                        attributions = ig.attribute(
                            img,
                            target=label.item(),
                            n_steps=10,
                            internal_batch_size=1
                        )
                        
                        # Convert to numpy
                        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        attr_np = attributions.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                        
                        # Create visualization
                        plt.ioff()  # Turn off interactive mode
                        fig, ax = viz.visualize_image_attr(
                            attr_np,
                            img_np,
                            method="blended_heat_map",
                            sign="positive",
                            show_colorbar=True,
                            cmap="inferno",
                            alpha_overlay=0.8,
                            title=f"Attribution for Class {label.item()}",
                            fig_size=(8, 8)
                        )
                        
                        # Save to correct location
                        output_path = output_folder / f"{img_path.stem}_captum.png"
                        fig.savefig(output_path, bbox_inches='tight', dpi=100)
                        plt.close(fig)
                        
                        # print(f"Saved: {output_path}")  # Debug output
                        
                        # Clean up
                        del img, attributions, attr_np, fig, ax
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"Error processing image {batch_paths[img_idx]}: {str(e)}")
                        continue

    #############################################################################################################
    # CALL

    def __call__(self):

        # Load dataset for prediction
        print("\nLoading dataset for prediction:") 
        self.ds.load_pred_dataset()
        if(self.ds.ds_loaded):
            print("Dataset successfully loaded!")

        # Load weights
        print("\nLoading weights:")
        self.cnn.load_checkpoint()

        # Iterate over images and predict
        self.predict_captum(self.ds.ds_pred, self.device)

        


    
