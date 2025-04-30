import os
import shutil
import torch
from tqdm import tqdm
from settings import setting
import functions as fn
from train import Train

class AutomateAnalysis:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device, cnn, dataset):

        # Passed parameters
        self.device = device
        self.cnn = cnn
        self.ds = dataset

    #############################################################################################################
    # METHODS:

    # Prepare training and prediction data
    def _prepare_data(self, dataset_path):
        # Clear and create directories
        for folder in ["WT", "KO"]:
            for path in [setting["pth_train"], setting["pth_test"]]:
                shutil.rmtree(os.path.join(path, folder), ignore_errors=True)
                os.makedirs(os.path.join(path, folder), exist_ok=True)        
        # Copy data
        for phase, src_dir, dest_dir in [
            ("train", os.path.join(dataset_path, "train"), setting["pth_train"]),
            ("prediction", os.path.join(dataset_path, "prediction"), setting["pth_test"])
        ]:
            for folder in ["WT", "KO"]:
                src = os.path.join(src_dir, folder)
                dst = os.path.join(dest_dir, folder)
                for img in os.listdir(src):
                    shutil.copy2(os.path.join(src, img), dst)

    # Execute training process
    def _train_model(self):
        self.ds.load_training_dataset()
        train = Train(self.cnn, self.ds)
        history = train.train()
        train.plot_metrics(history, setting["pth_plots"], show_plot=False, save_plot=True)

    # Evaluate all checkpoints on test data
    def _evaluate_checkpoints(self, dataset_path):
        self.ds.load_prediction_dataset()
        result_dir = os.path.join(dataset_path, "result")
        os.makedirs(result_dir, exist_ok=True)
        # Iterate over checkpoints
        for checkpoint in os.listdir(setting["pth_checkpoint"]):
            if checkpoint.endswith('.model'):
                # Load weights and predict
                self.cnn.load_weights(os.path.join(setting["pth_checkpoint"], checkpoint))
                _, cm = self.cnn.predict(self.ds.ds_pred)    
                # Save confusion matrix
                epoch = f"e{checkpoint.split('_e')[1].split('_')[0]}"
                fn.plot_confusion_matrix(
                    cm, 
                    self.cnn.get_class_list(), 
                    setting["pth_plots"], 
                    show_plot=False, 
                    save_plot=True,
                    filename=f"cm_{epoch}"
                )     
                # Copy files to results
                for f in [checkpoint, f"cm_{epoch}.png"]:
                    src = os.path.join(setting["pth_plots"] if 'cm_' in f else setting["pth_checkpoint"], f)
                    dst = os.path.join(result_dir, f)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)

    # Clean up working directories
    def _cleanup(self):
        # Cleanup training and prediction images
        for folder in [setting["pth_train"], setting["pth_test"]]:
            for cell_type in ["WT", "KO"]:
                shutil.rmtree(os.path.join(folder, cell_type), ignore_errors=True)
                os.makedirs(os.path.join(folder, cell_type), exist_ok=True)
        # Cleanup checkpoints and plots        
        for folder in [setting["pth_checkpoint"], setting["pth_plots"]]:
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

    # Execute the complete automated analysis pipeline
    def run_full_analysis(self):
        try:
            for dataset_idx in tqdm(range(1, 21), desc="Processing datasets"):
                dataset_path = os.path.join(setting["pth_ds_gen_output"], f"dataset_{dataset_idx}")
                
                self._prepare_data(dataset_path)
                self._train_model()
                self._evaluate_checkpoints(dataset_path)
                self._cleanup(dataset_path)
                
                print(f"\nCompleted dataset {dataset_idx}/20 - Results saved to {dataset_path}/result")
            
            print("\nAutomated analysis completed successfully!")
            return True
            
        except Exception as e:
            print(f"\nError during automated analysis: {str(e)}")
            return False

"""
elif(menu1 == 10):  # Add new menu option
    print("\n:AUTOMATED ANALYSIS:")
    from automate_analysis import AutomateAnalysis
    analyzer = AutomateAnalysis()
    if analyzer.run_full_analysis():
        print("All datasets processed successfully!")
    else:
        print("Automated analysis encountered errors")
"""