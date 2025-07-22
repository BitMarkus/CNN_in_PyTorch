from pathlib import Path
import warnings
# Own modules
from dataset import Dataset
from settings import setting
import functions as fn
from train import Train
from dataset_gen import DatasetGenerator
from model import CNN_Model

class AutoCrossValidation:

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, device):

        # Passed parameters
        self.device = device

        # Settings parameters
        self.wt_lines = setting['wt_lines']
        self.ko_lines = setting['ko_lines']
        self.acv_results_dir = setting['pth_acv_results']
        self.data_dir = setting['pth_data']
        self.class_list = setting["classes"] 

        # Validation split settings (False or percentage 0.0-1.0)
        # Validation split from training dataset
        self.val_from_train_split = setting["ds_val_from_train_split"]
        # Validation split from test dataset
        self.val_from_test_split = setting["ds_val_from_test_split"]  

        # Objects
        # Create a dataset generation object (in automatic cross validation mode)
        self.ds_gen = DatasetGenerator(mode = "acv")
        # Create a dataset object
        self.ds = Dataset()
        # Create model wrapper
        self.cnn_wrapper = CNN_Model()  

    #############################################################################################################
    # METHODS:

    def __call__(self):

        # Clean up folders train and test in the data directory (old training and test data)
        print("\nCleaning up old train and test data...")
        self.ds_gen.cleanup(self.data_dir) 
        print("Cleanup finished.")  

        # Ensure the acv_results directory exists
        self.acv_results_dir.mkdir(parents=True, exist_ok=True)

        # Generate list of cross validation datasets
        configs = self.ds_gen.get_dataset_configs()

        # Iterate over each dataset
        for config in configs:
            # Clean up before creating new dataset to ensure fresh start
            self.ds_gen.cleanup(self.data_dir)

            ##################
            # Create dataset #
            ##################
            print(f"\n>> PROCESSING DATASET {config['dataset_idx']} OF {len(configs)}:")

            print(f"Cell line for testing WT group: {config['test_wt']}")
            print(f"Cell line for testing KO group: {config['test_ko']}")

            print(f"\n> Create dataset {config['dataset_idx']}...")
            dataset_dir = self.ds_gen.generate_dataset(**config) # Unpacks dict as kwargs

            # Create a subfolder for checkpoints
            ckeckpoint_dir = dataset_dir / "checkpoints"
            ckeckpoint_dir.mkdir(exist_ok=True)
            # Create a subfolder for plots
            plot_dir = dataset_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            print(f"Dataset {config['dataset_idx']} successfully created.")

            ######################
            # Load train dataset #
            ######################
            print(f"\n> Load dataset {config['dataset_idx']} for training...")
            
            # Check for correct settings in settings file
            self.ds.validate_validation_settings()
            # Load training dataset (always needed)
            self.ds.load_training_dataset()
            # Load test dataset if validation should come from test images
            if self.ds.validation_from_test:
                self.ds.load_test_dataset()
            
            # Print dataset info
            self.ds.print_dataset_info()
            
            if(self.ds.ds_loaded):
                print(f"Dataset {config['dataset_idx']} successfully loaded.")
                print(f"Number training images/batches: {self.ds.num_train_img}/{self.ds.num_train_batches}")
                print(f"Number validation images/batches: {self.ds.num_val_img}/{self.ds.num_val_batches}") 

            ##############
            # Create CNN #
            ##############
            print(f"Creating new {self.cnn_wrapper.cnn_type} network...")
            self.cnn = self.cnn_wrapper.load_model(self.device).to(self.device)
            print("Network successfully created.")   

            ####################
            # Train on dataset #
            ####################
            self.train = Train(self.cnn_wrapper, self.ds, self.device)
            print(f"\n> Start training on dataset {config['dataset_idx']}...")
            self.train.train(ckeckpoint_dir, plot_dir)
            print(f"\nTraining on dataset {config['dataset_idx']} successfully finished.")

            ################
            # Testing Loop # 
            ################
            # Always load fresh test dataset for final evaluation
            print(f"\n> Load test images for final evaluation...")
            self.ds.load_test_dataset()
            print(f"Test images for dataset {config['dataset_idx']} successfully loaded.")             
            print(f"Number test images/batch size: {self.ds.num_pred_img}/1")
            
            checkpoint_list = self.cnn_wrapper.get_checkpoints_list(ckeckpoint_dir)
            if not checkpoint_list:
                print("No checkpoint for this dataset!")
            else:
                for checkpoint_file in checkpoint_list:
                    print(f"\n> Load weight file {checkpoint_file[1]} for dataset {config['dataset_idx']}...")
                    self.cnn_wrapper.load_weights(ckeckpoint_dir, checkpoint_file[1]) 

                    print('\n> Starting prediction...')
                    _, cm = self.cnn_wrapper.predict(self.ds.ds_test)

                    # Plot confusion matrix and results
                    fn.plot_confusion_matrix(cm, self.class_list, plot_dir, chckpt_name=checkpoint_file[1], show_plot=False, save_plot=True)
                    fn.save_confusion_matrix_results(cm, self.class_list, plot_dir, chckpt_name=checkpoint_file[1])

                    # Load confusion matrix results
                    loaded_results = fn.load_confusion_matrix_results(plot_dir, file_name=checkpoint_file[1])
                    print(f"Overall accuracy: {(loaded_results['overall_accuracy']*100):.2f}%")
                    print(f"WT accuracy: {(loaded_results['class_accuracy']['WT']*100):.2f}%")
                    print(f"KO accuracy: {(loaded_results['class_accuracy']['KO']*100):.2f}%")

                    print(f'Prediction successfully finished. Confusion matrix and results saved to {plot_dir}.')

        # FINAL CLEANUP (once at the end)
        print("\nCleaning up all temporary data...")
        self.ds_gen.cleanup(self.data_dir)
        print("Cross-validation complete.")