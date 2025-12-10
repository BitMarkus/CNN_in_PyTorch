from pathlib import Path
import json
import torch
from collections import defaultdict
import gc
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
        self.cnn_wrapper = None

    #############################################################################################################
    # METHODS:

    # Record which images were used for validation vs testing
    # Record validation/test split only when validation comes from test dataset
    # Only records REAL images (filters out synthetic)
    def _record_val_test_split(self, dataset_idx):
        if self.val_from_test_split is False:
            return

        try:
            # Get both datasets (they should share the same underlying dataset)
            val_loader = self.ds.ds_test_for_val
            test_loader = self.ds.ds_test_for_test
            
            # Verify the expected structure
            if not all(hasattr(loader, 'sampler') and hasattr(loader.sampler, 'indices') 
                for loader in [val_loader, test_loader]):
                raise ValueError("DataLoaders don't have proper sampler indices")

            # Get the dataset
            dataset = val_loader.dataset
            
            # Check if it's a Subset and get the underlying dataset
            if hasattr(dataset, 'dataset'):  # It's a Subset
                full_dataset = dataset.dataset
                # Map Subset indices to original dataset indices
                val_indices = [dataset.indices[i] for i in val_loader.sampler.indices]
                test_indices = [dataset.indices[i] for i in test_loader.sampler.indices]
            else:
                full_dataset = dataset
                val_indices = list(val_loader.sampler.indices)
                test_indices = list(test_loader.sampler.indices)
            
            # Convert to sets for proper comparison
            val_indices_set = set(val_indices)
            test_indices_set = set(test_indices)
            
            # Create guaranteed unique splits
            true_val_indices = val_indices_set - test_indices_set
            true_test_indices = test_indices_set - val_indices_set
            
            # Report any issues
            if len(val_indices_set & test_indices_set) > 0:
                print(f"Warning: Corrected {len(val_indices_set & test_indices_set)} overlapping images")
            
            # Get image paths from the guaranteed unique sets
            val_images = [full_dataset.samples[i][0] for i in true_val_indices]
            test_images = [full_dataset.samples[i][0] for i in true_test_indices]
            
            # Function to identify synthetic vs real
            def is_synthetic_image(path):
                filename = Path(path).name
                # Synthetic images start with 's' followed by numbers
                return filename.startswith('s') and filename[1:2].isdigit()
            
            # FILTER OUT SYNTHETIC IMAGES - ALWAYS (only record real images)
            original_val_count = len(val_images)
            original_test_count = len(test_images)
            
            real_val_images = [p for p in val_images if not is_synthetic_image(p)]
            real_test_images = [p for p in test_images if not is_synthetic_image(p)]
            
            synthetic_val_filtered = original_val_count - len(real_val_images)
            synthetic_test_filtered = original_test_count - len(real_test_images)
            
            if synthetic_val_filtered > 0 or synthetic_test_filtered > 0:
                print(f"Filtered out {synthetic_val_filtered} synthetic images from validation set")
                print(f"Filtered out {synthetic_test_filtered} synthetic images from test set")
            
            # Verify we still have enough images
            if len(real_val_images) == 0:
                print("WARNING: No real images left in validation set!")
            if len(real_test_images) == 0:
                print("WARNING: No real images left in test set!")
            
            # Prepare split info with ONLY REAL images
            split_info = {
                'validation': {
                    'WT': sorted(list({Path(p).name for p in real_val_images if 'WT' in str(p)})),
                    'KO': sorted(list({Path(p).name for p in real_val_images if 'KO' in str(p)}))
                },
                'test': {
                    'WT': sorted(list({Path(p).name for p in real_test_images if 'WT' in str(p)})),
                    'KO': sorted(list({Path(p).name for p in real_test_images if 'KO' in str(p)}))
                },
                'metadata': {
                    'total_images_found': original_val_count + original_test_count,
                    'real_images_used': len(real_val_images) + len(real_test_images),
                    'synthetic_filtered': synthetic_val_filtered + synthetic_test_filtered,
                    'note': 'Only real images are recorded for validation/testing'
                }
            }
            
            # Write split info text to json file
            output_path = Path(self.acv_results_dir) / f"dataset_{dataset_idx}" / "split_info.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(split_info, f, indent=2)
            
            print(f"Recorded {len(real_val_images)} real validation images and {len(real_test_images)} real test images")
                
        except Exception as e:
            print(f"Failed to record splits: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    # Reinitializes the model by creating a fresh instance
    def reset_model(self):
        print("Creating fresh model instance for new training...")
        # Force a new model load
        self.cnn_wrapper.model = self.cnn_wrapper.load_model(self.device)
        # Move to device (if not already done in load_model)
        self.cnn_wrapper.model = self.cnn_wrapper.model.to(self.device)

    #############################################################################################################
    # CALL:

    def __call__(self):

        # Clean up folders train and test in the data directory
        print("\nCleaning up old train and test data...")
        self.ds_gen.cleanup(self.data_dir) 
        print("Cleanup finished.")  

        # Ensure the acv_results directory exists
        self.acv_results_dir.mkdir(parents=True, exist_ok=True)

        # Generate list of cross validation datasets
        configs = self.ds_gen.get_dataset_configs()

        # Iterate over each dataset
        for config in configs:
            
            # Create FRESH dataset object for this fold
            self.ds = Dataset()
            
            # Create FRESH model wrapper for this fold  
            self.cnn_wrapper = CNN_Model() 
            self.reset_model()
            
            # Clean up data folders for this fold
            self.ds_gen.cleanup(self.data_dir)
            # Clear GPU memory
            torch.cuda.empty_cache()  

            ##################
            # Create dataset #
            ##################

            print(f"\n>> PROCESSING DATASET {config['dataset_idx']} OF {len(configs)}:")

            print(f"Cell line for testing WT group: {config['test_wt']}")
            print(f"Cell line for testing KO group: {config['test_ko']}")

            print(f"\n> Create dataset {config['dataset_idx']}...")
            dataset_dir = self.ds_gen.generate_dataset(**config)

            # Create subfolders
            checkpoint_dir = dataset_dir / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            plot_dir = dataset_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            print(f"Dataset {config['dataset_idx']} successfully created.")

            ######################
            # Load train dataset #
            ######################

            print(f"\n> Load dataset {config['dataset_idx']} for training...")
            
            # Check for correct settings
            self.ds.validate_validation_settings()

            # Load datasets
            self.ds.load_training_dataset()
            
            if self.val_from_test_split is not False: 
                self.ds.load_test_dataset()
                self._record_val_test_split(config['dataset_idx']) 
            
            # Print dataset info
            self.ds.print_dataset_info()
            if(self.ds.ds_loaded):
                print(f"Dataset {config['dataset_idx']} successfully loaded.")
                print(f"Number training images/batches: {self.ds.num_train_img}/{self.ds.num_train_batches}")
                print(f"Number validation images/batches: {self.ds.num_val_img}/{self.ds.num_val_batches}") 

            ####################
            # Train on dataset #
            #################### 

            self.train = Train(self.cnn_wrapper, self.ds, self.device)
            print(f"\n> Start training on dataset {config['dataset_idx']}...")
            self.train.train(checkpoint_dir, plot_dir)
            print(f"\nTraining on dataset {config['dataset_idx']} successfully finished.")

            ################
            # Testing Loop # 
            ################

            # Always load fresh test dataset for final evaluation
            print(f"\n> Load test images for final evaluation...")

            # Load split info if it exists
            split_file = Path(self.acv_results_dir) / f"dataset_{config['dataset_idx']}" / "split_info.json"
            real_test_images = None

            if split_file.exists():
                with open(split_file, 'r') as f:
                    split_info = json.load(f)
                
                # Check if this split_info contains filtered data (has metadata about synthetic filtering)
                if 'metadata' in split_info and split_info['metadata'].get('synthetic_filtered', 0) > 0:
                    # This dataset has synthetic images that were filtered
                    real_test_images = set(split_info['test']['WT'] + split_info['test']['KO'])
                    print(f"Found {len(real_test_images)} real test images (synthetic data detected)")
                else:
                    # Pure real data - use all images
                    print(f"Using all test images (pure real data detected)")
                    real_test_images = None
            else:
                print("WARNING: No split_info.json found. Loading all test images.")
                real_test_images = None

            # Load appropriate dataset
            if real_test_images is not None:
                # Load only real images (synthetic data case)
                success = self.ds.load_real_test_dataset_only(real_test_images)
            else:
                # Load all images (pure real data case)
                success = self.ds.load_real_test_dataset_only()  # No filter
                
            if success:
                test_dataset_to_use = self.ds.ds_test_real_only
                data_type = "real-only" if real_test_images is not None else "all (pure real)"
            else:
                print("Failed to load filtered dataset, using standard loading")
                self.ds.load_test_dataset()
                test_dataset_to_use = self.ds.ds_test
                data_type = "all (fallback)"
                self.ds.num_pred_real = self.ds.num_pred_img  # Estimate

            # Clear output
            print(f"Test images for dataset {config['dataset_idx']} successfully loaded.")
            print(f"Evaluating on {self.ds.num_pred_real} test images ({data_type})")
            print(f"Batch size: 1")

            # Load checkpoint
            checkpoint_list = self.cnn_wrapper.get_checkpoints_list(checkpoint_dir)
            if not checkpoint_list:
                print("No checkpoint for this dataset!")
            else:
                for checkpoint_file in checkpoint_list:
                    print(f"\n> Load weight file {checkpoint_file[1]} for dataset {config['dataset_idx']}...")
                    self.cnn_wrapper.load_weights(checkpoint_dir, checkpoint_file[1])

                    print('\n> Starting prediction on REAL test images only...')
                    
                    # Use the filtered dataset
                    _, cm = self.cnn_wrapper.predict(test_dataset_to_use)

                    # Plot confusion matrix and results
                    fn.plot_confusion_matrix(cm, self.class_list, plot_dir, chckpt_name=checkpoint_file[1], show_plot=False, save_plot=True)
                    fn.save_confusion_matrix_results(cm, self.class_list, plot_dir, chckpt_name=checkpoint_file[1])

                    # Load confusion matrix results
                    loaded_results = fn.load_confusion_matrix_results(plot_dir, file_name=checkpoint_file[1])
                    print(f"Overall accuracy: {(loaded_results['overall_accuracy']*100):.2f}%")
                    print(f"WT accuracy: {(loaded_results['class_accuracy']['WT']*100):.2f}%")
                    print(f"KO accuracy: {(loaded_results['class_accuracy']['KO']*100):.2f}%")

                    print(f'Prediction successfully finished. Confusion matrix and results saved to {plot_dir}.')    

            print(f"Completed dataset {config['dataset_idx']}. Moving to next fold...")    

        # Final cleanup (once at the end)
        print("\nCleaning up all temporary data...")
        self.ds_gen.cleanup(self.data_dir)
        print("Cross-validation complete.")