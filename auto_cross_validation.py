"""
AUTOMATIC CROSS-VALIDATION SYSTEM FOR CELL LINE CLASSIFICATION

================================================================================
PURPOSE:
Performs leave-one-WT-line-out AND leave-one-KO-line-out cross-validation
for binary classification of fibroblast images (WT = healthy vs KO = diseased).

KEY CONCEPT:
TRAIN on N-1 cell lines, TEST on the 1 left-out cell line.
For 5 WT lines + 4 KO lines = 20 total folds (5 × 4 combinations).

================================================================================
EXACT FOLDER STRUCTURE REQUIREMENTS:

MODE 1: "train_data_source": "synthetic_only" (Recommended for your use case)
------------------------------------------------
dataset_gen/
├── input_synthetic/           # SYNTHETIC IMAGES ONLY - FOR TRAINING
│   ├── WT_1618-02/           # Wild-type cell line 1
│   │   ├── s001.jpg          # Synthetic images start with 's' + numbers
│   │   ├── s002.jpg
│   │   └── ... (100s of synthetic images)
│   ├── WT_JG/                # Wild-type cell line 2
│   ├── WT_JT/                # WT cell line 3
│   ├── WT_KM/                # WT cell line 4
│   ├── WT_MS/                # WT cell line 5
│   ├── KO_1096-01/           # Knockout cell line 1
│   ├── KO_1618-01/           # KO cell line 2
│   ├── KO_BR2986/            # KO cell line 3
│   └── KO_BR3075/            # KO cell line 4
└── input_real/               # REAL IMAGES ONLY - FOR VALIDATION/TESTING
    ├── WT_1618-02/           # Same cell line names as above
    │   ├── img001.jpg        # Real images (ANY filename NOT starting with 's')
    │   ├── cell_02.png
    │   └── ... (typically 20-50 real images per cell line)
    ├── WT_JG/
    ├── WT_JT/
    ├── WT_KM/
    ├── WT_MS/
    ├── KO_1096-01/
    ├── KO_1618-01/
    ├── KO_BR2986/
    └── KO_BR3075/

MODE 2: "train_data_source": "mixed" (Backward compatibility)
------------------------------------------------
dataset_gen/
└── input/                    # MIXED synthetic + real images
    ├── WT_1618-02/
    │   ├── s001.jpg          # Synthetic
    │   ├── s002.jpg          # Synthetic
    │   ├── real_001.jpg      # Real
    │   └── ... (both types mixed)
    ├── WT_JG/
    └── ... (same structure for all cell lines)

================================================================================
EXACT VALIDATION/TESTING STRATEGIES:

STRATEGY A: VALIDATE ON TRAINING SET SPLIT
------------------------------------------
settings.py:
    "ds_val_from_train_split": 0.2,    # 20% of training images
    "ds_val_from_test_split": False,   # No validation from test set

HOW IT WORKS for Fold 1 (test_wt="WT_1618-02", test_ko="KO_1096-01"):
1. DATASET GENERATION:
   - Training cell lines: WT_JG, WT_JT, WT_KM, WT_MS, KO_1618-01, KO_BR2986, KO_BR3075
   - Test cell lines: WT_1618-02, KO_1096-01

2. DATA ALLOCATION:
   Training folder (data/train/):
     - WT/ (images from WT_JG, WT_JT, WT_KM, WT_MS) - 100% synthetic
     - KO/ (images from KO_1618-01, KO_BR2986, KO_BR3075) - 100% synthetic
   
   Test folder (data/test/):
     - WT/ (images from WT_1618-02) - 100% REAL
     - KO/ (images from KO_1096-01) - 100% REAL

3. VALIDATION SPLIT (20% from training):
   - Training: 80% of synthetic images from training cell lines
   - Validation: 20% of synthetic images from training cell lines
   - Test: 100% of REAL images from test cell lines

4. USAGE SCENARIO:
   - You have LOTS of synthetic data, FEW real images
   - Want to validate DURING training on synthetic data
   - Final test on completely unseen REAL data

STRATEGY B: VALIDATE ON TEST SET SPLIT (RECOMMENDED FOR YOU)
------------------------------------------------------------
settings.py:
    "ds_val_from_train_split": False,   # No validation from training
    "ds_val_from_test_split": 1.0,      # 100% of test set for validation

HOW IT WORKS for Fold 1 (test_wt="WT_1618-02", test_ko="KO_1096-01"):
1. DATASET GENERATION (Same as above):
   - Training cell lines: WT_JG, WT_JT, WT_KM, WT_MS, KO_1618-01, KO_BR2986, KO_BR3075
   - Test cell lines: WT_1618-02, KO_1096-01

2. DATA ALLOCATION (Same as above):
   Training folder (data/train/):
     - WT/ (images from WT_JG, WT_JT, WT_KM, WT_MS) - 100% synthetic
     - KO/ (images from KO_1618-01, KO_BR2986, KO_BR3075) - 100% synthetic
   
   Test folder (data/test/):
     - WT/ (images from WT_1618-02) - 100% REAL
     - KO/ (images from KO_1096-01) - 100% REAL

3. VALIDATION SPLIT (100% from test = NO separate testing):
   - Training: 100% of synthetic images from training cell lines
   - Validation: 100% of REAL images from test cell lines
   - Test: NONE (validation and testing are the SAME images)

4. CRITICAL POINT - What gets recorded in split_info.json:
   The system will record that ALL real images from test cell lines were used
   for validation. Since ds_val_from_test_split = 1.0, there's NO separate test set.

5. USAGE SCENARIO (YOUR CASE):
   - Train on synthetic images only
   - Validate at end of each epoch on REAL images from left-out cell lines
   - No separate "final test" - validation IS your performance metric
   - This mimics real-world: model sees only synthetic during training,
     performance measured on real biological data

STRATEGY C: SPLIT TEST SET FOR VALIDATION + TESTING
---------------------------------------------------
settings.py:
    "ds_val_from_train_split": False,   # No validation from training
    "ds_val_from_test_split": 0.7,      # 70% of test set for validation, 30% for testing

HOW IT WORKS:
1. Same dataset generation as above
2. Test folder images (100% REAL) are split:
   - Validation: 70% of real images from test cell lines
   - Test: 30% of real images from test cell lines
3. split_info.json records which specific images went to validation vs test

================================================================================
EXACT CROSS-VALIDATION FOLDS (20 total):

FOLD 1:
  Training: WT_JG, WT_JT, WT_KM, WT_MS + KO_1618-01, KO_BR2986, KO_BR3075
  Test: WT_1618-02 + KO_1096-01

FOLD 2:
  Training: WT_1618-02, WT_JT, WT_KM, WT_MS + KO_1618-01, KO_BR2986, KO_BR3075
  Test: WT_JG + KO_1096-01

... (continues through all combinations) ...

FOLD 20:
  Training: WT_1618-02, WT_JG, WT_JT, WT_KM + KO_1096-01, KO_1618-01, KO_BR2986
  Test: WT_MS + KO_BR3075

================================================================================
EXACT WORKFLOW FOR YOUR USE CASE:

settings.py:
    "train_data_source": "synthetic_only",
    "ds_val_from_train_split": False,
    "ds_val_from_test_split": 1.0,
    "wt_lines": ["WT_1618-02", "WT_JG", "WT_JT", "WT_KM", "WT_MS"],
    "ko_lines": ["KO_1096-01", "KO_1618-01", "KO_BR2986", "KO_BR3075"]

1. FOLD PREPARATION (e.g., Fold 1):
   - Copies synthetic images from WT_JG, WT_JT, WT_KM, WT_MS to data/train/WT/
   - Copies synthetic images from KO_1618-01, KO_BR2986, KO_BR3075 to data/train/KO/
   - Copies REAL images from WT_1618-02 to data/test/WT/
   - Copies REAL images from KO_1096-01 to data/test/KO/

2. TRAINING:
   - Model trains ONLY on synthetic data (data/train/)
   - At end of each epoch: validates on REAL data (data/test/)
   - Since ds_val_from_test_split = 1.0, ALL real test images are used for validation

3. RECORDING:
   - split_info.json created in acv_results/dataset_1/
   - Lists ALL real images used for validation (since no separate test)

4. FINAL EVALUATION:
   - Loads model checkpoint
   - Runs prediction on SAME validation set (data/test/ images)
   - Calculates accuracy on REAL images only

================================================================================
OUTPUT STRUCTURE:

acv_results/
├── dataset_1/                    # Results for Fold 1
│   ├── checkpoints/              # Model weights saved each epoch
│   │   ├── epoch_01.pt
│   │   ├── epoch_02.pt
│   │   └── ...
│   ├── plots/                    # Visual results
│   │   ├── confusion_matrix_epoch_01.png
│   │   ├── loss_curve.png
│   │   └── ...
│   ├── split_info.json           # EXACT images used for validation
│   │   {
│   │     "validation": {
│   │       "WT": ["img001.jpg", "img002.jpg", ...],  # REAL images only
│   │       "KO": ["cell_01.png", "cell_02.png", ...] # REAL images only
│   │     },
│   │     "test": { ... },        # Empty if ds_val_from_test_split = 1.0
│   │     "metadata": {
│   │       "note": "Only real images are recorded for validation/testing"
│   │     }
│   │   }
│   └── dataset_1_info.txt        # Configuration for this fold
├── dataset_2/                    # Results for Fold 2
└── ... (20 total folders)

================================================================================
USAGE:

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
acv = AutoCrossValidation(device)

# Run complete cross-validation (20 folds)
acv()

# After completion, analyze results in acv_results/ folder
# Each fold has independent results

================================================================================
KEY SETTINGS FOR DIFFERENT SCENARIOS:

SCENARIO 1: Train on synthetic, validate on real (YOUR CASE)
    "train_data_source": "synthetic_only",
    "ds_val_from_train_split": False,
    "ds_val_from_test_split": 1.0,

SCENARIO 2: Train on mixed data, validate on training split
    "train_data_source": "mixed",
    "ds_val_from_train_split": 0.2,
    "ds_val_from_test_split": False,

SCENARIO 3: Train on real only, validate on test split
    "train_data_source": "real_only",
    "ds_val_from_train_split": False,
    "ds_val_from_test_split": 0.7,

================================================================================
IMPORTANT NOTES:

1. SYNTHETIC IMAGE NAMING: Must start with 's' followed by numbers (s001.jpg, s123.png)
2. REAL IMAGE NAMING: Can be anything EXCEPT starting with 's' followed by numbers
3. IMAGE FORMATS: Can mix .jpg, .png, .tif, etc.
4. FOLDER NAMES: Must exactly match wt_lines and ko_lines in settings.py
5. MEMORY: Each fold creates fresh model instance (prevents weight contamination)
6. CLEANUP: data/ folder is cleaned between folds (temporary storage only)
"""

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
            print(f"Training data source: {self.ds_gen.training_data_source}")

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

            # Load split info if it exists
            split_file = Path(self.acv_results_dir) / f"dataset_{config['dataset_idx']}" / "split_info.json"
            real_test_images = None

            if split_file.exists():
                with open(split_file, 'r') as f:
                    split_info = json.load(f)
                
                # ALWAYS use split_info when we have validation split from test
                # (unless val_from_test_split = 1.0, which means all test images are validation)
                if self.val_from_test_split is not False and self.val_from_test_split != 1.0:
                    # We have a separate test set (not all test images used for validation)
                    real_test_images = set(split_info['test']['WT'] + split_info['test']['KO'])
                    print(f"Found {len(real_test_images)} test images from split_info.json (70% of real images)")
                else:
                    # Either val_from_test_split = 1.0 (no separate test) or not using test for validation
                    if self.val_from_test_split == 1.0:
                        print("Using all test images for validation (ds_val_from_test_split = 1.0)")
                    else:
                        print("Not using test set for validation, loading all test images")
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

            print('\n> Starting prediction on REAL test images using FINAL trained weights...')

            # Use the CURRENT model weights (already trained) - NO checkpoint loading needed!
            _, cm = self.cnn_wrapper.predict(test_dataset_to_use)

            # Plot confusion matrix and results
            chckpt_name = f"final_trained_model_fold_{config['dataset_idx']}"
            fn.plot_confusion_matrix(cm, self.class_list, plot_dir, chckpt_name=chckpt_name, show_plot=False, save_plot=True)
            fn.save_confusion_matrix_results(cm, self.class_list, plot_dir, chckpt_name=chckpt_name)

            # Load confusion matrix results
            loaded_results = fn.load_confusion_matrix_results(plot_dir, file_name=chckpt_name)
            print(f"=== REAL IMAGE TEST RESULTS ===")
            print(f"Overall accuracy: {(loaded_results['overall_accuracy']*100):.2f}%")
            print(f"WT accuracy: {(loaded_results['class_accuracy']['WT']*100):.2f}%")
            print(f"KO accuracy: {(loaded_results['class_accuracy']['KO']*100):.2f}%")

            print(f'Prediction successfully finished. Confusion matrix and results saved to {plot_dir}.')

            print(f"Completed dataset {config['dataset_idx']}. Moving to next fold...")    

        # Final cleanup (once at the end)
        print("\nCleaning up all temporary data...")
        self.ds_gen.cleanup(self.data_dir)
        print("Cross-validation complete.")