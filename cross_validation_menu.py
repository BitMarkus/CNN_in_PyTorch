# ===== Own Modules =====
import functions as fn
from settings import setting
from cross_validation import DatasetGenerator, AutoCrossValidation, ConfidenceAnalyzer

class CrossValidationMenu:

    # Cross-Validation Menu - Collection of tools for cross-validation.

    #############################################################################################################
    # CONSTRUCTOR

    def __init__(self) -> None:
        pass

    #############################################################################################################
    # METHODS

    # Display the cross-validation submenu and route to the selected option.
    # Args:
    #   device (torch.device): Device to run operations on
    def menu(self, device) -> None:
        while True:
            print("\n:CROSS VALIDATION MENU:")
            print("1) Dataset Generator (Prepare datasets)")
            print("2) Automatic Cross Validation (Run training)")
            print("3) Confidence Analyzer (Analyze results)")
            print("4) Back to Main Menu")

            choice = fn.input_int("Please choose: ")

            if choice == 1:
                self._run_dataset_generator()
            elif choice == 2:
                self._run_auto_cross_validation(device)
            elif choice == 3:
                self._run_confidence_analyzer(device)
            elif choice == 4:
                print("\nReturning to main menu...")
                break
            else:
                print("Not a valid option!")

    # Run the Dataset Generator.
    def _run_dataset_generator(self) -> None:
        print("\n:DATASET GENERATOR FOR CROSS VALIDATION:")
        # Quick verification
        synthetic_dir = setting["pth_ds_gen_input_synthetic"]
        real_dir = setting["pth_ds_gen_input_real"]
        print(f"Training data source: {setting['train_data_source']}")
        print(f"Synthetic folder: {synthetic_dir}")
        print(f"Real folder: {real_dir}")
        if not synthetic_dir.exists():
            print(f"ERROR: Synthetic folder not found!")
            return
        if not real_dir.exists():
            print(f"ERROR: Real folder not found!")
            return
        # Create a dataset generation object (in generation mode)
        ds_gen = DatasetGenerator(mode="gen")
        print('\nGeneration of datasets is starting...')
        print(f"Mode: {ds_gen.training_data_source}")
        # Create datasets for cross validation
        ds_gen.generate_all_datasets()
        print(f"\nDatasets successfully created and saved to {setting['pth_ds_gen_output']}!")
        print("You can inspect the generated datasets in that folder.")

    # Run the Automatic Cross Validation.
    # Args:
    #   device (torch.device): Device to run training on
    def _run_auto_cross_validation(self, device) -> None:
        print("\n:AUTOMATIC CROSS VALIDATION:")
        print("  Results will be saved to output/cross_validation/")
        acv = AutoCrossValidation(device)
        acv()

    # Run the Confidence Analyzer.
    # Args:
    #   device (torch.device): Device to run analysis on
    def _run_confidence_analyzer(self, device) -> None:
        print("\n:CONFIDENCE ANALYZER:")
        print("  Input: output/cross_validation/")
        print("  Output: output/conf_analyzer/\n")
        confa = ConfidenceAnalyzer(device)
        confa()