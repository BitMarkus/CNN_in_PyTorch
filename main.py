# ===== Own Modules =====
import functions as fn
from model import CNN_Model
from dataset_gen import DatasetGenerator
from auto_cross_validation import AutoCrossValidation
from conf_analyzer import ConfidenceAnalyzer
from dataset import Dataset
from train import Train
from class_analyzer import ClassAnalyzer
from gradcam_analyzer import GradCAMAnalyzer
from class_sorter import ClassSorter
from dim_red import DimRed
from fid_calculator import FIDCalculator
from settings import setting

###########
# OBJECTS #
###########

# Create a dataset object
ds = Dataset()
# Create wrapper (contains all metadata)
cnn_wrapper = CNN_Model()

########
# MAIN #
########

# Main entry point for the program.
# Displays a menu and routes user input to the appropriate functionality.
# Handles initialization of objects, device selection, and folder creation.
def main() -> None:
    # Show system information and select device (cpu or gpu)
    device = fn.show_cuda_and_versions()
    # Create program folders if they don't exist already
    fn.create_prg_folders()

    #############
    # Main Menu #
    #############

    while True:
        print("\n:MAIN MENU:")
        print("1) Create CNN Network")
        print("2) Show Network Summary")
        print("3) Load Training Data")
        print("4) Train Network")
        print("5) Load Weights")
        print("6) Predict Class from Input Folder")
        print("7) Dataset Generator (ACV)")
        print("8) Automatic Cross Validation (ACV)")
        print("9) Confidence Analyzer (based on ACV)")
        print("10) GradCAM Analyzer")
        print("11) Class Sorter")
        print("12) Dimension Reduction")
        print("13) FID Calculator")
        print("14) Exit Program")
        menu1 = int(fn.input_int("Please choose: "))

        ######################
        # Create CNN Network #
        ######################

        if menu1 == 1:
            print("\n:NEW CNN NETWORK:")
            # Check if a model was already loaded
            if cnn_wrapper.model_loaded:
                print("A network was already loaded!")
            else:
                if cnn_wrapper:
                    # Print class information
                    cnn_wrapper.print_class_list()
                    # Load model
                    print(f"Creating new {cnn_wrapper.cnn_type} network...")
                    # Get actual model (nn.Module)
                    cnn = cnn_wrapper.load_model(device).to(device)
                    print("New network was successfully created.")
                    cnn_wrapper.print_model_size()
                else:
                    print("Unable to load the requested cnn architecture!")

        ########################
        # Show Network Summary #
        ########################

        elif menu1 == 2:
            print("\n:SHOW NETWORK SUMMARY:")
            if cnn_wrapper.model_loaded:
                cnn_wrapper.model_summary(device)
            else:
                print("No network was generated yet!")

        ######################
        # Load Training Data #
        ######################

        elif menu1 == 3:
            print("\n:LOAD TRAINING DATA:")
            # Check for correct settings in settings file
            ds.validate_validation_settings()
            # Load training dataset and validation dataset if validation set
            # comes from training images
            ds.load_training_dataset()
            # Load validation dataset if validation set comes from test images
            if ds.validation_from_test:
                ds.load_test_dataset()
            # Print dataset info
            ds.print_dataset_info()

            # OPTIONAL: Export validation images
            if setting['ds_save_val_images']:
                print("Exporting validation images. Please wait....")
                val_img_export_pth = setting['pth_ds_gen_output'] / "validation_images"
                # Check if folder exists, if not create it
                if not val_img_export_pth.exists():
                    val_img_export_pth.mkdir(parents=True, exist_ok=True)
                ds.export_validation_images(val_img_export_pth)

            if ds.ds_loaded:
                print("Training and validation datasets successfully loaded.")
                print(f"Number training images/batches: {ds.num_train_img}/{ds.num_train_batches}")
                print(f"Number validation images/batches: {ds.num_val_img}/{ds.num_val_batches}")

        #################
        # Train Network #
        #################

        elif menu1 == 4:
            print("\n:TRAIN NETWORK:")
            if not cnn_wrapper.model_loaded:
                print('No CNN generated yet!')
            elif not ds.ds_loaded:
                print('No training data loaded yet!')
            else:
                print("Start training...")
                # Create a training object
                train = Train(cnn_wrapper, ds, device)
                # Train network
                train.train()
                print("\nTraining finished!")

        ################
        # Load Weights #
        ################

        elif menu1 == 5:
            # Load checkpoint weights
            print("\n:LOAD WEIGHTS:")
            if not cnn_wrapper.model_loaded:
                print('No CNN generated yet!')
            else:
                cnn_wrapper.load_checkpoint()

        ############################
        # Predict images in folder #
        ############################

        elif menu1 == 6:
            print("\n:PREDICT CLASS FROM INPUT FOLDER:")
            print("  Place images to classify in the input/ folder")
            print("  Results will be saved to output/")

            analyzer = ClassAnalyzer(device)
            analyzer.analyze_prediction_folder()

        #####################
        # Dataset Generator #
        #####################

        elif menu1 == 7:
            print("\n:DATASET GENERATOR FOR CROSS VALIDATION:")
            # Quick verification
            synthetic_dir = setting["pth_ds_gen_input_synthetic"]
            real_dir = setting["pth_ds_gen_input_real"]
            print(f"Training data source: {setting['train_data_source']}")
            print(f"Synthetic folder: {synthetic_dir}")
            print(f"Real folder: {real_dir}")
            if not synthetic_dir.exists():
                print(f"ERROR: Synthetic folder not found!")
                continue
            if not real_dir.exists():
                print(f"ERROR: Real folder not found!")
                continue
            # Create a dataset generation object (in generation mode)
            ds_gen = DatasetGenerator(mode="gen")
            print('\nGeneration of datasets is starting...')
            print(f"Mode: {ds_gen.training_data_source}")
            # Create datasets for cross validation
            ds_gen.generate_all_datasets()
            print(f"\nDatasets successfully created and saved to {setting['pth_ds_gen_output']}!")
            print("You can inspect the generated datasets in that folder.")

        ##############################
        # Automatic Cross Validation #
        ##############################

        elif menu1 == 8:
            print("\n:AUTOMATIC CROSS VALIDATION:")
            print("  Results will be saved to output/cross_validation/")
            acv = AutoCrossValidation(device)
            acv()

        #######################
        # Confidence Analyzer #
        #######################

        elif menu1 == 9:
            print("\n:CONFIDENCE ANALYZER:")
            print("  Input: output/cross_validation/")
            print("  Output: output/conf_analyzer/")
            confa = ConfidenceAnalyzer(device)
            confa()

        ####################
        # GradCAM Analyzer #
        ####################

        elif menu1 == 10:
            print("\n:GradCAM ANALYZER:")
            print("  Input: input/ (place images to analyze)")
            print("  Output: output/gradcam/")
            gradcam = GradCAMAnalyzer(device)
            gradcam()

        ################
        # Class Sorter #
        ################

        elif menu1 == 11:
            print("\n:CLASS SORTER:")
            print("  Input: input/ (place images to sort)")
            print("  Output: output/class_sorter/")
            # Create and run sorter (all configuration is loaded from settings.py)
            sorter = ClassSorter(device)
            output_dir = sorter.run()

        #######################
        # Dimension Reduction #
        #######################

        elif menu1 == 12:
            print("\n:DIMENSION REDUCTION:")
            print("  Input: input/ (place images in folders or directly)")
            print("  Output: output/dim_red/")
            # DimRed simple version
            dimred = DimRed(device)
            dimred()

        ##################
        # FID Calculator #
        ##################

        elif menu1 == 13:
            print("\n:FID CALCULATOR:")
            print("  Input: input/ (place folders with images to compare)")
            print("  Output: output/fid/")
            fid_score_calc = FIDCalculator(device)
            fid_score_calc()

        ################
        # Exit Program #
        ################

        elif menu1 == 14:
            print("\nExit program...")
            break

        # Wrong Input
        else:
            print("Not a valid option!")


if __name__ == "__main__":
    main()