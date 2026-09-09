# ===== Own Modules =====
import functions as fn
from model import CNN_Model
from dataset import Dataset
from train import Train
from class_analyzer import ClassAnalyzer
from gradcam_analyzer import GradCAMAnalyzer
from utilities_menu import Utilities
from cross_validation_menu import CrossValidationMenu
from preprocessing_menu import PreprocessingMenu
from export_plotting_menu import ExportPlottingMenu
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
        print("  ─── Training ───")
        print("1) Create CNN Network")
        print("2) Show Network Summary")
        print("3) Load Training Data")
        print("4) Train Network")
        print("5) Load Weights")
        print("6) Cross Validation ↓")
        print("  ─── Analysis ───")
        print("7) Predict Class from Input Folder")
        print("8) GradCAM Analyzer")
        print("  ─── Tools ───")
        print("9) Utilities ↓")
        print("10) Preprocessing ↓")
        print("11) Export & Plotting ↓")
        print("12) Exit Program")
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
            print("  Results will be saved to output/train/[timestamp]/")
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

        #########################
        # Cross Validation Menu #
        #########################

        elif menu1 == 6:
            cv_menu = CrossValidationMenu()
            cv_menu.menu(device)

        #############################
        # Predict from Input Folder #
        #############################

        elif menu1 == 7:
            print("\n:PREDICT CLASS FROM INPUT FOLDER:")
            print("  Place images to classify in the input/ folder")
            print("  Results will be saved to output/")

            analyzer = ClassAnalyzer(device)
            analyzer.analyze_prediction_folder()

        ####################
        # GradCAM Analyzer #
        ####################

        elif menu1 == 8:
            print("\n:GradCAM ANALYZER:")
            print("  Input: input/ (place images to analyze)")
            print("  Output: output/gradcam/")
            gradcam = GradCAMAnalyzer(device)
            gradcam()

        ##############
        # Utilities  #
        ##############

        elif menu1 == 9:
            utilities = Utilities()
            utilities.menu()

        #################
        # Preprocessing #
        #################

        elif menu1 == 10:
            preproc_menu = PreprocessingMenu()
            preproc_menu.menu()

        ######################
        # Export & Plotting  #
        ######################

        elif menu1 == 11:
            export_menu = ExportPlottingMenu()
            export_menu.menu()

        ################
        # Exit Program #
        ################

        elif menu1 == 12:
            print("\nExit program...")
            break

        # Wrong Input
        else:
            print("Not a valid option!")


if __name__ == "__main__":
    main()