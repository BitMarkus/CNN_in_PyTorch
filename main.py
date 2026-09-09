# ===== Own Modules =====
import functions as fn
from single_training import CNN_Model, Dataset
from single_training_menu import SingleTrainingMenu
from cross_validation_menu import CrossValidationMenu
from analysis_menu import AnalysisMenu
from utilities_menu import Utilities
from preprocessing_menu import PreprocessingMenu
from export_plotting_menu import ExportPlottingMenu

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
        print("1) Single CNN Training ↓")
        print("2) Cross Validation ↓")
        print("3) Analysis ↓")
        print("4) Utilities ↓")
        print("5) Preprocessing ↓")
        print("6) Export & Plotting ↓")
        print("7) Exit Program")
        menu1 = int(fn.input_int("Please choose: "))

        #########################
        # Single CNN Training   #
        #########################

        if menu1 == 1:
            st_menu = SingleTrainingMenu()
            st_menu.menu(device, cnn_wrapper, ds)

        #########################
        # Cross Validation      #
        #########################

        elif menu1 == 2:
            cv_menu = CrossValidationMenu()
            cv_menu.menu(device)

        #########################
        # Analysis              #
        #########################

        elif menu1 == 3:
            analysis_menu = AnalysisMenu()
            analysis_menu.menu(device)

        #########################
        # Utilities             #
        #########################

        elif menu1 == 4:
            utilities = Utilities()
            utilities.menu()

        #########################
        # Preprocessing         #
        #########################

        elif menu1 == 5:
            preproc_menu = PreprocessingMenu()
            preproc_menu.menu()

        #########################
        # Export & Plotting     #
        #########################

        elif menu1 == 6:
            export_menu = ExportPlottingMenu()
            export_menu.menu()

        #########################
        # Exit Program          #
        #########################

        elif menu1 == 7:
            print("\nExit program...")
            break

        # Wrong Input
        else:
            print("Not a valid option!")


if __name__ == "__main__":
    main()