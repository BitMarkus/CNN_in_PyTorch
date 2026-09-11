# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Own Modules =====
import functions as fn
from settings import setting
from . import Train

class SingleTrainingMenu:

    # Single CNN Training Menu - Tools for creating, training, and managing a single CNN model.

    #############################################################################################################
    # CONSTRUCTOR

    def __init__(self) -> None:
        pass

    #############################################################################################################
    # METHODS

    # Display the Single CNN Training submenu and route to the selected option.
    # Args:
    #   device (torch.device): Device to run operations on
    #   cnn_wrapper (CNN_Model): Wrapper containing the model
    #   ds (Dataset): Dataset handler
    def menu(self, device, cnn_wrapper, ds) -> None:
        while True:
            print("\n:SINGLE CNN TRAINING:")
            print("1) Create CNN Network")
            print("2) Show Network Summary")
            print("3) Load Training Data")
            print("4) Train Network")
            print("5) Load Weights")
            print("6) Back to Main Menu")

            choice = fn.input_int("Please choose: ")

            if choice == 1:
                self._create_cnn_network(cnn_wrapper, device)
            elif choice == 2:
                self._show_network_summary(cnn_wrapper, device)
            elif choice == 3:
                self._load_training_data(ds)
            elif choice == 4:
                self._train_network(cnn_wrapper, ds, device)
            elif choice == 5:
                self._load_weights(cnn_wrapper)
            elif choice == 6:
                print("\nReturning to main menu...")
                break
            else:
                print("Not a valid option!")

    # Create CNN Network.
    def _create_cnn_network(self, cnn_wrapper, device) -> None:
        print("\n:NEW CNN NETWORK:")
        if cnn_wrapper.model_loaded:
            print("A network was already loaded!")
        else:
            if cnn_wrapper:
                cnn_wrapper.print_class_list()
                print(f"Creating new {cnn_wrapper.cnn_type} network...")
                cnn = cnn_wrapper.load_model(device).to(device)
                print("New network was successfully created.")
                cnn_wrapper.print_model_size()
            else:
                print("Unable to load the requested cnn architecture!")

    # Show Network Summary.
    def _show_network_summary(self, cnn_wrapper, device) -> None:
        print("\n:SHOW NETWORK SUMMARY:")
        if cnn_wrapper.model_loaded:
            cnn_wrapper.model_summary(device)
        else:
            print("No network was generated yet!")

    # Load Training Data.
    def _load_training_data(self, ds) -> None:
        print("\n:LOAD TRAINING DATA:")
        ds.validate_validation_settings()
        ds.load_training_dataset()
        if ds.validation_from_test:
            ds.load_test_dataset()
        ds.print_dataset_info()

        if setting.get('ds_save_val_images', False):
            print("Exporting validation images. Please wait....")
            val_img_export_pth = setting['pth_ds_gen_output'] / "validation_images"
            if not val_img_export_pth.exists():
                val_img_export_pth.mkdir(parents=True, exist_ok=True)
            ds.export_validation_images(val_img_export_pth)

        if ds.ds_loaded:
            print("Training and validation datasets successfully loaded.")
            print(f"Number training images/batches: {ds.num_train_img}/{ds.num_train_batches}")
            print(f"Number validation images/batches: {ds.num_val_img}/{ds.num_val_batches}")

    # Train Network.
    def _train_network(self, cnn_wrapper, ds, device) -> None:
        print("\n:TRAIN NETWORK:")
        print("  Results will be saved to output/train/[timestamp]/")
        if not cnn_wrapper.model_loaded:
            print('No CNN generated yet!')
        elif not ds.ds_loaded:
            print('No training data loaded yet!')
        else:
            print("\nStart training...\n")
            train = Train(cnn_wrapper, ds, device)
            train.train()
            print("\nTraining finished!")

    # Load Weights.
    def _load_weights(self, cnn_wrapper) -> None:
        print("\n:LOAD WEIGHTS:")
        if not cnn_wrapper.model_loaded:
            print('No CNN generated yet!')
        else:
            cnn_wrapper.load_checkpoint()