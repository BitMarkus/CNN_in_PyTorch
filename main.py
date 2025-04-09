import torch
from PIL import Image
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
import os
# Own modules
import functions as fn
from model import CNN_Model
from custom_model import Custom_CNN_Model
from dataset import Dataset
from train import Train
from settings import setting

###########
# OBJECTS #
###########

# Create a dataset object
ds = Dataset()
# Create a model object
cnn = Custom_CNN_Model()

########
# MAIN #
########

def main():

    # Show system information and select device (cpu or gpu)
    device = fn.show_cuda_and_versions()
    # Create program folders if they don't exist already
    fn.create_prg_folders()

    #############
    # Main Menu #
    #############

    while(True):  
        print("\n:MAIN MENU:")
        print("1) Create CNN Network")
        print("2) Show Network Summary")
        print("3) Load Training Data")
        print("4) Train Network")
        print("5) Load Weights")
        print("6) Predict Images in Prediction Folder")
        print("7) Captum Test")
        print("8) Exit Program")
        menu1 = int(fn.input_int("Please choose: "))

        ######################
        # Create CNN Network # 
        ###################### 

        if(menu1 == 1):       
            print("\n:NEW CNN NETWORK:")  
            # Check if a model was already loaded
            if(cnn.model_loaded):
                print("A network was already loaded!")
            else:
                if(cnn):
                    # Print class information
                    cnn.print_class_list()
                    # Load model
                    print(f"Creating new {cnn.cnn_type} network...")
                    if(setting["cnn_type"] == "custom"):
                        cnn.model = Custom_CNN_Model().to(device)
                        cnn.model_loaded = True  
                    else:
                        cnn.model = cnn.load_model(device)  
                    print("New network was successfully created.")   
                    cnn.print_model_size()  
                else:
                    print("Unable to load the requested cnn architecture!")          

        ########################
        # Show Network Summary #  
        ########################

        elif(menu1 == 2):        
            print("\n:SHOW NETWORK SUMMARY:")   
            if(cnn.model_loaded):
                cnn.model_summary(device) # print(cnn.model)   
            else:
                print("No network was generated yet!") 

        ######################
        # Load Training Data # 
        ######################
        
        elif(menu1 == 3):       
            print("\n:LOAD TRAINING DATA:") 
            ds.load_training_dataset()
            if(ds.ds_loaded):
                print("Training and validation datasets successfully loaded.")
                print(f"Number training images/batches: {ds.num_train_img}/{ds.num_train_batches}")
                print(f"Number validation images/batches: {ds.num_val_img}/{ds.num_val_batches}") 

        #################
        # Train Network #  
        #################
                
        elif(menu1 == 4):
            print("\n:TRAIN NETWORK:") 
            if not (cnn.model_loaded):
                print('No CNN generated yet!')
            elif not (ds.ds_loaded):
                print('No training data loaded yet!')
            else:
                print("Start training...")
                # Create a training object
                train = Train(cnn, ds)
                # Train network
                history = train.train()
                print("\nTraining finished!")
                # Show/save training plots
                train.plot_metrics(history, setting["pth_plots"], show_plot=True, save_plot=True)

        ################
        # Load Weights #
        ################

        elif(menu1 == 5):
            # Load checkpoint weights
            print("\n:LOAD WEIGHTS:") 
            if not (cnn.model_loaded):
                print('No CNN generated yet!')
            else:
                # Load model and weights for inference
                cnn.load_weights(device)

        ############################
        # Predict images in folder #  
        ############################

        elif(menu1 == 6):  
            print("\n:PREDICT IMAGES IN FOLDER:") 
            if not (cnn.model_loaded):
                print('No CNN generated yet!')
            else:
                print('Load prediction dataset...')
                ds.load_prediction_dataset()
                print('Prediction dataset successfully loaded.')
                print(f"Number test images/batches: {ds.num_pred_img}/1")
                print('Starting prediction...')
                pred_acc, cm = cnn.predict(ds.ds_pred)
                print(f"Accuracy: {pred_acc:.2f}")  

                # Plot confusion matrix
                class_list = cnn.get_class_list()
                fn.plot_confusion_matrix(cm, class_list, setting["pth_plots"], show_plot=True, save_plot=True)

        ###############
        # Captum Test #  
        ###############

        elif(menu1 == 7):  
            print("\n:CAPTUM:") 

            # ds.load_prediction_dataset()
            # cnn.model = Custom_CNN_Model().to(device)
            ds.load_training_dataset()
            cnn.load_weights(device)

            cnn.predict_single(ds.ds_train)

        ################
        # Exit Program #
        ################

        elif(menu1 == 8):
            print("\nExit program...")
            break
        
        # Wrong Input
        else:
            print("Not a valid option!")    


if __name__ == "__main__":
    main()
