import torch
# Own modules
import functions as fn
from model import CNN_Model
from dataset import Dataset
from train import Train
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
# Own modules
from settings import setting

###########
# OBJECTS #
###########

# Create a model object
cnn = CNN_Model()
# Create a dataset object
ds = Dataset()

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
        print("6) Predict Images in Predict Folder")
        print("7) Exit Program")
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
                # Print class information
                cnn.print_class_list()
                # Load model
                print(f"Creating new {cnn.model_name} network...")  
                model = cnn.load_model(device)  
                print("New network was successfully created.")            

        ########################
        # Show Network Summary #  
        ########################

        elif(menu1 == 2):        
            print("\n:SHOW NETWORK SUMMARY:")   
            if(cnn.model_loaded):
                print(model)   
            else:
                print("No network was generated yet!") 

        ######################
        # Load Training Data # 
        ######################
        
        elif(menu1 == 3):       
            print("\n:LOAD TRAINING DATA:") 
            ds.load_training_dataset()
            if(ds.ds_loaded):
                print("Training and validation datasets successfully loaded!")
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
                train = Train(model, ds)
                # Train network
                history = train.train()
                print("Training finished!")
                # Show/save training plots
                train.plot_metrics(history, setting["pth_plots"], show_plot=True, save_plot=True)

        ################
        # Load Weights #
        ################

        elif(menu1 == 5):
            # Load checkpoint weights
            print("\n:LOAD WEIGHTS:") 
            if('model' not in globals() and 'model' not in locals()):
                print('No network generated yet!')
            else:
                # Load model and weights for inference
                # https://stackoverflow.com/questions/49941426/attributeerror-collections-ordereddict-object-has-no-attribute-eval
                model.load_state_dict(torch.load(setting["pth_checkpoint"] + setting["chckpt_weights"]))
                model.to(device)
                print(f'Weights from checkpoint {setting["chckpt_weights"]} successfully loaded.')

        ############################
        # Predict images in folder #  
        ############################

        elif(menu1 == 6):  
            print("\n:PREDICT IMAGES IN FOLDER:") 
            if('model' not in globals() and 'model' not in locals()):
                print('No network generated yet!')
            else:
                print('Load prediction dataset...')
                prediction_ds = ds.load_prediction_dataset()
                print('Prediction dataset successfully loaded.')
                print('Starting prediction...')
                pred_acc, cm = fn.predict(model, prediction_ds)
                print(f"Accuracy: {pred_acc:.2f}")  

                # # Print confusion matrix 
                # https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
                # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
                ConfusionMatrixDisplay.from_predictions(
                    cm["y"], 
                    cm["y_hat"], 
                    display_labels=cnn.get_class_list(), 
                    cmap='Blues', 
                    normalize='all',
                )
                plt.show()          

        ################
        # Exit Program #
        ################

        elif(menu1 == 7):
            print("\nExit program...")
            break
        
        # Wrong Input
        else:
            print("Not a valid option!")    


if __name__ == "__main__":
    main()
