# Own modules
import functions as fn
from custom_model import Custom_CNN_Model
from dataset_gen import DatasetGenerator
from auto_cross_validation import AutoCrossValidation
from conf_analyzer import ConfidenceAnalyzer
from dataset import Dataset
from train import Train
from class_analyzer import ClassAnalyzer
from captum_analyzer import CaptumAnalyzer
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
        # print("6) Predict Images in data/test Folder")
        print("6) Predict Class from Predict Folder")
        print("7) Dataset Generator")
        print("8) Automatic Cross Validation (ACV)")
        print("9) Confidence Analyzer (based on ACV)")
        print("10) Captum Test")
        print("11) Exit Program")
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
                train.train(setting["pth_checkpoint"], setting["pth_plots"])
                print("\nTraining finished!")

        ################
        # Load Weights #
        ################

        elif(menu1 == 5):
            # Load checkpoint weights
            print("\n:LOAD WEIGHTS:") 
            if not (cnn.model_loaded):
                print('No CNN generated yet!')
            else:
                cnn.load_checkpoint()

        ############################
        # Predict images in folder #  
        ############################

        elif(menu1 == 6):  
            print("\n:PREDICT CLASS FROM PREDICTION FOLDER:") 

            analyzer = ClassAnalyzer(device)
            analyzer.analyze_prediction_folder()

            """
            if not (cnn.model_loaded):
                print('No CNN generated yet!')
            else:
                # Load prediction dataset
                print('Load prediction dataset...')
                ds.load_test_dataset()
                print('Prediction dataset successfully loaded.')
                print(f"Number test images/batch size: {ds.num_pred_img}/1")
                print('Starting prediction...')
                _, cm = cnn.predict(ds.ds_pred)
                # Get class list
                class_list = cnn.get_class_list()
                # Get name of the checkpoint file the confusion matrix is based on
                if(cnn.checkpoint_loaded):
                    ckpt = checkpoint_file
                else:
                    ckpt = None
                # Save confusion matrix results
                fn.save_confusion_matrix_results(cm, class_list, setting["pth_plots"], chckpt_name=ckpt)
                # Plot confusion matrix
                fn.plot_confusion_matrix(cm, class_list, setting["pth_plots"], chckpt_name=ckpt, show_plot=True, save_plot=True)
                # Load confusion matrix results
                loaded_results = fn.load_confusion_matrix_results(setting["pth_plots"], file_name=ckpt)
                # Access the data
                print(f"Overall accuracy: {(loaded_results['overall_accuracy']*100):.2f}%")
                print(f"WT accuracy: {(loaded_results['class_accuracy']['WT']*100):.2f}")
                print(f"KO accuracy: {(loaded_results['class_accuracy']['KO']*100):.2f}")
            """

        #####################
        # Dataset Generator #  
        #####################

        elif(menu1 == 7):  
            print("\n:DATASET GENERATOR FOR CROSS VALIDATION:")  
            # Create a dataset generation object (in generation mode)
            ds_gen = DatasetGenerator(mode = "gen")
            print('Generation of datasets is starting...')
            # Create datasets for cross validation
            ds_gen.generate_all_datasets()
            print(f"Datasets successfully created and saved to {setting['pth_ds_gen_output']}!")

        ##############################
        # Automatic Cross Validation #  
        ##############################

        elif(menu1 == 8):  
            print("\n:AUTOMATIC CROSS VALIDATION:")  
            acv = AutoCrossValidation(device)
            acv()

        #######################
        # Confidence Analyzer #  
        #######################

        elif(menu1 == 9):  
            print("\n:CONFIDENCE ANALYZER:")  
            confa = ConfidenceAnalyzer(device)
            confa()

        ###############
        # Captum Test #  
        ###############

        elif(menu1 == 10):  
            print("\n:CAPTUM:") 
            # print("Under construction...")
            capta = CaptumAnalyzer(device)
            capta()

        ################
        # Exit Program #
        ################

        elif(menu1 == 11):
            print("\nExit program...")
            break
        
        # Wrong Input
        else:
            print("Not a valid option!")    


if __name__ == "__main__":
    main()
