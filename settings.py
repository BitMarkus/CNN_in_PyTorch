####################
# Program settings #
####################

setting = {

    ############
    # TRAINING #
    ############
 
    # Weight decay
    "train_weight_decay": 0.0,  
    # Number of epochs
    "train_num_epochs": 3,  
    # Initial learning rate (later determined by lr scheduler)
    "train_init_lr": 0.0001, 
    # Learning rate scheduler:
    # No of steps after which he lr is multiplied by the lr multiplier
    "train_lr_step_size": 10, 
    "train_lr_multiplier": 0.4,   

    ###########
    # DATASET #
    ###########

    # Shuffle dataset
    "ds_shuffle": True,
    # Shuffle seed
    "ds_shuffle_seed": 123,
    # Batch size for training and validation datasets
    "ds_batch_size": 24, 
    # Batch size for prediction dataset
    "ds_batch_size_pred": 1, 
    # Fraction of images which go into the validation dataset 
    "ds_val_split": 0.1, 

    #########
    # MODEL #
    #########

    # Name of the model architecture
    "model_name": "ResNet-50",     

    ##########
    # IMAGES #
    ##########

    # Image dimensions
    "img_width": 512,   
    "img_height": 512,
    "img_channels": 1,          

    ###############
    # CHECKPOINTS #
    ###############

    # Name of the checkpoint file to load weights for predictions
    "chckpt_weights": "2024-07-30-16-54_checkpoint_e28_vacc84.model",  
    # Set to True, if checkpoints shall be saved during training
    "chckpt_save": True,  
    # Mininmun validation accuracy from which on checkpoints are saved
    "chckpt_min_acc": 0.8,  

    #########
    # PATHS #
    #########

    "pth_data": "data/",
    "pth_checkpoint": "checkpoints/",
    "pth_plots": "plots/",
    "pth_prediction": "prediction/",
 
}