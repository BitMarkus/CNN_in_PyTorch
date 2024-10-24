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
    "train_num_epochs": 60,  
    # Initial learning rate (later determined by lr scheduler)
    "train_init_lr": 0.01,    # ADAM: 0.0001. SGD: 0.001
    # Learning rate scheduler:
    # No of steps after which he lr is multiplied by the lr multiplier
    "train_lr_step_size": 10, 
    "train_lr_multiplier": 0.5,   

    ###########
    # DATASET #
    ###########

    # Shuffle dataset
    "ds_shuffle": True,
    # Shuffle seed
    "ds_shuffle_seed": 123,
    # Batch size for training and validation datasets
    "ds_batch_size": 1, 
    # Fraction of images which go into the validation dataset 
    "ds_val_split": 0.1, 

    #########
    # MODEL #
    #########

    # Name of the model architecture:
    # ResNet: ResNet-18, -34, -50, -101, -152
    # AlexNet
    # VGG: VGG-11, -13, -16, -19 (models with batch normalization)
    # DenseNet-121, -161, -169, -201
    # "cnn_type": "ResNet-152",  
    # Custom CNN architecture
    "cnn_type": "custom", 

    ################
    # CUSTOM MODEL #
    ################

    # Dropout
    "cnn_dropout": 0.3,

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
    "chckpt_weights_file": "2024-09-19-14-26_checkpoint_ResNet-152_e23_vacc82.model",  
    # Set to True, if checkpoints shall be saved during training
    "chckpt_save": True,  
    # Mininmun validation accuracy from which on checkpoints are saved
    "chckpt_min_acc": 0.7,  

    #########
    # PATHS #
    #########

    "pth_data": "data/",
    "pth_checkpoint": "checkpoints/",
    "pth_plots": "plots/",
    "pth_prediction": "prediction/",
 
}