####################
# Program settings #
####################

setting = {

    ############
    # TRAINING #
    ############

    # Number of epochs
    "train_num_epochs": 30,  
    # Batch size for training and validation datasets
    # EfficientNet-B0: max. 26
    "ds_batch_size": 64, 

    # Optimizer:
    # Initial learning rate (later determined by lr scheduler)
     # ADAM: 0.0001-0.0003 (3e-4), SGD: 0.01-0.001
    "train_init_lr": 0.001,    
    # Weight decay = L2 regularization
    # ADAM and SGD: 1e-4 
    "train_weight_decay": 1e-4,  
    # Momentum
    # SGD: 0.9 
    "train_momentum": 0.9,       

    # Learning rate scheduler:
    # No of steps after which he lr is multiplied by the lr multiplier
    # Linear StepLR:
    "train_lr_step_size": 10, 
    "train_lr_multiplier": 0.5, 
    # CosineAnnealingLR 
    "train_lr_eta_min": 1e-5,

    ###########
    # DATASET #
    ###########

    # Shuffle dataset
    "ds_shuffle": True,
    # Shuffle seed
    "ds_shuffle_seed": 123,
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
    "cnn_type": "EfficientNet-B0",  
    # Custom CNN architecture: custom
    # "cnn_type": "custom", 

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
    "chckpt_weights_file": ".model",  
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