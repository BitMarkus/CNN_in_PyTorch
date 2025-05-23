####################
# Program settings #
####################

setting = {

    ############
    # TRAINING #
    ############

    # Number of epochs
    "train_num_epochs": 1,  # 30
    # Batch size for training and validation datasets
    "ds_batch_size": 50, 

    # Optimizer:
    # Initial learning rate (later determined by lr scheduler)
    # ADAM: 0.0001-0.0003 (3e-4), 
    # SGD: 0.01-0.001, 0.0001 for pretrained weights!
    "train_init_lr": 0.0001,    
    # Weight decay = L2 regularization
    # ADAM and SGD: 1e-4 
    "train_weight_decay": 1e-4,  
    # Momentum
    # SGD: 0.9 
    "train_momentum": 0.9,       

    # Learning rate scheduler:
    # No of steps after which he lr is multiplied by the lr multiplier
    # Warmup scheduler:
    "train_lr_warmup_epochs": 5, 
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
    # Define cell lines (for dataset generator)
    "wt_lines": ["WT_1618-02", "WT_JG", "WT_JT", "WT_KM", "WT_MS"],
    "ko_lines": ["KO_1096-01", "KO_1618-01", "KO_BR2986", "KO_BR3075"],
    # "wt_lines": ["WT_JG", "WT_JT"],
    # "ko_lines": ["KO_BR2986", "KO_BR3075"],
    "classes": ["KO", "WT"],
    # "classes": ["KO_1096-01", "KO_1618-01", "KO_BR2986", "KO_BR3075", "WT_1618-02", "WT_JG", "WT_JT", "WT_KM", "WT_MS"],

    #########
    # MODEL #
    #########

    # Name of the model architecture:
    # ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
    # ResNeXt variants: resnext101_32x8d, resnext101_64x4d
    # AlexNet: alexnet
    # VGG (without batch norm): vgg11, vgg13", vgg16, vgg19
    # VGG (with batch norm): vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
    # DenseNet: densenet121, densenet169, densenet201
    # EfficientNet: efficientnet_b0, efficientnet_b7
    # Custom CNN architecture: custom
    "cnn_type": "densenet121",  
    # Pretrained or initialized weights
    "cnn_is_pretrained": True,  
    # Initialization type for non-pretrained cnns
    # Options: kaiming and xavier
    # Kaiming: Designed for ReLU-like activations (ReLU, LeakyReLU, GELU),
    # Default for modern CNNs (ResNet, EfficientNet, etc.) with ReLU/LeakyReLU.
    # Xavier: Designed for Sigmoid, Tanh, and linear activations.
    # Older architectures like AlexNet (originally used Tanh), Output layers with Sigmoid (e.g., binary classification)
    "cnn_initialization": "kaiming",  

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

    # Set to True, if checkpoints shall be saved during training
    "chckpt_save": True,  
    # Mininmun validation accuracy from which on checkpoints are saved
    "chckpt_min_acc": 0.40,  # 0.8

    #######################
    # CONFIDENCE ANALYZER #
    #######################

    # Min confidence for image sorting
    "ca_min_conf": 0.8,     # 80%

    #########
    # PATHS #
    #########

    "pth_data": "data/",
    "pth_train": "data/train/",
    "pth_test": "data/test/",
    "pth_checkpoint": "checkpoints/",
    "pth_plots": "plots/",
    "pth_prediction": "prediction/",
    # Dataset generator
    "pth_ds_gen_input": "dataset_gen/input/",
    "pth_ds_gen_output": "dataset_gen/output/",   
    # Automatic cross validation
    "pth_acv_results": "acv_results/",
    # Confidence analyzer results
    "pth_conf_analizer_results": "ca_results/",
 
}