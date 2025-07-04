####################
# Program settings #
####################

# Path handling
from pathlib import Path
# Base directory
BASE_DIR = Path(__file__).parent

setting = {

    ############
    # TRAINING #
    ############

    # Number of epochs
    "train_num_epochs": 40,  # 30
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

    # Augmentations for training
    "train_use_augment": False, 

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
    # Define classes
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
    "chckpt_min_acc": 0.8,  # 0.8 for 2 classes, 0.6 for 9 classess

    #######################
    # CONFIDENCE ANALYZER #
    #######################

    # Min confidence for image sorting
    "ca_min_conf": 0.8,     # 80%
    # Meximum number of checkpoints which are analyzed for a dataset
    "ca_max_ckpts": 2,
    # Method for best checkpoint selection
    # Options: balanced_sum, f1_score, min_difference, balanced_accuracy
    "ca_ckpt_select_method": 'balanced_sum',

    ##########
    # CAPTUM #
    ##########

    # Show overlay image in addidion to original and heatmap image
    "captum_show_overlay": True,
    # Determines how to handle positive/negative attributions. 
    # Options: "positive": Only positive attributions
    # The options "all", "negative" and "absolute_value" DO NOT WORK YET!
    "captum_sign": 'positive',
    # Colormap for the heatmap. Common options:
    # "viridis", "plasma", "magma", "inferno" (default), "cividis", etc
    # Any valid Matplotlib colormap name
    # Color map for original image
    "captum_cmap_orig": 'gray',
    # Color map for heatmap
    "captum_cmap_heatmap": 'coolwarm',   # viridis
    # Color map for overlay heatmap
    "captum_cmap_overlay": 'coolwarm',
    # Show color bar for different images
    "captum_show_color_bar_orig": True,
    "captum_show_color_bar_heatmap": True,
    "captum_show_color_bar_overlay": True,
    # The n_steps parameter in Integrated Gradients (IG) controls the number of 
    # interpolation steps used when approximating the integral for computing attributions
    # Higher n_steps = Smoother approximation of the integral (more accurate but slower)
    # Lower n_steps = Rougher approximation (faster but potentially noisier)
    # With too few steps (e.g., 5), attributions may appear pixelated or noisy
    # With more steps (e.g., 50), heatmaps become smoother but take longer to compute
    # Default: 50
    "captum_n_steps_ig": 25,
    # Image size of the result output
    "captum_output_size": 8,
    # Output figure resolution
    'captum_dpi': 300,   
    # Control transparency of the heatmap
    # 0.0: Heatmap completely transparent (only original image visible)
    # 1.0: Heatmap fully opaque (original image barely visible under intense colors)
    "captum_alpha_overlay": 0.5,
    # Controls how aggressively visualization focuses on the most important features by filtering out weaker attributions
    # e.g., 80 = top 20% most important pixels
    'captum_threshold_percentile': 70, 
    # Clips the top/bottom X% of attribution values before visualization to:
    # Improve contrast by ignoring extreme outliers
    # Make heatmaps more comparable across images
    # Default: 1
    'captum_outlier_perc': 1, 
    # Gaussian blur strength (lower = sharper)
    'captum_sigma': 0.5, 

    ###########
    # GradCAM #
    ###########

    # Parameters for second iteration with blurring
    "gradcam_second_iteration": True,
    # Percentage of most prominent pixels to blur (0-1)
    "gradcam_threshold_percent": 0.40,
    # Gaussian blur strength
    "gradcam_blurr_sigma": 15,


    #########
    # PATHS #
    #########

    "pth_data": BASE_DIR / "data/",
    "pth_train": BASE_DIR / "data/train/",
    "pth_test": BASE_DIR / "data/test/",
    "pth_checkpoint": BASE_DIR / "checkpoints/",
    "pth_plots": BASE_DIR / "plots/",
    "pth_prediction": BASE_DIR / "prediction/",
    # Dataset generator
    "pth_ds_gen_input": BASE_DIR / "dataset_gen/input/",
    "pth_ds_gen_output": BASE_DIR / "dataset_gen/output/",   
    # Automatic cross validation
    "pth_acv_results": BASE_DIR / "acv_results/",
    # Confidence analyzer results
    "pth_conf_analizer_results": BASE_DIR / "ca_results/",
 
}