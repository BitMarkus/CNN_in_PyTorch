####################
# Program settings #
####################

# ===== Standard Library Imports =====
from pathlib import Path
# Base directory
BASE_DIR = Path(__file__).parent

setting = {

    ############
    # TRAINING #
    ############

    # Number of epochs
    "train_num_epochs": 40,
    # Batch size for training and validation datasets
    "ds_batch_size": 50,

    # Optimizer:
    # Options: "SGD", "ADAM", and "ADAMW"
    "train_optimizer_type": "ADAMW",
    # Initial learning rate (later determined by lr scheduler)
    # ADAM: 0.0001 (1e-4) - 0.0003 (3e-4)
    # ADAMW: 0.0001 (3e-4) - 0.0005 (5e-4)
    # SGD: 0.01-0.001, 0.0001 (1e-4) for pretrained weights!
    "train_init_lr": 1e-4,
    # Weight decay = L2 regularization
    # ADAM: 1e-4 (0.0001) - 1e-3 (0.001): 1e-4
    # ADAMW: 1e-3 (0.001) - 1e-2 (0.01): 1e-3
    # SGD: 1e-4
    "train_weight_decay": 1e-3,
    # Momentum
    "train_sgd_momentum": 0.9,
    # Nesterov momentum for SGD (only works if momentum > 0)
    "train_sgd_use_nesterov": True,
    # ADAM/ADAMW beta 1 and 2
    "train_adam_beta1": 0.9,
    "train_adam_beta2": 0.99,

    # Loss function:
    # Parameter to use weighted loss function or normal loss function
    # Also training metrics will change to weighted versions
    # Useful for class imbalance
    "train_use_weighted_loss": True,

    # Learning rate scheduler:
    # Number of steps after which the lr is multiplied by the lr multiplier
    # Warmup scheduler:
    "train_lr_warmup_epochs": 5,
    # CosineAnnealingLR - Minimum learning rate (eta_min) that the scheduler decays toward:
    # SGD: 1e-5
    # ADAM: 1e-4
    # ADAMW: 1e-5 - 1e-6
    "train_lr_eta_min": 1e-5,

    ###########
    # DATASET #
    ###########

    # Dataset parameters:
    # Shuffle dataset
    "ds_shuffle": True,
    # Shuffle seed
    "ds_shuffle_seed": 44,
    # How many subprocesses are used to load data in parallel
    "ds_num_workers": 3,
    # Validation split settings
    # Validation split from images in folder data/train/ (False or percentage 0.0-1.0)
    "ds_val_from_train_split": False,
    # Validation split from images in folder data/test/ (False or percentage 0.0-1.0)
    "ds_val_from_test_split": 1.0,
    # Export validation images to a folder
    "ds_save_val_images": False,

    ##########################
    # CLASSES AND CELL LINES #
    ##########################

    ### PROJECT 1 (CLN7) ###
    # Define classes
    # 2 classes (WT and KO):
    "classes": ["KO", "WT"],
    # 9 classes (one for each cell line):
    # "classes": ["KO_1096-01", "KO_1618-01", "KO_BR2986", "KO_BR3075", "WT_1618-02", "WT_JG", "WT_JT", "WT_KM", "WT_MS"],
    # Define cell lines (for dataset generator)
    "wt_lines": ["WT_1618-02", "WT_JG", "WT_JT", "WT_KM", "WT_MS"],
    "ko_lines": ["KO_1096-01", "KO_1618-01", "KO_BR2986", "KO_BR3075"],

    ### PROJECT 2 (MDD) ###
    # Define classes
    # 2 classes (WT and MDD):
    # "classes": ["MDD", "WT"],
    # Define cell lines (for dataset generator)
    # "wt_lines": ["WT_BJ", "WT_LF", "WT_MP", "WT_MW", "WT_NH"],
    # "ko_lines": ["MMD_155", "MMD_160", "MMD_169", "MMD_177"],
    
    ###############
    # CHECKPOINTS #
    ###############

    # Set to True, if checkpoints shall be saved during training
    # Checkpoint saving occurs when either balanced accuracy OR composite score improves
    # compared to the previous best checkpoint, and when minimum thresholds are met (if enabled).
    "chckpt_save": True,

    # Checkpoint selection method:
    # Options: "balanced_accuracy", "composite_score", "both"
    # "balanced_accuracy": Uses only balanced accuracy for checkpoint selection
    # "composite_score": Uses only composite score for checkpoint selection
    # "both": Uses both methods
    "chckpt_selection_method": "balanced_accuracy",

    # Composite score settings:
    # Minimum acceptable per-class accuracy (0.60 = 60%)
    "chckpt_min_class_acc_threshold": 0.65,
    # Penalty weight for checkpoint selection
    # Higher = more penalty for class imbalance (range: 1.0 to 4.0)
    "chckpt_penalty_weight": 2.0,

    # Balanced accuracy settings:
    # Minimum acceptable overall balanced accuracy
    "chckpt_min_balanced_acc_threshold": 0.65,
    # Minimum per-class accuracy for balanced accuracy selection
    # Set to 0.0 to disable (use only balanced accuracy threshold)
    "chckpt_min_per_class_acc_balanced": 0.60,

    #########################
    # AUTO CROSS VALIDATION #
    #########################

    # Determines the source folder for training data
    # Options: "mixed", "synthetic_only", "real_only"
    "train_data_source": "real_only",

    #################
    # AUGMENTATIONS #
    #################

    # Use augmentations
    "train_use_augment": True,

    # FLIP AND ROTATION AUGMENTATIONS:
    # Horizontal flip probability
    "aug_hori_flip_prob": 0.5,
    # Vertical flip probability
    "aug_vert_flip_prob": 0.5,
    # Probability of 90° angle rotations
    "aug_90_angle_rot_prob": 0.5,
    # Probability of small angle rotations
    "aug_small_angle_rot_prob": 0.5,
    # Small-angle rotation
    "aug_small_angle_rot": 10,
    # Fill color for gaps due to small angle rotation
    # fill=0: black background, fill=255: white background
    "aug_small_angle_fill_gray": 100,
    "aug_small_angle_fill_rgb": (255, 255, 255),

    # INTENSITY AUGMENTATIONS:
    "aug_intense_prob": 0.5,
    "aug_brightness": 0.2,
    "aug_contrast": 0.2,
    "aug_saturation": 0.2,  # only for RGB images
    # Gamma correction
    # Gamma = 1: No change. The image looks "natural" (linear brightness)
    # Gamma < 1 (e.g., 0.5): Dark areas get brighter, bright areas stay mostly the same
    # Gamma > 1 (e.g., 2.0): Bright areas get darker, dark areas stay mostly the same
    "aug_gamma_prob": 0.4,
    "aug_gamma_min": 0.7,
    "aug_gamma_max": 1.3,

    # OPTICAL AUGMENTATIONS:
    # Gaussian Blur Parameters
    # Probability
    "aug_gauss_prob": 0.3,
    # Kernel size
    "aug_gauss_kernel_size": 5,
    # Sigma: controls the "spread" of the blur (how intense/smooth it is)
    "aug_gauss_sigma_min": 0.1,
    "aug_gauss_sigma_max": 0.5,
    # Poisson noise
    # Probability
    "aug_poiss_prob": 0.4,
    # Controls how much the noise depends on image brightness
    # Suggested range: 0.01-0.1 (higher = more noise)
    "aug_poiss_scaling": 0.05,
    # Noise Strength: Final noise intensity multiplier
    "aug_poiss_noise_strength": 0.1,

    # LABEL SMOOTHING
    # 0.0: No smoothing (default CrossEntropyLoss). Hard labels (0 or 1)
    # 0.1: 10% smoothing (e.g., correct class = 0.9, others share 0.1/classes)
    # 0.2: 20% smoothing, etc.
    "train_label_smoothing": 0.1,

    #########
    # MODEL #
    #########

    # Name of the model architecture:
    # ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
    # ResNeXt variants: resnext101_32x8d, resnext101_64x4d
    # AlexNet: alexnet
    # VGG (without batch norm): vgg11, vgg13, vgg16, vgg19
    # VGG (with batch norm): vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
    # DenseNet: densenet121, densenet169, densenet201
    # EfficientNet: efficientnet_b0, efficientnet_b3, efficientnet_b4, efficientnet_b7
    # ConvNeXt: convnext_tiny, convnext_small
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

    # Dropout for CUSTOM CNN architecture (only used if cnn_type = "custom")
    "cnn_dropout": 0.3,

    ##########
    # IMAGES #
    ##########

    # Training image dimensions
    "img_width": 512,
    "img_height": 512,
    "img_channels": 1,

    ################
    # CLASS SORTER #
    ################

    # Selection mode and value: "top_n", "threshold" or "interval"
    "sort_selection_mode": "interval",
    # Number of images for top_n, or threshold value
    # If sort_selection_mode is "top_n", this selects the top X most confident images per class (e.g. 50)
    # If sort_selection_mode is "threshold", this needs to be a single number (e.g. 0.2)
    # If sort_selection_mode is "interval", this needs to be a list of min/max values (e.g. [0.2, 0.8])
    "sort_selection_value": [0.5, 1.0],
    # Filter criteria: 'confidence_only', 'logits_only', or 'combined'
    'sort_filter_mode': 'confidence_only',
    # Minimum max_logit value to keep
    "sort_logit_threshold": 0.0,
    # Rename files with either confidence scores, logit values, or both
    "sort_rename_images": True,
    # Batch size for prediction
    "sort_pred_batch_size": 50,
    # Interval borders for confidence statistics
    "sort_conf_intervals": [10, 20, 30, 40, 50, 60, 70, 80, 90],
    # Interval borders for logit statistics
    "sort_logit_intervals": [-10, -5, -2, 0, 2, 5, 10],

    ##################
    # CLASS ANALYZER #
    ##################

    # Rename files with confidence scores
    'analyze_rename_with_confidence': False,
    # Include logits in filenames
    'analyze_include_logits_in_rename': False,

    #######################
    # CONFIDENCE ANALYZER #
    #######################

    # Min confidence for image sorting
    "ca_min_conf": 0.8,
    # Max confidence for image sorting
    'ca_max_conf': 1.0,
    # Filter type for image sorting
    # "correct": Images correctly classified in all test folds, with confidence within [min_conf, max_conf] -> Reliable predictions for downstream analysis
    # "incorrect": Images incorrectly classified in all test folds, with confidence within [min_conf, max_conf] -> Systematic errors to investigate
    # "low_confidence": Images with confidence below min_conf in all test folds (ignores max_conf) (regardless of correctness) -> Ambiguous cases needing manual review
    # "unsure": Images with confidence within [min_conf, max_conf] (regardless of correctness) -> Intermediate-confidence predictions
    'ca_filter_type': 'correct',
    # Maximum number of checkpoints which are analyzed for a dataset
    "ca_max_ckpts": 1,
    # Method for best checkpoint selection
    # Options: 'balanced_sum', 'f1_score', 'min_difference', 'balanced_accuracy' and 'composite_score'
    "ca_ckpt_select_method": 'balanced_accuracy',
    # Which confusion matrix JSON file to use for checkpoint selection
    # When you have BOTH validation AND test evaluations during training, this decides which metrics to use for selecting the "best" checkpoint
    # Options: "validation", "test"
    "ca_use_test_cm": "validation",
    # Which set of images to run predictions on for confidence analysis
    # Decides which actual images to feed through the model for prediction
    # Options: "validation", "test", "all"
    "ca_split_to_use": "validation",

    ###########
    # GradCAM #
    ###########

    # Parameters for second iteration with blurring
    "gradcam_second_iteration": False,
    # Percentage of most prominent pixels to blur (0-1)
    "gradcam_threshold_percent": 0.40,
    # Gaussian blur strength
    "gradcam_blurr_sigma": 15,
    # Export mode:
    # False: Composition of original, gradcam and overlay images
    # True: Only export of gradcam image in 512x512 px
    "gradcam_export_only_overlay": True,

    #######################
    # DIMENSION REDUCTION #
    #######################

    # Choose the method for dimension reduction
    "dimred_use_umap": True,
    "dimred_use_tsne": True,
    "dimred_use_trimap": True,
    "dimred_use_pacmap": True,

    # Mode can be "train", "test", or "groups"
    # "test": Images are test images in the data/test/ folder
    # "train": Images are training images in the data/training/ folder
    # "groups": Arbitrary number of groups, defined by the number of folders in the prediction/ folder
    "dimred_mode": "groups",
    # Group configuration mode: 'auto' (detects folders) or 'manual' (uses explicit mapping)
    "dimred_group_mode": "auto",
    # For manual mode: Explicit mapping of folder names to (display_name, label)
    "dimred_group_mapping": {
        "WT": ("Real WT", 0),
        "WT_GAN": ("Fake WT", 1),
        "KO": ("Real KO", 2),
        "KO_GAN": ("Fake KO", 3),
        # Add more groups as needed
    },
    # Color palette for dimensionality reduction plots
    # Options: 'default' or any matplotlib colormap name, like 'rainbow', 'jet', etc.
    'dimred_color_palette': 'jet',

    # Export format of raw data
    'dimred_export_format': 'csv',  # 'csv' or 'json'

    # UMAP parameters
    "dimred_umap_n_neighbors": 15,
    "dimred_umap_min_dist": 0.1,
    # t-SNE parameters
    "dimred_tsne_perplexity": 30,
    "dimred_tsne_learning_rate": 'auto',
    # TriMAP parameters
    "dimred_trimap_n_inliers": 10,
    "dimred_trimap_n_outliers": 5,
    # PaCMAP parameters
    "dimred_pacmap_n_neighbors": 15,
    "dimred_pacmap_MN_ratio": 0.5,
    "dimred_pacmap_FP_ratio": 2.0,

    #############
    # FID SCORE #
    #############

    # Batch size for processing images for FID calculation
    "fid_batch_size": 32,
    # When this is set to True, the number of images is determined by the folder with the least images
    # to balance the number of images for the calculation
    "fid_balance_samples": True,
    # Random seed for randomly choosing images if balance samples is set to True
    "fid_random_seed": 123,

    #########
    # PATHS #
    #########

    # ===== Training & Validation =====
    "pth_data": BASE_DIR / "data/",
    "pth_train": BASE_DIR / "data/train/",
    "pth_test": BASE_DIR / "data/test/",

    # ===== Dataset Generator =====
    "pth_ds_gen_input_synthetic": BASE_DIR / "dataset_gen/input_synthetic/",
    "pth_ds_gen_input_real": BASE_DIR / "dataset_gen/input_real/",
    "pth_ds_gen_input_mixed": BASE_DIR / "dataset_gen/input_mixed/",
    "pth_ds_gen_output": BASE_DIR / "dataset_gen/output/",

    # ===== Input Folder (Analysis Inputs) =====
    "pth_input": BASE_DIR / "input/",
    # All analysis inputs go directly into input/ (no subfolders)

    # ===== Output Folder (Analysis Outputs) =====
    "pth_output": BASE_DIR / "output/",
    # All analysis outputs go into output/ with subfolders created by each script
    # e.g., output/train/, output/cross_validation/, output/conf_analyzer/, etc.
}