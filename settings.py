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
    "train_num_epochs": 10, # 40
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
    "ds_val_from_train_split": False,    # False
    # Validation split from images in folder data/test/ (False or percentage 0.0-1.0)
    "ds_val_from_test_split": 1.0,    # 1.0
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

    ########################
    # UTILITIES SETTINGS   #
    ########################

    # Dataset Merger
    # Maximum depth to scan for images (None = unlimited)
    "util_merger_recursive_depth": None,
    # True = copy duplicates with rename, False = skip duplicates
    "util_merger_copy_duplicates": False,
    # Print progress messages
    "util_merger_verbose": True,

    # Dataset Remover
    # Target number of images per folder after reduction
    "util_remover_target_per_folder": 500,
    # Random seed for reproducibility
    "util_remover_seed": 42,
    # Print progress messages
    "util_remover_verbose": True,

    # Dataset Splitter
    # List of ratios for splitting (e.g., [0.5, 0.5] for 50/50 split)
    "util_splitter_ratios": [0.3],
    # Random seed for reproducibility
    "util_splitter_random_seed": 42,

    # Dataset Subtraction
    # Name of the first dataset folder (inside input/)
    "util_subtraction_dataset_a": "dataset_a",
    # Name of the second dataset folder (inside input/)
    "util_subtraction_dataset_b": "dataset_b",
    # Name of the result folder (inside output/)
    "util_subtraction_result": "result_dataset",
    # Print progress messages
    "util_subtraction_verbose": True,

    # Dataset Merger
    # List of image extensions to collect
    "util_merge_extensions": ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'],
    # Print progress messages
    "util_merge_verbose": True,

    # Sort Images by Frame
    # Image naming format: "flux" or "stylegan"
    "util_sort_frame_format": "flux",
    # Print progress messages
    "util_sort_frame_verbose": True,

    # Sort Images by Seed
    # Print progress messages
    "util_sort_seed_verbose": True,

    ##########################
    # PREPROCESSING SETTINGS #
    ##########################

    # CZI Export
    # Number of tiles in the mosaic file in x and y direction
    "preproc_num_tiles": {'x': 20, 'y': 26},
    # Define sharpest plane of z-stack (None = auto-detect)
    "preproc_sharpest_z_plane": None,
    # Number of image channels: 1=grayscale, 3=RGB
    "preproc_num_img_channels": 1,
    # Scale factor when mosaic image is imported
    "preproc_czi_import_scale": 1.0,
    # Extension of a Carl Zeiss image file
    "preproc_czi_img_ext": '.czi',
    # Percentile normalization
    "preproc_perc_min": 5.0,
    "preproc_perc_max": 97.0,
    # Define region for focus stacking
    "preproc_focus_stacking_region": 5,
    # Size of sliced images from original size
    "preproc_slice_size": {'x': 1766, 'y': 1766},
    # Size if resized slices for training
    "preproc_slice_resize": {'x': 512, 'y': 512},

    # Caption Generator
    # Caption mode: "cell_line_only", "phenotype_only", or "both"
    # cell_line_only: "KO_1096-01 cells, grayscale"
    # phenotype_only: "knockout cells, grayscale"
    # both: "KO_1096-01 knockout cells, grayscale"
    "preproc_caption_mode": "phenotype_only",
    # Overwrite existing caption files
    "preproc_caption_overwrite": True,
    # Cell line definitions
    "preproc_caption_wt_lines": ["WT_1618-02", "WT_JG", "WT_JT", "WT_KM", "WT_MS"],
    "preproc_caption_ko_lines": ["KO_1096-01", "KO_1618-01", "KO_BR2986", "KO_BR3075"],
    # Image extensions to process
    "preproc_caption_image_extensions": ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'],

    ##########################
    # EXPORT & PLOTTING      #
    ##########################

    ### General Export Settings ###
    # Mode for all export tools: 'auto' (auto-detect), 'crossval', or 'single'
    # 'auto': Detects from folder structure (dsXX = crossval, timestamp = single)
    # 'crossval': Forces cross-validation mode (expects dsXX folders)
    # 'single': Forces single training mode (expects timestamp folders)
    "export_mode": "auto",

    ### Excel Exporter (extract_metrics_to_excel.py) ###
    # Converts TensorBoard logs to Excel with charts

    # Which epoch to use for ROC curves
    # 'composite_score': Uses epoch with highest composite score from TensorBoard
    # 'balanced_accuracy': Uses epoch with highest balanced accuracy
    # 'last': Uses the last available epoch
    # integer: Uses specific epoch number (1-indexed)
    "export_excel_roc_epoch": "balanced_accuracy",
    # Which epoch to use for Precision-Recall curves
    # Same options as roc_epoch above
    "export_excel_pr_epoch": "balanced_accuracy",


    ### UMAP/t-SNE/PaCMAP Plotter (plot_umap_publication.py) ###
    # Creates publication-ready plots from embedding CSV files

    # Output format: 'png', 'tiff', 'svg', 'pdf'
    # 'tiff': Recommended for print publications (600 DPI)
    # 'png': Good for presentations, smaller file size
    # 'svg'/'pdf': Vector graphics, editable in Illustrator/Corel Draw
    "export_umap_format": "tiff",
    # Resolution for raster formats (PNG, TIFF)
    # 300: Minimum for presentations/posters
    # 600: Recommended for print publications (Nature, Science, Cell)
    # 1200: Extreme quality (very large files)
    "export_umap_dpi": 600,
    # Size of each point in the scatter plot (points = 1/72 inch)
    # 1-5: Very small, good for massive datasets (>50k points)
    # 10-20: Small, good for large datasets (10k-50k points)
    # 30-50: Medium, good for medium datasets (1k-10k points)
    # 80-150: Large, good for small datasets (<1k points)
    # 30: 2,5-5,0 points
    "export_umap_point_size": 30,
    # Transparency of points (0.0 = fully transparent, 1.0 = fully opaque)
    # For overlapping points, lower alpha reveals density patterns
    # 0.3-0.5: Recommended for large datasets (overlapping points)
    # 0.5-0.8: Good for medium datasets
    # 0.9-1.0: Only for small datasets (<500 points)
    "export_umap_point_alpha": 0.5,
    # Show 95% confidence ellipses around each group
    # True: Adds ellipses showing group distribution (good for presentations)
    # False: Cleaner look (recommended for publications with many groups)
    "export_umap_show_ellipses": False,
    # Color palette for groups
    # 'colorblind': Color-blind friendly (recommended for publications)
    # 'jet': Classic rainbow (high contrast but not color-blind friendly)
    # 'viridis': Modern, perceptually uniform
    # 'plasma': Modern, perceptually uniform
    # 'tab10': 10 distinct colors (good for up to 10 groups)
    # 'tab20': 20 distinct colors (good for up to 20 groups)
    # Any valid matplotlib colormap name
    "export_umap_palette": "jet",
    # Preserve data aspect ratio (True = 1:1, False = stretch to fit)
    # True: Maintains the true shape of the embedding (recommended)
    # False: Stretches to fill axes (can distort distances)
    "export_umap_fixed_aspect": True,
    # Height of the axes in inches (ALL plots have identical axes height)
    # 4-6: Good for publications
    # 6-8: Good for presentations
    # 8-10: Good for posters
    # Larger values = more detail, larger files
    "export_umap_axes_height": 5.0,
    # Show/hide legend
    # True: Shows legend with group names
    # False: Hides legend (useful if groups are labeled in the plot)
    "export_umap_show_legend": True,
    # Legend position
    # 'inside': Places legend inside the plot area (recommended for small number of groups)
    # 'outside': Places legend to the right of the plot (recommended for >8 groups)
    # 'auto': Automatically chooses based on number of groups
    "export_umap_legend_position": "outside",
    # Size of legend markers (points)
    # 30-50: Small legend markers
    # 80-120: Medium legend markers (recommended)
    # 150-200: Large legend markers
    "export_umap_legend_marker_size": 100,
    # Font family for all text
    # 'Arial': Standard sans-serif (recommended for publications)
    # 'Helvetica': Similar to Arial
    # 'Times New Roman': Serif font (for some journals)
    # 'DejaVu Sans': Open-source alternative
    "export_umap_font_family": "Arial",
    # Font size for axis labels (points)
    # 12-16: Good for publications
    # 18-24: Good for presentations
    # 24-36: Good for posters
    "export_umap_axis_label_size": 22,
    # Font size for legend text (points)
    # Same recommendations as axis_label_size
    "export_umap_legend_font_size": 22,
    # Font size for tick labels (axis numbers) (points)
    # Same recommendations as axis_label_size
    "export_umap_tick_label_size": 22,
    # Show background grid
    # True: Adds grid lines (helps readability)
    # False: Clean background (minimalist)
    "export_umap_show_grid": True,
    # Grid line transparency (0.0 = invisible, 1.0 = solid)
    # 0.2-0.4: Subtle grid (recommended)
    # 0.5-0.8: More visible grid
    "export_umap_grid_alpha": 0.3,
    # Margins around the axes in inches
    # Control spacing between axes and figure edges
    "export_umap_left_margin": 1.2,         # Space for y-axis label and tick labels
    "export_umap_right_margin": 1.0,        # Space on right side of axes
    "export_umap_bottom_margin": 0.9,       # Space for x-axis label and tick labels
    "export_umap_top_margin": 0.5,          # Space above axes
    "export_umap_legend_margin_extra": 3.0, # Extra space for outside legend (right side)


    ### Confusion Matrix Plotter (plot_conf_matrix_publication.py) ###
    # Creates publication-ready confusion matrix plots from JSON files

    # Output format: 'png', 'tiff', 'svg', 'pdf'
    "export_cm_format": "tiff",
    # Resolution for raster formats (PNG, TIFF)
    # 300: Minimum for presentations
    # 600: Recommended for print publications
    "export_cm_dpi": 600,
    # Normalization mode for the confusion matrix
    # 'rows': Row-normalized (shows recall/TPR per class)
    #        Each row sums to 1. Good for showing per-class accuracy.
    # 'columns': Column-normalized (shows precision per class)
    #           Each column sums to 1. Good for showing prediction reliability.
    # 'none': Raw counts (no normalization)
    #        Good for debugging or small datasets.
    "export_cm_normalize": "rows",
    # Show raw counts in cell annotations alongside normalized values
    # Example: "0.750\n(150)" shows 75% normalized with 150 raw count
    # True: Shows both values (recommended for publications)
    # False: Shows only normalized values
    "export_cm_show_counts": True,
    # Combined mode: Show raw + normalized side-by-side
    # True: Two matrices side by side (raw and normalized)
    # False: Single matrix (normalized as selected)
    "export_cm_combined": False,
    # Use fixed axes height for all plots
    # True: All plots have identical axes height (recommended for consistency)
    # False: Uses manual figsize
    "export_cm_use_fixed_height": True,
    # Height of the axes in inches (when use_fixed_height = True)
    # 4-6: Good for small matrices (<10 classes)
    # 6-8: Good for medium matrices (10-20 classes)
    # 8-12: Good for large matrices (>20 classes)
    "export_cm_fixed_height": 6.0,
    # Colormap for the heatmap
    # 'Blues': Classic blue gradient (recommended)
    # 'Reds': Red gradient
    # 'Greens': Green gradient
    # 'viridis': Modern perceptually uniform
    # 'plasma': Modern perceptually uniform
    # Any valid matplotlib colormap name
    "export_cm_cmap": "Blues",
    # Show/hide plot title
    # True: Shows title with overall accuracy
    # False: No title (cleaner)
    "export_cm_show_title": True,
    # Show/hide axis labels
    # True: Shows "Predicted Class" and "True Class"
    # False: No axis labels (cleaner)
    "export_cm_show_axis_labels": True,
    # Show/hide colorbar
    # True: Shows colorbar indicating value scale
    # False: No colorbar (cleaner)
    "export_cm_show_colorbar": True,
    # Show per-class accuracy on y-axis labels
    # Example: "KO_1096-01\n(93.1%)" shows class name with accuracy
    # True: Shows accuracy in labels (informative)
    # False: Shows only class names (cleaner)
    "export_cm_show_per_class_acc": False,
    # Show overall accuracy in the title
    # Example: "Confusion Matrix\nOverall Accuracy: 87.3%"
    # True: Shows overall accuracy in title
    # False: Title only shows matrix name
    "export_cm_show_overall_acc": True,
    # Font family for all text
    # 'Arial': Recommended for publications
    # 'Times New Roman': Serif font
    "export_cm_font_family": "Arial",
    # Master font size: If set, all fonts use this size (overrides individual sizes)
    # None: Use individual sizes below
    # 10-14: Good for publications
    # 16-20: Good for presentations
    "export_cm_master_font_size": None,
    # Individual font sizes (only used when master_font_size is None)
    # All sizes are in points (1/72 inch)
    "export_cm_axis_label_size": 30,           # Size for axis labels (xlabel, ylabel)
    "export_cm_title_font_size": 30,           # Size for plot title
    "export_cm_tick_label_size": 25,           # Size for tick labels (class names)
    "export_cm_annotation_font_size": 16,      # Size for numbers inside matrix cells
    "export_cm_legend_font_size": 30,          # Size for colorbar label
    # Rotation of tick labels (degrees)
    # 45: Good for long class names (recommended)
    # 90: Vertical (good for many classes)
    # 0: Horizontal (only for short names)
    "export_cm_xtick_rotation": 45,            # Rotation of x-axis labels (predicted classes)
    "export_cm_ytick_rotation": 0,             # Rotation of y-axis labels (true classes)
    # Number of decimal places for annotation values
    # 3: Shows 3 decimal places (e.g., 0.123)
    # 2: Shows 2 decimal places (e.g., 0.12)
    # 1: Shows 1 decimal place (e.g., 0.1)
    "export_cm_annotation_decimal_places": 3,


    ### Training Metrics Plotter (plot_train_metrics_publication.py) ###
    # Creates publication-ready plots from TensorBoard logs

    # Output format: 'png', 'tiff', 'svg', 'pdf'
    "export_train_format": "tiff",
    # Resolution for raster formats (PNG, TIFF)
    # 300: Good for presentations
    # 600: Recommended for print publications
    "export_train_dpi": 300,
    # Which epoch to use for ROC curves (same options as excel exporter)
    "export_train_roc_epoch": "balanced_accuracy",
    "export_train_pr_epoch": "balanced_accuracy",
    # Use fixed axes height for all plots
    # True: All plots have identical axes height (recommended for consistency)
    # False: Different plots may have different heights
    "export_train_use_fixed_height": True,
    # Height of the axes in inches (when use_fixed_height = True)
    # 4-6: Good for publications
    # 6-8: Good for presentations
    "export_train_fixed_height": 5.0,
    # Master font size for all text
    # 10-14: Good for publications
    # 16-22: Good for presentations
    # 24-36: Good for posters
    "export_train_master_font_size": 22,
    # Line width for plot lines (points)
    # 1.0-1.5: Thin lines (good for publications with many lines)
    # 2.0-2.5: Medium lines (recommended)
    # 3.0-4.0: Thick lines (good for presentations)
    "export_train_line_width": 2.0,
    # Show markers at data points
    # True: Shows dots at each epoch (helps identify individual points)
    # False: Only shows lines (cleaner look)
    "export_train_show_markers": True,
    # Size of markers (points)
    # 2-4: Small markers (good for many epochs)
    # 5-8: Medium markers (recommended)
    # 10-15: Large markers (good for few epochs)
    "export_train_marker_size": 5,
    # Show every Nth marker (reduces clutter)
    # 1: Show all markers (recommended for <50 epochs)
    # 2: Show every 2nd marker
    # 5: Show every 5th marker (recommended for >50 epochs)
    # 10: Show every 10th marker
    "export_train_marker_frequency": 1,
    # Smoothing factor for curves
    # 0.0: No smoothing (raw data, recommended)
    # 0.5: Moderate smoothing
    # 0.9: Strong smoothing (hides noise, may hide patterns)
    # 0.95: Very strong smoothing
    "export_train_smoothing": 0.0,
    # Font family for all text
    "export_train_font_family": "Arial",
    # Individual font sizes (when master_font_size is None)
    "export_train_axis_label_size": 12,        # Size for axis labels
    "export_train_title_font_size": 14,        # Size for plot titles
    "export_train_tick_label_size": 10,        # Size for tick labels
    "export_train_legend_font_size": 10,       # Size for legend text
    # Show/hide legend
    "export_train_show_legend": True,
    # Grid settings
    "export_train_show_grid": True,
    "export_train_grid_alpha": 0.3,            # Grid transparency (0.0-1.0)
    "export_train_grid_linestyle": "--",       # Grid line style: '-', '--', ':', '-.'
    "export_train_grid_color": "gray",         # Grid line color
    # Minimum class accuracy threshold (shown as red dashed line)
    # 0.60: 60% threshold (typical minimum acceptable accuracy)
    # 0.65: 65% threshold
    # None: No threshold line shown
    "export_train_min_class_acc_threshold": 0.65,
    #Plot Selection Flags
    # Set to True to generate, False to skip
    "export_train_plot_loss": True,                 # Training/validation loss curves
    "export_train_plot_accuracy": True,             # Training/validation accuracy (weighted + standard)
    "export_train_plot_f1": True,                   # Macro and weighted F1 scores
    "export_train_plot_lr": False,                  # Learning rate schedule
    "export_train_plot_auc": False,                 # AUC over epochs (overall + per-class)
    "export_train_plot_ap": False,                  # Average Precision over epochs
    "export_train_plot_per_class_acc": True,        # Per-class accuracy curves
    "export_train_plot_class_weights": False,       # Bar chart of loss function class weights
    "export_train_plot_class_counts": False,        # Bar chart of class distribution
    "export_train_plot_composite": False,           # Composite score (balanced performance metric)
    "export_train_plot_balanced_acc": True,         # Balanced accuracy over epochs
    "export_train_plot_class_std": False,           # Standard deviation of class accuracies
    "export_train_plot_min_class_acc": False,       # Minimum class accuracy over epochs
    "export_train_plot_gpu_memory": False,          # GPU memory usage over training
    "export_train_plot_roc": True,                  # ROC curves (requires .npz files)
    "export_train_plot_pr": True,                   # Precision-Recall curves (requires .npz files)
    #Legend Outside Settings
    # Place legend outside the plot (right side)
    # True: Legend placed to the right of the plot (recommended for many classes)
    # False: Legend placed inside the plot
    "export_train_per_class_legend_outside": True,  # Per-class accuracy legend outside
    "export_train_roc_legend_outside": True,        # ROC curves legend outside
    "export_train_pr_legend_outside": True,         # PR curves legend outside
    # Legend Position Settings (Inside)
    # Each plot type can have its own legend position
    # Options: 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'upper center', etc.
    "export_train_loss_legend_loc": "upper right",
    "export_train_accuracy_legend_loc": "lower right",
    "export_train_f1_legend_loc": "lower right",
    "export_train_per_class_legend_loc": "best",
    "export_train_roc_legend_loc": "lower right",
    "export_train_pr_legend_loc": "lower left",
    "export_train_class_std_legend_loc": "upper right",
    "export_train_min_class_acc_legend_loc": "lower right",
    "export_train_composite_legend_loc": "lower right",
    "export_train_balanced_acc_legend_loc": "lower right",
    "export_train_gpu_memory_legend_loc": "best",
    "export_train_lr_legend_loc": "best",
    "export_train_auc_legend_loc": "lower right",
    "export_train_ap_legend_loc": "lower right",

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

    # ===== Checkpoints (for loading existing models) =====
    "pth_checkpoint": BASE_DIR / "checkpoints/",

    # ===== Input Folder (Analysis Inputs) =====
    "pth_input": BASE_DIR / "input/",
    # All analysis inputs go directly into input/ (no subfolders)

    # ===== Output Folder (Analysis Outputs) =====
    "pth_output": BASE_DIR / "output/",
    # All analysis outputs go into output/ with subfolders created by each script
    # e.g., output/train/, output/cross_validation/, output/conf_analyzer/, etc.
}