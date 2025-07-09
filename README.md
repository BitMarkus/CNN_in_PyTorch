# CNN_in_PyTorch
A CNN (convolutional neural network, different architectures) for classification and analysis of microscopic images (in python, pytorch) 

Tested with the following environment: NVIDIA RTX 4090 or NVIDIA RTX A5000 (24 GB VRAM), Python: 3.8.19, Pytorch: 2.3.1, CUDA: 12.1, cuDNN: 8907

## 1. Create CNN Network
**Description**: Initializes a new convolutional neural network based on predefined or custom architectures.

#### Key Settings (from `settings.py`)
| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `cnn_type` | str | Architecture to load | `"densenet121"`, `"custom"` |
| `cnn_is_pretrained` | bool | Use pretrained weights | `True` |
| `cnn_initialization` | str | Weight init method (`"kaiming"`/`"xavier"`) | `"kaiming"` |
| `img_channels` | int | Input channels (1=grayscale, 3=RGB) | `3` |
| `img_width`/`img_height` | int | Input dimensions | `224` |
| `classes` | list | Class names | `["WT", "KO"]` |

#### Required Folder Structure
```plaintext
data/
├── train/               # Training images (set in `pth_train`)
│   ├── WT/              # Wild-type class folder
│   └── KO/              # Knockout class folder
└── predict/             # Prediction images (set in `pth_prediction`)
    ├── unclassified/    # Optional: for manual classification
    └── results/         # Optional: prediction outputs

checkpoints/             # Saved models (set in `pth_checkpoint`)
├── ckpt_densenet121_e10_vacc85.model
└── ckpt_custom_e5_vacc72.model
```

#### Example Workflow
1. Configure `settings.py`:
   ```python
   cnn_type = "densenet121"
   classes = ["WT", "KO"]
   img_channels = 3

2. Run program and select Option 1:
:NEW CNN NETWORK:
Number of classes: 2
Classes: WT, KO
Creating new densenet121 network...
New network was successfully created.
Trainable parameters: 7,978,856   

## 2. Show Network Summary  
**Description**: Displays detailed architecture information including layer dimensions and parameters.

#### Example Output
```plaintext
+---------------------+-------------------+------------+-----------+
| Layer (type)        | Output Shape      | Param #    | Trainable |
+---------------------+-------------------+------------+-----------+
| Conv2d              | [32,64,112,112]   | 9,408      | True      |
| BatchNorm2d         | [32,64,112,112]   | 128        | True      |
| DenseBlock          | [32,256,28,28]    | 1,200,000  | True      |
| Linear              | [32,2]            | 512        | True      |
+---------------------+-------------------+------------+-----------+

Input: [32, 3, 224, 224] (Device: cuda:0)
Total params: 7,978,856
Trainable params: 7,978,856
Non-trainable params: 0
```

---

## 3. Load Training Data  
**Description**: Prepares image datasets for training and validation with configurable augmentations and splits.

#### Key Settings (from `settings.py`)
| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `pth_train` | Path | Training image directory | `"./data/train"` |
| `ds_shuffle` | bool | Shuffle before validation split | `True` |
| `ds_shuffle_seed` | int | Random seed for reproducibility | `42` |
| `ds_batch_size` | int | Batch size for training | `32` |
| `ds_val_split` | float | Validation set fraction (0-1) | `0.2` |
| `train_use_augment` | bool | Enable/disable augmentations | `True` |

#### Required Folder Structure
```plaintext
data/
└── train/
    ├── WT/       # Wild-type images
    └── KO/       # Knockout images
```

#### Augmentation Pipeline (when enabled)
| Transformation | Purpose | Parameters |
|----------------|---------|------------|
| 90° Rotations | Simulate microscope orientation | 0°, 90°, 180°, 270° |
| Color Jitter | Adjust brightness/contrast | ±20% variation |
| Gaussian Blur | Simulate focus shifts | Kernel=3, σ=0.1-0.5 |
| Normalization | Standardize pixel values | Grayscale/RGB-specific |

#### Example Workflow
1. Configure `settings.py`:
   ```python
   pth_train = "./data/train"
   ds_val_split = 0.2
   train_use_augment = True
   ```
2. Run program and select Option 3:
   ```plaintext
   Loading training dataset with augmentations...
   Dataset split:
   - Training images: 800 (25 batches)
   - Validation images: 200 (7 batches)
   ```

#### Technical Notes
- Uses PyTorch's `ImageFolder` for automatic class labeling
- Validation split preserves class ratios
- Grayscale/RGB handled automatically via `img_channels` setting

#### Error Handling
- ❌ Fails if:
  - Invalid image channels (not 1 or 3)
  - Missing `pth_train` directory
  - Class folders don't match `settings.py` classes

## 4. Train Network  
**Description**: Executes model training with configurable optimization, learning rate scheduling, and mixed-precision acceleration.

#### Key Settings (from `settings.py`)
| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `train_num_epochs` | int | Total training epochs | `50` |
| `train_init_lr` | float | Initial learning rate | `0.01` |
| `train_lr_warmup_epochs` | int | Linear warmup duration | `5` |
| `train_lr_eta_min` | float | Minimum LR for cosine annealing | `0.0001` |
| `train_weight_decay` | float | L2 regularization strength | `0.001` |
| `train_momentum` | float | SGD momentum value | `0.9` |
| `chckpt_min_acc` | float | Minimum accuracy to save checkpoints | `0.7` |

#### Prerequisites
- ✅ Model loaded (Option 1)
- ✅ Training data loaded (Option 3)

#### Training Pipeline
```plaintext
1. Warmup Phase (Linear LR)
   └── 5 epochs @ 1% → 100% of target LR

2. Main Training (Cosine Annealing)
   └── 45 epochs with LR decay to η_min

3. Mixed Precision
   └── Automatic FP16/FP32 selection
```

#### Output Metrics
| Metric | Description | Visualization |
|--------|-------------|---------------|
| Training Loss | CrossEntropyLoss | ![Loss Curve](https://i.imgur.com/XYzKv1l.png) |
| Validation Accuracy | Per-class balanced accuracy | Green/red curves |
| Learning Rate | Scheduled values | Blue curve |

#### Example Workflow
1. Configure `settings.py`:
   ```python
   train_num_epochs = 50
   train_init_lr = 0.01
   chckpt_min_acc = 0.75  # Only save models >75% accuracy
   ```
2. Run training:
   ```plaintext
   >> Epoch [1/50]:
   Train: 100%|████| 25/25 [00:10<00:00]
   > train_loss: 1.892, train_acc: 0.41, lr: 0.0001
   Valid: 100%|████| 7/7 [00:02<00:00]
   > val_loss: 1.532, val_acc: 0.58
   Model with test accuracy 0.58 saved!
   ```

#### Technical Highlights
- **Mixed Precision**: Automatic GradScaler for FP16/FP32 safety
- **LR Scheduling**: Warmup + cosine annealing
- **Checkpointing**: Saves best model when `val_acc > chckpt_min_acc`
- **Metrics Plot**: Auto-saves to `plot_pth/train_metrics.png`

#### Error Handling
- ❌ Fails if:
  - No model/dataset loaded
  - CUDA OOM (reduces batch size)
  - Invalid LR scheduler config

## 5. Load Weights  
**Description**: Loads pre-trained model weights from checkpoint files for inference or continued training.

#### Key Settings (from `settings.py`)
| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `pth_checkpoint` | Path | Checkpoint directory | `"./checkpoints"` |
| `chckpt_min_acc` | float | Minimum accuracy filter (when saving) | `0.75` |

#### Required Folder Structure
```plaintext
checkpoints/
├── ckpt_densenet121_e50_vacc92.model
├── ckpt_resnet18_e30_vacc85.model
└── ckpt_custom_e10_vacc78.model
```

#### Workflow
1. **Automatic Detection**:
   ```plaintext
   Found 3 checkpoints:
   +----+----------------------------------+
   | ID | Checkpoint Name                  |
   +----+----------------------------------+
   | 1  | ckpt_densenet121_e50_vacc92.model|
   | 2  | ckpt_resnet18_e30_vacc85.model   |
   | 3  | ckpt_custom_e10_vacc78.model     |
   +----+----------------------------------+
   ```
2. **Interactive Selection**:
   ```plaintext
   Select a checkpoint: 1
   Loading weights from ckpt_densenet121_e50_vacc92.model...
   Successfully loaded weights (92% val acc)
   ```

#### Technical Notes
- **File Format**: Uses PyTorch `.state_dict()` format
- **Compatibility Check**:
  - Matches architecture to current model
  - Verifies class count matches
- **Metadata**: Preserves epoch count and validation accuracy in filename

#### Error Handling
- ❌ Fails if:
  - No checkpoints exist
  - Architecture mismatch
  - File corruption (invalid `.model` file)
  - CUDA device mismatch (GPU vs CPU)

#### Example Usage
1. Configure `settings.py`:
   ```python
   pth_checkpoint = "./checkpoints"
   ```
2. Run program:
   ```plaintext
   >> Loading weights:
   Found single checkpoint: ckpt_densenet121_e10_vacc85.model
   Loading automatically...
   Weights loaded (85% val accuracy)
   ```

## 6. Predict Class from Predict Folder  
**Description**: Classifies images in subfolders using a trained model and generates CSV reports with class distributions.

#### Key Settings (from `settings.py`)
| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `pth_prediction` | Path | Root folder containing subfolders to analyze | `"./data/predict"` |
| `pth_checkpoint` | Path | Directory containing saved models | `"./checkpoints"` |
| `classes` | list | Class names matching training | `["WT", "KO"]` |
| `img_channels` | int | Input channels (1=grayscale, 3=RGB) | `3` |

#### Required Folder Structure
```plaintext
data/
└── predict/                  # Set in pth_prediction
    ├── experiment_1/         # Folder 1 (any name)
    │   ├── img001.png        # Supports .png/.jpg/.jpeg
    │   └── img002.png
    ├── experiment_2/         # Folder 2
    │   └── img003.png
    └── results/              # Auto-created for reports
```

#### Workflow
1. **Model Loading**:
   ```plaintext
   Found 3 checkpoints:
   +----+----------------------------------+
   | ID | Checkpoint Name                  |
   +----+----------------------------------+
   | 1  | ckpt_densenet121_e50_vacc92.model|
   Select a checkpoint: 1
   Successfully loaded weights (92% val acc)
   ```

2. **Prediction**:
   ```plaintext
   > PROCESSING FOLDER: experiment_1
   Predicting experiment_1: 100%|████████| 2/2 [00:01<00:00]
   > RESULTS:
   Total images processed: 2
   Class distribution:
   - WT: 1 images (50.00%)
   - KO: 1 images (50.00%)
   ```

3. **Report Generation**:
   ```plaintext
   Analysis complete. Results saved to: 
   ./data/predict/results/result_ckpt_densenet121_e50_vacc92.csv
   ```

#### Output CSV Format
| Folder | Most Likely Class | Total Images | WT_Count | KO_Count | WT_Percentage | KO_Percentage |
|--------|-------------------|--------------|----------|----------|---------------|---------------|
| experiment_1 | WT | 2 | 1 | 1 | 50.0 | 50.0 |

#### Technical Highlights
- **Batch Processing**: Handles folders with arbitrary numbers of images
- **Progress Tracking**: Real-time counts with tqdm
- **Flexible Input**: Supports mixed image formats (PNG/JPG/JPEG)
- **Device-Agnostic**: Works on CPU/GPU automatically

#### Error Handling
- ❌ Fails if:
  - No subfolders exist in `pth_prediction`
  - Image dimensions mismatch model
  - Invalid checkpoint selection
  - Unsupported image formats

#### Example Usage
1. Prepare folder structure:
   ```bash
   mkdir -p data/predict/experiment_{1..3}
   ```
2. Run prediction:
   ```python
   analyzer = ClassAnalyzer(device='cuda')
   analyzer.analyze_prediction_folder()
   ```

## 7. Dataset Generator  
**Description**: Automates dataset creation for cross-validation by splitting cell line images into train/test sets.

#### Key Settings (from `settings.py`)
| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `pth_ds_gen_input` | Path | Source images directory | `"./data/raw_images"` |
| `pth_ds_gen_output` | Path | Output directory (gen mode) | `"./data/generated"` |
| `wt_lines` | list | Wild-type cell line names | `["WT1", "WT2"]` |
| `ko_lines` | list | Knockout cell line names | `["KO1", "KO2"]` |
| `pth_acv_results` | Path | ACV output directory | `"./results/acv"` |

#### Folder Structure (Input)
```plaintext
data/
└── raw_images/          # Source images (set in pth_ds_gen_input)
    ├── WT1/             # Wild-type line 1
    ├── WT2/             # Wild-type line 2  
    ├── KO1/             # Knockout line 1
    └── KO2/             # Knockout line 2
```

#### Generation Modes
| Mode | Output Location | Purpose |
|------|-----------------|---------|
| `gen` | `pth_ds_gen_output` | Manual dataset creation |
| `acv` | `pth_acv_results` | Automatic cross-validation prep |

#### Workflow Example (ACV Mode)
1. **Configuration**:
   ```python
   mode = "acv"
   wt_lines = ["WT1", "WT2"]
   ko_lines = ["KO1", "KO2"]
   ```

2. **Dataset Creation**:
   ```plaintext
   Generating dataset 1/4:
   - Test WT: WT1
   - Test KO: KO1
   Copied 120 training images (80 WT, 40 KO)
   Copied 30 test images (15 WT, 15 KO)
   ```

3. **Output Structure** (per dataset):
   ```plaintext
   results/acv/dataset_1/
   ├── train/
   │   ├── WT/       # All WT except WT1
   │   └── KO/       # All KO except KO1
   ├── test/
   │   ├── WT/       # Only WT1 images
   │   └── KO/       # Only KO1 images
   └── dataset_1_info.txt  # Metadata
   ```

#### Metadata File Contents
```plaintext
Dataset 1 Configuration:
=== TRAINING DATA ===
WT lines:
- WT2: 80 images
Total WT training images: 80

KO lines:
- KO2: 40 images
Total KO training images: 40

=== TESTING DATA ===
WT test line: WT1 (15 images)
KO test line: KO1 (15 images)
```

#### Technical Notes
- **Leave-One-Out Strategy**: Each dataset excludes one WT/KO pair as test set
- **Reproducibility**: Maintains consistent train/test splits via fixed indexing
- **Parallelization**: Supports incremental generation (unlike `generate_all_datasets`)

#### Error Handling
- ❌ Fails if:
  - Missing input cell line folders
  - Image filename collisions
  - Invalid mode selection

## 8. Automatic Cross Validation (ACV)  
**Description**: Performs end-to-end cross-validation by training and testing on all WT/KO cell line combinations.

#### Key Settings (from `settings.py`)
| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `wt_lines` | list | Wild-type cell lines | `["WT1", "WT2"]` |
| `ko_lines` | list | Knockout cell lines | `["KO1", "KO2"]` |
| `pth_acv_results` | Path | Output directory | `"./results/acv"` |
| `classes` | list | Class names | `["WT", "KO"]` |

#### Workflow
```plaintext
1. Dataset Generation
   ├── Creates N datasets (N = wt_lines × ko_lines)
   ├── Each dataset holds out one WT/KO pair as test set

2. Training Phase
   ├── Trains model on remaining cell lines
   ├── Saves checkpoints when val_acc > threshold

3. Testing Phase
   ├── Evaluates on held-out cell lines
   ├── Generates confusion matrices and accuracy reports
```

#### Folder Structure (Output)
```plaintext
results/acv/
├── dataset_1/
│   ├── checkpoints/
│   │   └── ckpt_e10_vacc85.model
│   ├── plots/
│   │   ├── confusion_matrix.png
│   │   └── train_metrics.png
│   └── dataset_1_info.txt
├── dataset_2/
...
```

#### Example Output
```plaintext
>> PROCESSING DATASET 1/4:
Test WT: WT1 | Test KO: KO1
Created dataset with:
- 120 training images (80 WT, 40 KO)
- 30 test images (15 WT, 15 KO)

Training: 100%|████████| 50/50 epochs
Validation accuracy: 85.2%
Saved checkpoint: ckpt_e50_vacc85.model

Testing:
Overall accuracy: 83.3% 
WT accuracy: 86.7% 
KO accuracy: 80.0%
```

#### Key Features
- **Leave-One-Out Validation**: Tests generalizability across cell lines
- **Automatic Reporting**: Saves per-dataset:
  - Training curves (loss/accuracy)
  - Confusion matrices
  - Checkpoint metadata
- **Reproducible Splits**: Fixed random seeds ensure consistency

#### Error Handling
- ❌ Fails if:
  - Missing input cell line folders
  - CUDA out of memory (reduce batch size)
  - Invalid model architecture

## 9. Confidence Analyzer (ACV-based)  
**Description**: Identifies systematically reliable/misclassified images across all cross-validation folds using prediction confidence thresholds.

#### Key Settings (from `settings.py`)
| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `ca_min_conf` | float | Minimum confidence threshold | `0.8` |
| `ca_max_conf` | float | Maximum confidence threshold | `1.0` |
| `ca_filter_type` | str | Filter mode (`correct`/`incorrect`/`low_confidence`/`unsure`) | `"correct"` |
| `ca_max_ckpts` | int | Max checkpoints per dataset | `3` |
| `ca_ckpt_select_method` | str | Checkpoint selection metric (`balanced_sum`/`f1_score`/`min_difference`) | `"balanced_sum"` |

#### Filter Types
| Type | Confidence Range | Correctness Requirement | Output Folder | Use Case |
|------|------------------|-------------------------|---------------|----------|
| `correct` | [min, max] | Must be correct in all folds | `high_confidence_correct` | Reliable predictions |
| `incorrect` | [min, max] | Must be wrong in all folds | `high_confidence_incorrect` | Systematic errors |
| `low_confidence` | < min | Ignored | `low_confidence` | Ambiguous cases |
| `unsure` | [min, max] | Ignored | `medium_confidence_unsure` | Borderline predictions |

#### Workflow
```plaintext
1. For each ACV dataset:
   ├── Load top checkpoints (selected by metric)
   ├── Predict on test images with confidence scores
   └── Track predictions across all folds

2. Apply filters:
   ├── correct: Images always right AND confident
   ├── incorrect: Images always wrong BUT confident
   └── low_confidence: Uncertain predictions

3. Organize results:
   ├── Copies filtered images to categorized folders
   └── Generates CSV reports
```

#### Output Structure
```plaintext
results/confidence_analysis/
├── high_confidence_correct/
│   ├── WT/
│   │   ├── img1_conf98_corr100.png
│   │   └── img2_conf95_corr100.png
│   └── KO/
│       └── img3_conf92_corr100.png
├── used_checkpoints.csv
└── confidence_analysis.csv
```

#### Example: Finding High-Confidence Correct
```python
# settings.py
ca_min_conf = 0.9  # 90% min confidence
ca_max_conf = 1.0
ca_filter_type = "correct"
```

```plaintext
>> PROCESSING DATASET 1/4:
Found 32 images meeting criteria:
- WT: 18 images (avg confidence: 94.2%)
- KO: 14 images (avg confidence: 93.7%)
```

#### Technical Highlights
- **Cross-Validation Consistency**: Requires unanimous correctness/errors across folds
- **Smart Checkpoint Selection**: Optimizes for balanced accuracy
- **Metadata-Rich Filenames**: Embeds confidence/correctness rates
- **Reproducible**: Uses original image paths from `pth_ds_gen_input`

#### Interpretation Guide
| Confidence | Meaning |
|------------|---------|
| 50% | Random guessing |
| 50-80% | Weak prediction |
| 80-95% | Confident |
| 95-100% | Very confident |

#### Error Handling
- ❌ Fails if:
  - Missing ACV results folders
  - Confidence thresholds invalid (min > max)
  - No images match filter criteria

### 10. Captum Analyzer  
**Description**: Visualizes model decision-making using Integrated Gradients to highlight biologically relevant image regions.

#### Key Settings (from `settings.py`)
| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `captum_n_steps_ig` | int | Integration steps for gradient calculation | `50` |
| `captum_sign` | str | Attribution sign (`all`/`positive`/`negative`) | `"positive"` |
| `captum_sigma` | float | Gaussian blur radius (pixels) | `2.0` |
| `captum_alpha_overlay` | float | Heatmap opacity (0-1) | `0.7` |
| `captum_threshold_percentile` | int | Min percentile for feature importance | `90` |
| `captum_dpi` | int | Output image resolution | `300` |

#### Visualization Modes
| Component | Setting | Options |
|-----------|---------|---------|
| Original Image | `captum_cmap_orig` | `"gray"`, `"viridis"` |
| Heatmap | `captum_cmap_heatmap` | `"coolwarm"`, `"jet"` |
| Overlay | `captum_cmap_overlay` | `"red"`, `"blue"` |

#### Required Folder Structure
```plaintext
data/
└── predict/                  # Input images (set in pth_prediction)
    ├── experiment_1/         # Original images
    └── experiment_1_captum/  # Auto-created output folder
```

#### Example Workflow
1. Configure analysis:
   ```python
   # settings.py
   captum_sign = "positive"  # Highlight activating features
   captum_sigma = 1.5        # Moderate smoothing
   ```

2. Run analysis:
   ```plaintext
   >> Loading weights:
   Found checkpoint: ckpt_densenet121_e50_vacc92.model
   Generating attribution maps: 100%|████████| 15/15 [00:32<00:00]
   Saved 15 visualization(s) to ./data/predict/experiment_1_captum/
   ```

#### Technical Highlights
- **Integrated Gradients**: Quantifies pixel-level importance
- **Adaptive Thresholding**: Focuses on top 10% significant features
- **Multi-Channel Support**: Handles both grayscale and RGB
- **Memory Efficient**: Automatic CUDA cache clearing

#### Biological Interpretation
| Heatmap Color | Meaning (with `sign="positive"`) |
|---------------|-----------------------------------|
| Red | Strong positive evidence for class |
| Blue | Neutral/no evidence |
| White | Negative evidence (if `sign="all"`) |

#### Error Handling
- ❌ Fails if:
  - No prediction folder exists
  - Checkpoint incompatible with model
  - Image dimensions mismatch
  - CUDA OOM (reduce batch size)

### 11. GradCAM Analyzer (DenseNet-121)  
**Description**: Visualizes class-specific activation patterns using gradient-weighted class activation mapping.

#### Key Settings (from `settings.py`)
| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `gradcam_second_iteration` | bool | Enable diagnostic blurring mode | `True` |
| `gradcam_threshold_percent` | float | Top features to blur (0-1) | `0.2` |
| `gradcam_blurr_sigma` | float | Gaussian blur strength | `5.0` |
| `captum_alpha_overlay` | float | Heatmap transparency | `0.4` |

#### Visualization Workflow
```plaintext
1. First Pass:
   ├── Generates standard Grad-CAM heatmap
   └── Identifies most salient regions

2. Second Pass (if enabled):
   ├── Blurs top 20% salient features
   ├── Re-runs Grad-CAM
   └── Highlights compensatory features
```

#### Output Structure
```plaintext
data/predict/
└── experiment_1_gradcam/
    ├── image1_gradcam.png       # First iteration
    └── image1_gradcam_iter2.png # Second iteration (blurred)
```

#### Interpretation Guide
| Feature | Biological Relevance |
|---------|-----------------------|
| Red Regions | Strongest evidence for predicted class |
| Yellow Regions | Secondary supporting features |
| Dark Blue | Non-contributing regions |

#### Diagnostic Mode (Second Iteration)
When enabled:
1. Blurs primary salient features
2. Reveals model's:
   - **Robustness**: If predictions change significantly
   - **Secondary Features**: Alternative decision pathways

#### Technical Notes
- **DenseNet-Specific**: Optimized for DenseNet-121 architecture
- **Memory Efficient**: Automatic CUDA cache management
- **Multi-Resolution**: Maintains detail in high-res images

#### Error Handling
- ❌ Fails if:
  - Non-DenseNet model loaded
  - Image dimensions mismatch
  - CUDA OOM (reduce batch size)
  - Invalid blur threshold (>1 or <0)

#### Example Usage
```python
# settings.py
gradcam_second_iteration = True  # Enable diagnostic mode
gradcam_threshold_percent = 0.3  # Blur top 30% features
```

```plaintext
>> Running Grad-CAM:
Processing image1.png [1/2]...
Blurring 30% of salient features...
Processing image1.png [2/2]...
Saved visualizations to ./data/predict/experiment_1_gradcam/
```

## 12. Exit Program  
**Description**: Closes program.