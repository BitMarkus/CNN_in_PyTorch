# CNN_in_PyTorch
A CNN (convolutional neural network, different architectures) for classification and analysis of microscopic images (in python, pytorch)

---

## 1. Single CNN Training

**Description**: The Single CNN Training module provides the core workflow for building, training, and saving a CNN classifier on a single dataset (no cross-validation). It handles model creation, dataset loading with augmentations, training with learning rate scheduling and mixed precision, and checkpoint management.

The module contains five sequential actions:

| # | Action | Purpose |
|---|--------|---------|
| 1 | Create CNN Network | Instantiate the chosen architecture |
| 2 | Show Network Summary | Inspect layer shapes and parameter counts |
| 3 | Load Training Data | Load train/val images with augmentations |
| 4 | Train Network | Run the training loop and save checkpoints |
| 5 | Load Weights | Load a trained checkpoint for inference or continued training |

---

### 1.1 Create CNN Network

**Description**: Initializes a new convolutional neural network based on the architecture defined in `settings.py`. Supports a wide range of torchvision architectures as well as a fully customizable CNN.

#### Key Settings (from `settings.py`)

| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `cnn_type` | str | Architecture to load (see below) | `"densenet121"` |
| `cnn_is_pretrained` | bool | Use ImageNet-pretrained weights | `True` |
| `cnn_initialization` | str | Weight init method for non-pretrained models (`"kaiming"` / `"xavier"`) | `"kaiming"` |
| `img_channels` | int | Input channels (1 = grayscale, 3 = RGB) | `1` |
| `img_width` / `img_height` | int | Input dimensions | `512` |
| `classes` | list | Class names | `["KO", "WT"]` |

#### Supported Architectures

| Family | Models |
|--------|--------|
| ResNet | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| ResNeXt | `resnext101_32x8d`, `resnext101_64x4d` |
| AlexNet | `alexnet` |
| VGG | `vgg11`, `vgg13`, `vgg16`, `vgg19` (also `_bn` variants) |
| DenseNet | `densenet121`, `densenet169`, `densenet201` |
| EfficientNet | `efficientnet_b0`, `efficientnet_b3`, `efficientnet_b4`, `efficientnet_b7` |
| ConvNeXt | `convnext_tiny`, `convnext_small` |
| Custom | `custom` (see below) |

#### Using a Custom CNN Architecture

Set `cnn_type = "custom"` in `settings.py` to use your own network. The custom architecture is implemented in:

```plaintext
single_training/custom_cnn.py
```

To create your own architecture:

1. Open `single_training/custom_cnn.py`
2. Modify the `CustomCNN` class to change the layer structure
3. The class must:
   - Accept `input_channels`, `num_classes`, `batch_size`, `img_size`, and `dropout` as constructor arguments
   - Return logits of shape `(batch_size, num_classes)` from `forward()`
4. Adjust these settings in `settings.py`:

| Setting | Description |
|---------|-------------|
| `cnn_type` | Must be `"custom"` |
| `img_channels` | Number of input channels (must match first conv layer) |
| `img_width` / `img_height` | Input dimensions (must match architecture) |
| `cnn_dropout` | Dropout rate used in the custom architecture |
| `classes` | Number of output classes |

Pretrained weights are not available for `custom`, so `cnn_is_pretrained` is ignored.

#### Example Workflow

1. Configure `settings.py`:
   ```python
   cnn_type = "densenet121"
   cnn_is_pretrained = True
   classes = ["KO", "WT"]
   img_channels = 1
   ```

2. Run the program and select **1 → 1**:
   ```plaintext
   :NEW CNN NETWORK:
   Number of classes: 2
   Classes: KO, WT
   Creating new densenet121 network...
   New network was successfully created.
   Trainable parameters: 6,965,826
   ```

---

### 1.2 Show Network Summary

**Description**: Displays a detailed summary of the current model architecture, including layer output shapes, parameter counts, and trainability. Uses forward hooks to capture actual tensor shapes from a real forward pass.

#### Example Output

```plaintext
+---------------------+-------------------+------------+-----------+
| Layer (type)        | Output Shape      | Param #    | Trainable |
+---------------------+-------------------+------------+-----------+
| Conv2d              | [50, 64, 256, 256]| 640        | True      |
| BatchNorm2d         | [50, 64, 256, 256]| 128        | True      |
| ...
| Linear              | [50, 2]           | 2,050      | True      |
+---------------------+-------------------+------------+-----------+

Input: [50, 1, 512, 512] (Device: cuda:0)
Total params: 6,965,826
Trainable params: 6,965,826
Non-trainable params: 0
```

---

### 1.3 Load Training Data

**Description**: Loads training and validation images from the `data/` folders. Validation images are obtained by splitting **either** the training set **or** the test set — there is **no dedicated validation folder**. The choice is controlled by two settings in `settings.py`.

During training, only the training set is used for gradient updates, while the validation set is used to monitor performance after each epoch. After training completes, the test set (which may have been partially used for validation) is evaluated separately.

#### Key Settings (from `settings.py`)

| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `ds_val_from_train_split` | float / False | Fraction of `data/train/` used for validation | `False` |
| `ds_val_from_test_split` | float / False | Fraction of `data/test/` used for validation | `1.0` |
| `ds_batch_size` | int | Batch size for training and validation | `50` |
| `ds_shuffle` | bool | Shuffle before validation split | `True` |
| `ds_shuffle_seed` | int | Random seed for reproducibility | `44` |
| `ds_num_workers` | int | Parallel data-loading subprocesses | `3` |
| `train_use_augment` | bool | Enable augmentations during training | `True` |
| `ds_save_val_images` | bool | Export validation images to a folder | `False` |

#### Validation Split Modes

Exactly one of the two split settings should be active (not `False`) at a time:

| Mode | `ds_val_from_train_split` | `ds_val_from_test_split` | Description |
|------|---------------------------|--------------------------|-------------|
| **Split from train** | `0.0 – 1.0` | `False` | Validation images are taken from `data/train/`; test set is untouched |
| **Split from test** | `False` | `0.0 – 1.0` | Training uses all of `data/train/`; validation and test images come from `data/test/` |

If both are set, the training split takes precedence and a warning is printed. If neither is set, a default of `0.1` from train is used.

#### Required Folder Structure

```plaintext
data/
├── train/               # Training images (set in `pth_train`)
│   ├── KO/              # Knockout class folder
│   └── WT/              # Wild-type class folder
└── test/                # Test images (set in `pth_test`)
    ├── KO/
    └── WT/
```

The number of classes and their folder names must match the `classes` list in `settings.py`.

#### Augmentation Pipeline (when enabled)

| Transformation | Purpose | Parameters |
|----------------|---------|------------|
| Horizontal / Vertical Flip | Simulate microscope orientation | p = 0.5 |
| 90° Rotations | All four cardinal orientations | p = 0.5 |
| Small-angle Rotation | Minor deviations | ±10°, fill = 100 |
| Color Jitter | Adjust brightness/contrast | ±20% variation |
| Gamma Correction | Simulate exposure variation | γ ∈ [0.7, 1.3] |
| Gaussian Blur | Simulate focus shifts | kernel = 5, σ ∈ [0.1, 0.5] |
| Poisson Noise | Simulate low-light noise | scaling = 0.05, strength = 0.1 |
| Normalization | Standardize pixel values | Mean = 0.5, Std = 0.5 (grayscale) |

#### Example Workflow

1. Configure `settings.py`:
   ```python
   ds_val_from_train_split = False
   ds_val_from_test_split = 1.0
   ds_batch_size = 50
   train_use_augment = True
   ```

2. Run the program and select **1 → 3**:
   ```plaintext
   :LOAD TRAINING DATA:
   Validation strategy: Using test set images for validation (100.0%)
   Training and validation datasets successfully loaded.
   Number training images/batches: 8750/175
   Number validation images/batches: 3100/62
   ```

---

### 1.4 Train Network

**Description**: Runs the training loop with the configured optimizer, learning rate scheduler, loss function, and mixed-precision acceleration. After each epoch, validation is performed and checkpoints are saved based on the selected checkpoint criterion.

#### Key Settings (from `settings.py`)

| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `train_num_epochs` | int | Total training epochs | `40` |
| `train_optimizer_type` | str | Optimizer (`"SGD"`, `"ADAM"`, `"ADAMW"`) | `"ADAMW"` |
| `train_init_lr` | float | Initial learning rate | `1e-4` |
| `train_weight_decay` | float | L2 regularization strength | `1e-3` |
| `train_lr_warmup_epochs` | int | Linear warmup duration | `5` |
| `train_lr_eta_min` | float | Minimum LR for cosine annealing | `1e-5` |
| `train_use_weighted_loss` | bool | Weight loss by inverse class frequency | `True` |
| `train_label_smoothing` | float | Label smoothing factor | `0.1` |

#### Checkpoint Selection Settings

| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `chckpt_save` | bool | Enable checkpoint saving | `True` |
| `chckpt_selection_method` | str | `"balanced_accuracy"`, `"composite_score"`, or `"both"` | `"balanced_accuracy"` |
| `chckpt_min_balanced_acc_threshold` | float | Minimum balanced accuracy to save | `0.65` |
| `chckpt_min_per_class_acc_balanced` | float | Minimum per-class accuracy | `0.60` |
| `chckpt_min_class_acc_threshold` | float | Minimum per-class accuracy for composite score | `0.65` |
| `chckpt_penalty_weight` | float | Penalty for class imbalance (1.0–4.0) | `2.0` |

**Checkpoint selection logic:**

- **`balanced_accuracy`** (default): Saves a checkpoint whenever balanced accuracy improves and both minimum thresholds are met. Best for most use cases.
- **`composite_score`**: Uses `overall_accuracy − penalty_weight × std(class_accuracies)`. More stringent against class imbalance.
- **`both`**: Triggers a save whenever *either* metric improves.

#### Training Pipeline

```plaintext
1. Warmup Phase (Linear LR)
   └── 5 epochs @ 1% → 100% of target LR

2. Main Training (Cosine Annealing)
   └── remaining epochs with LR decay to eta_min

3. Mixed Precision
   └── Automatic FP16/FP32 selection via GradScaler

4. Gradient Clipping
   └── max_norm = 1.0
```

#### Output

After training, results are stored under:

```plaintext
output/train/[timestamp]/
├── checkpoints/             # Saved .pt files
├── plots/                   # Metrics, confusion matrices, ROC/PR data
├── logs/                    # TensorBoard event files
│   └── probabilities/       # Per-epoch probabilities (.npz + .json)
├── settings_copy.py         # Snapshot of settings.py
└── training_examples.png    # Image grid from the training set
```

#### TensorBoard

To visualize training progress, launch TensorBoard from the project root:

```bash
python -m tensorboard.main --logdir "output/train/[timestamp]/logs"
```

Then open the printed URL in your browser (usually `http://localhost:6006`). TensorBoard displays loss curves, accuracy, per-class metrics, ROC curves, and confusion matrices.

#### Checkpoint Naming Convention

```plaintext
ckpt_{pretrained}_{arch}_e{epoch}_bal{balanced_acc}_comp{composite_score}{dataset_suffix}.pt
```

| Part | Meaning | Example |
|------|---------|---------|
| `pretr` / `scratch` | Pretrained or from-scratch weights | `pretr` |
| `{arch}` | Architecture name | `densenet121` |
| `e{epoch}` | Epoch (zero-padded) | `e23` |
| `bal{acc}` | Balanced accuracy | `bal0.860` |
| `comp{score}` | Composite score | `comp0.812` |
| `{dataset_suffix}` | Cross-validation suffix (`_dsXX`), omitted for single training | *(empty)* |

**Example:** `ckpt_pretr_densenet121_e23_bal0.860_comp0.812.pt`

#### Example Workflow

1. Configure `settings.py`:
   ```python
   train_num_epochs = 40
   chckpt_selection_method = "balanced_accuracy"
   chckpt_min_balanced_acc_threshold = 0.65
   chckpt_min_per_class_acc_balanced = 0.60
   ```

2. Run the program and select **1 → 4**:
   ```plaintext
   :TRAIN NETWORK:
     Results will be saved to output/train/[timestamp]/
   Start training...

   >> Epoch [1/40]:
   Train: 100%|████████| 175/175 [02:15<00:00]
   > Train Loss: 0.583 | Weighted Train Acc: 0.712 | Learning Rate: 0.000020
   Valid: 100%|████████| 62/62 [00:23<00:00]
   > Val Loss: 0.412 | Weighted Val Acc: 0.781 | Per-class: (0.79 KO | 0.77 WT)
   > Balanced Accuracy: 0.781 | Composite Score: 0.764
   > Class STD: 0.008 | Min Class Acc: 77.3%
   ✓ Model saved! Epoch 1
   ```

---

### 1.5 Load Weights

**Description**: Loads a previously saved checkpoint for inference or continued training. If the `checkpoints/` folder contains a single file, it is loaded automatically; otherwise, an interactive table allows selection.

#### Key Settings (from `settings.py`)

| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `pth_checkpoint` | Path | Directory containing saved checkpoints | `BASE_DIR / "checkpoints/"` |

#### Required Folder Structure

```plaintext
checkpoints/
├── ckpt_pretr_densenet121_e23_bal0.860_comp0.812.pt
├── ckpt_pretr_densenet121_e18_bal0.842_comp0.798.pt
└── ckpt_scratch_resnet50_e11_bal0.771_comp0.743.pt
```

#### Example Workflow

1. Place one or more `.pt` files in `checkpoints/`
2. Run the program and select **1 → 5**:
   ```plaintext
   :LOAD WEIGHTS:
   Found single checkpoint: ckpt_pretr_densenet121_e23_bal0.860_comp0.812.pt
   Loading automatically...
   Weights from checkpoint ckpt_pretr_densenet121_e23_bal0.860_comp0.812.pt successfully loaded.
   ```

   With multiple checkpoints:
   ```plaintext
   +----+---------------------------------------------------------------+
   | ID | Checkpoint                                                    |
   +----+---------------------------------------------------------------+
   | 1  | ckpt_pretr_densenet121_e23_bal0.860_comp0.812.pt              |
   | 2  | ckpt_pretr_densenet121_e18_bal0.842_comp0.798.pt              |
   | 3  | ckpt_scratch_resnet50_e11_bal0.771_comp0.743.pt               |
   +----+---------------------------------------------------------------+
   Select a checkpoint: 1
   Weights from checkpoint ckpt_pretr_densenet121_e23_bal0.860_comp0.812.pt successfully loaded.
   ```

---

## 2. Cross Validation

**Description**: The Cross Validation module automates leave-one-cell-line-out validation, ensuring that the CNN learns disease-associated morphological features rather than cell-line-specific artifacts. It handles dataset generation, iterative training across all WT/KO cell-line combinations, automatic test evaluation on held-out cell lines, and identification of consistently reliable images across all folds.

#### Real vs. Synthetic Images

This module distinguishes between two types of images:

| Type | Source | Purpose |
|------|--------|---------|
| **Real images** | Primary fibroblasts acquired via DIC microscopy | Ground truth for evaluation; used for both training and validation/testing |
| **Synthetic images** | Generated by a FLUX.1 diffusion model fine-tuned with LoRA on real images | Training data only; never used for validation or testing |

Synthetic images are a **data augmentation strategy for training only**. Validation and test sets must consist exclusively of real images, because the goal is to measure how well the model generalizes to real cells — the actual target of classification. For this reason, the Confidence Analyzer explicitly filters synthetic images out of all downstream analyses.

The recommended configurations are therefore:

- **`real_only`**: Train, validate, and test all on real images (baseline, always valid)
- **`synthetic_only`**: Train on synthetic images, validate and test on real images (recommended when synthetic data is available)

The `mixed` option allows training on both real and synthetic images in a single pool. When this mode is used, the script automatically excludes synthetic images from validation and test splits — only real images are recorded in `split_info.json` (see section 2.2) and used for evaluation. This ensures that performance is always measured on real cells, regardless of which training mode was chosen. The user should still be aware that this filtering happens silently, and that the effective real-image pool for validation and testing is smaller than the raw contents of the test folder.

The module contains three sequential actions:

| # | Action | Purpose |
|---|--------|---------|
| 1 | Dataset Generator | Prepare all train/test dataset combinations |
| 2 | Automatic Cross Validation | Train and test on every WT/KO combination |
| 3 | Confidence Analyzer | Identify reliably classified images across all folds |

---

### 2.1 Dataset Generator

**Description**: Creates all possible leave-one-cell-line-out dataset combinations. For each combination, one WT and one KO cell line are held out for testing while the remaining cell lines are used for training. This generates `N_WT × N_KO` independent datasets (e.g., 5 WT × 4 KO = 20 datasets). Each dataset contains a `train/` and `test/` folder with the WT/KO class subdirectories.

**Note on workflow**: The Dataset Generator is **not required** for running cross-validation. When you start Automatic Cross Validation (section 2.2), datasets are generated on the fly for each fold and cleaned up afterwards. This standalone generator is meant for cases where you want to **inspect, modify, or reuse a specific dataset combination** — for example, to experiment on a particular WT/KO pair, to compare against a different model, or to reproduce a previous result. It writes the datasets to `dataset_gen/output/` so they persist across sessions, unlike the automatic version which produces temporary datasets inside `output/cross_validation/`.

The dataset generator supports three training data source modes, controlled by `train_data_source`. In all modes, validation and test images are drawn from **real images only** — synthetic images are used exclusively for training.

| Mode | Training data | Validation / test data | Recommended |
|------|--------------|------------------------|-------------|
| `real_only` | Real images | Real images | ✅ Baseline |
| `synthetic_only` | Synthetic images | Real images | ✅ For synthetic-augmented training |
| `mixed` | Real + synthetic images (pooled) | Real images | ⚠️ Not recommended (see note above) |

#### Key Settings (from `settings.py`)

| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `train_data_source` | str | `"mixed"`, `"synthetic_only"`, or `"real_only"` | `"real_only"` |
| `wt_lines` | list | Wild-type cell line folder names | `["WT_1618-02", "WT_JG", "WT_JT", "WT_KM", "WT_MS"]` |
| `ko_lines` | list | Knockout cell line folder names | `["KO_1096-01", "KO_1618-01", "KO_BR2986", "KO_BR3075"]` |
| `classes` | list | Class names matching folder structure | `["KO", "WT"]` |

#### Required Folder Structure (Input)

```plaintext
dataset_gen/
├── input_synthetic/           # Synthetic images (for synthetic_only mode)
│   ├── WT_1618-02/
│   ├── WT_JG/
│   ├── ...
│   ├── KO_1096-01/
│   └── ...
├── input_real/                # Real images (for synthetic_only and real_only modes)
│   ├── WT_1618-02/
│   ├── WT_JG/
│   ├── ...
│   ├── KO_1096-01/
│   └── ...
└── input_mixed/               # Mixed source (for mixed mode)
    ├── WT_1618-02/
    └── ...
```

Each cell-line folder contains images of that specific cell line. Images are split by cell line, not by image — this is the key to the leave-one-cell-line-out design.

#### Output Structure (Generated Datasets)

For `N_WT = 5` and `N_KO = 4`, 20 datasets are generated under `dataset_gen/output/`:

```plaintext
dataset_gen/output/
├── dataset_1/
│   ├── train/
│   │   ├── WT/                 # All WT lines except the held-out one
│   │   └── KO/                 # All KO lines except the held-out one
│   ├── test/
│   │   ├── WT/                 # Only the held-out WT line
│   │   └── KO/                 # Only the held-out KO line
│   └── dataset_1_info.txt      # Metadata (cell lines, image counts)
├── dataset_2/
...
└── dataset_20/
```

#### Example Workflow

1. Place cell-line folders with images in `dataset_gen/input_real/`
2. Configure `settings.py`:
   ```python
   train_data_source = "real_only"
   wt_lines = ["WT_1618-02", "WT_JG", "WT_JT", "WT_KM", "WT_MS"]
   ko_lines = ["KO_1096-01", "KO_1618-01", "KO_BR2986", "KO_BR3075"]
   ```

3. Run the program and select **2 → 1**:
   ```plaintext
   :DATASET GENERATOR FOR CROSS VALIDATION:
   Training data source: real_only
   Synthetic folder: .../dataset_gen/input_synthetic
   Real folder: .../dataset_gen/input_real

   Generation of datasets is starting...
   Mode: real_only
   Dataset 1: real_only mode
     Training images: WT=1250, KO=980
     Test images: WT=280, KO=210
   ...
   Datasets successfully created and saved to .../dataset_gen/output!
   ```

---

### 2.2 Automatic Cross Validation

**Description**: This action generates datasets internally and does not require the standalone Dataset Generator to have been run first. Runs the complete cross-validation loop over all generated datasets. For each fold, it creates the dataset, trains a model on the training cell lines, evaluates all saved checkpoints on the held-out test cell lines, and saves confusion matrices and per-checkpoint test metrics.

Since validation images are drawn from the test set (`ds_val_from_test_split`), test evaluation uses the **remaining** images in the test set that were not consumed by validation. If `ds_val_from_test_split` is `1.0`, all test images are used for validation and no test evaluation takes place.

#### Key Settings (from `settings.py`)

All training settings from Single Training apply. The following are the cross-validation-specific settings:

| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `ds_val_from_test_split` | float / False | Fraction of test set used for validation | `0.3` |
| `ds_val_from_train_split` | float / False | Fraction of train set used for validation | `False` |
| `chckpt_selection_method` | str | `"balanced_accuracy"`, `"composite_score"`, or `"both"` | `"balanced_accuracy"` |
| `chckpt_min_balanced_acc_threshold` | float | Minimum balanced accuracy to save | `0.65` |
| `chckpt_min_per_class_acc_balanced` | float | Minimum per-class accuracy | `0.60` |

#### Test Set Split Behavior

For test evaluation to run, `ds_val_from_test_split` must be between `0.0` and `1.0`:

| `ds_val_from_test_split` | Behavior |
|--------------------------|----------|
| `False` | Test set untouched during validation; but test evaluation is still skipped (only triggers when split is active) |
| `0.3` | 30% of test set used for validation, 70% used for final test evaluation |
| `1.0` | All test images used for validation; **no test evaluation** |

To evaluate on the full test set, use the **training split for validation** instead (`ds_val_from_train_split = 0.1`, `ds_val_from_test_split = False`).

#### Output Structure

Each fold produces its own result folder:

```plaintext
output/cross_validation/
├── dataset_1/
│   ├── checkpoints/            # Saved .pt files for this fold
│   │   ├── ckpt_pretr_densenet121_e23_bal0.860_comp0.812_ds1.pt
│   │   └── ...
│   ├── plots/                  # Validation AND test confusion matrices + ROC/PR data
│   │   ├── ckpt_..._val_cm.png
│   │   ├── ckpt_..._val_cm.json
│   │   ├── ckpt_..._test_cm.png
│   │   ├── ckpt_..._test_cm.json
│   │   └── ...
│   ├── logs/                   # TensorBoard event files
│   │   └── probabilities/      # Per-epoch probability data
│   └── split_info.json         # Records which real images were in validation vs test
├── dataset_2/
...
└── dataset_20/
```

The `split_info.json` records **only real images** used for validation and testing, filtered from any synthetic images. It is used downstream by the Confidence Analyzer.

#### Split Tracking with `split_info.json`

After each fold is created, the cross-validation script writes a `split_info.json` file into the fold's result folder (`output/cross_validation/dataset_XX/split_info.json`). This file records exactly which images were assigned to validation and which to testing.

Its purpose is to prevent **validation leakage** — the situation where an image used for monitoring during training is later reused for the final test evaluation, which would artificially inflate the test accuracy.

Key properties of `split_info.json`:

- **Real images only**: Synthetic images are filtered out before recording. This is enforced by the `is_synthetic_image()` check, which detects synthetic filenames by their `s<digit>...` pattern.
- **Disjoint splits**: The validation and test sets are guaranteed to be non-overlapping. If any image appears in both, the script corrects the overlap before writing the file.
- **Consumed by downstream steps**: Both the test evaluation in this module (2.2) and the Confidence Analyzer (2.3) read `split_info.json` to determine which images belong to which split.

Example structure of the file:

```json
{
  "validation": {
    "WT": ["img_0001.png", "img_0002.png", ...],
    "KO": ["img_0100.png", "img_0101.png", ...]
  },
  "test": {
    "WT": ["img_0050.png", "img_0051.png", ...],
    "KO": ["img_0150.png", "img_0151.png", ...]
  },
  "metadata": {
    "total_images_found": 620,
    "real_images_used": 520,
    "synthetic_filtered": 100,
    "note": "Only real images are recorded for validation/testing"
  }
}
```

The `metadata` block makes the filtering transparent: the user can see how many synthetic images were removed and how many real images remained for evaluation.

#### Example Workflow

1. Configure `settings.py`:
   ```python
   ds_val_from_train_split = False
   ds_val_from_test_split = 0.3
   chckpt_selection_method = "balanced_accuracy"
   ```

2. Run the program and select **2 → 2**:
   ```plaintext
   :AUTOMATIC CROSS VALIDATION:
     Results will be saved to output/cross_validation/

   Cleaning up old train and test data...
   Cleanup finished.

   >> PROCESSING DATASET 1 OF 20:
   Cell line for testing WT group: WT_1618-02
   Cell line for testing KO group: KO_1096-01

   > Create dataset 1...
   Dataset 1 successfully created.

   > Load dataset 1 for training...
   Number training images/batches: 4280/86
   Number validation images/batches: 147/3

   > Start training on dataset 1...
   >> Epoch [1/40]: ...
   ...
   Training on dataset 1 successfully finished.

   >>> Evaluating ALL checkpoints on TEST set (343 images)...
   Found 3 checkpoints to evaluate on test set
     > Testing checkpoint epoch 23 (val_acc=86%)...
       Test accuracy: 84.55%
       Test WT: 85.12%
       Test KO: 83.98%
   ✅ Finished testing all checkpoints for dataset 1
   ...
   Cross-validation complete.
   ```

---

### 2.3 Confidence Analyzer

**Description**: Analyzes predictions across all cross-validation folds to identify images that are consistently and reliably classified by multiple independently trained models. This is used to build a **high-confidence dataset** for downstream applications such as LoRA training on diffusion models or further analysis.

Because synthetic images are used only for training, they are explicitly excluded from all confidence analysis. The analyzer filters them out using the `split_info.json` file generated during cross-validation, which records only real images for validation and test splits. This ensures that the high-confidence dataset used for downstream applications (e.g., LoRA training) consists exclusively of real cells.

The analyzer loads the top checkpoints from each fold (selected by a configurable metric), runs predictions on the corresponding held-out test images, aggregates the results per image, and copies images that meet the configured filter criteria into a categorized output folder.

#### Key Settings (from `settings.py`)

| Setting | Type | Description | Example Value |
|---------|------|-------------|---------------|
| `ca_min_conf` | float | Minimum confidence threshold | `0.8` |
| `ca_max_conf` | float | Maximum confidence threshold | `1.0` |
| `ca_filter_type` | str | Filter mode: `"correct"`, `"incorrect"`, `"low_confidence"`, `"unsure"` | `"correct"` |
| `ca_max_ckpts` | int | Maximum checkpoints analyzed per fold | `1` |
| `ca_ckpt_select_method` | str | Checkpoint selection metric | `"balanced_accuracy"` |
| `ca_use_test_cm` | str | Which confusion matrix to use for checkpoint selection: `"validation"` or `"test"` | `"validation"` |
| `ca_split_to_use` | str | Which split to analyze: `"validation"`, `"test"`, or `"all"` | `"validation"` |

#### Filter Types

| Type | Confidence Range | Correctness Requirement | Output Folder | Use Case |
|------|------------------|-------------------------|---------------|----------|
| `correct` | `[min, max]` | Must be correct in all folds | `high_confidence_correct` | Reliable predictions |
| `incorrect` | `[min, max]` | Must be wrong in all folds | `high_confidence_incorrect` | Systematic errors |
| `low_confidence` | `< min` | Ignored | `low_confidence` | Ambiguous cases |
| `unsure` | `[min, max]` | Ignored | `medium_confidence_unsure` | Borderline predictions |

#### Checkpoint Selection Metrics

| Metric | Description |
|--------|-------------|
| `balanced_accuracy` | Average of WT and KO accuracy (recommended) |
| `balanced_sum` | `(WT + KO) − |WT − KO|` (favors balanced performance) |
| `f1_score` | Harmonic mean of WT and KO accuracy |
| `min_difference` | Minimum of WT and KO accuracy |
| `composite_score` | `overall_accuracy − penalty_weight × std(class_accuracies)` |

#### Output Structure

```plaintext
output/conf_analyzer/
├── high_confidence_correct/
│   ├── WT/
│   │   ├── img1_conf98_corr100.png
│   │   └── img2_conf95_corr100.png
│   └── KO/
│       └── img3_conf92_corr100.png
├── confidence_analysis.csv      # Per-fold, per-class prediction statistics
├── used_checkpoints.csv         # Which checkpoints were analyzed per fold
└── README.txt                   # Description of the filter applied
```

Filenames embed the average confidence (`confXX`) and correctness rate (`corrXX`) across all folds.

#### Expected Outcome

The retention rate depends on the filter type and the total number of folds. For `ca_filter_type = "correct"`, the analyzer typically retains **20–35%** of real images — the ones consistently recognized by all independently trained models. This subset is enriched for unambiguous, high-confidence examples of each phenotype and is well-suited for downstream tasks such as LoRA training on diffusion models.

| Filter | Typical Yield | Interpretation |
|--------|---------------|----------------|
| `correct` | 20–35% | Images consistently recognized by all folds |
| `incorrect` | 1–5% | Images systematically misclassified (potential label issues or biologically ambiguous) |
| `low_confidence` | 5–15% | Images with high biological variability |
| `unsure` | 10–20% | Borderline cases near the decision boundary |

Exact retention rates vary with the number of folds, the choice of cell lines, and the confidence thresholds.

#### Example Workflow

1. Ensure the cross-validation has completed and produces `output/cross_validation/dataset_XX/` folders
2. Configure `settings.py`:
   ```python
   ca_min_conf = 0.8
   ca_max_conf = 1.0
   ca_filter_type = "correct"
   ca_max_ckpts = 1
   ca_ckpt_select_method = "balanced_accuracy"
   ca_use_test_cm = "validation"
   ca_split_to_use = "validation"
   ```

3. Run the program and select **2 → 3**:
   ```plaintext
   :CONFIDENCE ANALYZER:
     Input: output/cross_validation/
     Output: output/conf_analyzer/

   Starting confidence analysis...
   Found 20 datasets to analyze
   Using VALIDATION confusion matrices for checkpoint selection
   Using 'VALIDATION' split for analysis
   ...
   Found 3421 images matching criteria. Organizing...
   Filtered images saved to: .../output/conf_analyzer/high_confidence_correct
   ...
   Analysis complete!
   ```

---