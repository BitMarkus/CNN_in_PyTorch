# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Standard Library Imports =====
from pathlib import Path
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')
# ===== Third-Party Imports =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing import event_accumulator
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# ===== Own Modules =====
from settings import setting

class TensorBoardPlotter:

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the TensorBoard plotter.
    # Args:
    #   logdir (Path): Path to TensorBoard logs directory
    #   prob_dir (Path, optional): Path to directory containing probability .npz files
    #   output_folder (Path): Folder for output plots
    #   output_format (str): 'png', 'tiff', 'svg', 'pdf'
    #   raster_dpi (int): DPI for raster formats
    #   mode (str): 'auto', 'crossval', or 'single'
    #   roc_curve_epoch (str or int): Which epoch to use for ROC curves
    #   pr_curve_epoch (str or int): Which epoch to use for PR curves
    #   checkpoint_map (dict): Map of run names to epoch numbers (from Confidence Analyzer)
    #   use_fixed_axes_height (bool): Consistent axes height across plots
    #   fixed_axes_height (float): Height of axes in inches
    #   master_font_size (int): Uniform font size for all text
    #   line_width (float): Width of plot lines
    #   show_markers (bool): Show markers at data points
    #   marker_size (int): Size of markers
    #   marker_frequency (int): Show every Nth marker
    #   smoothing (float): Smoothing factor for curves
    #   font_family (str): Font family
    #   axis_label_size (int): Font size for axis labels
    #   title_font_size (int): Font size for plot titles
    #   tick_label_size (int): Font size for tick labels
    #   legend_font_size (int): Font size for legend text
    #   show_legend (bool): Show/hide legend
    #   show_grid (bool): Show background grid
    #   grid_alpha (float): Grid line transparency
    #   grid_linestyle (str): Grid line style
    #   grid_color (str): Grid line color
    #   min_class_acc_threshold (float): Threshold for minimum class accuracy
    #   loss_legend_loc (str): Legend position for loss curves
    #   accuracy_legend_loc (str): Legend position for accuracy curves
    #   f1_legend_loc (str): Legend position for F1 curves
    #   per_class_legend_loc (str): Legend position for per-class accuracy
    #   roc_legend_loc (str): Legend position for ROC curves
    #   pr_legend_loc (str): Legend position for PR curves
    #   per_class_legend_outside (bool): Place per-class legend outside
    #   roc_legend_outside (bool): Place ROC legend outside
    #   pr_legend_outside (bool): Place PR legend outside
    #   plot_loss (bool): Generate loss plots
    #   plot_accuracy (bool): Generate accuracy plots
    #   plot_f1 (bool): Generate F1 plots
    #   plot_lr (bool): Generate learning rate plots
    #   plot_auc (bool): Generate AUC plots
    #   plot_ap (bool): Generate Average Precision plots
    #   plot_per_class_accuracy (bool): Generate per-class accuracy plots
    #   plot_class_weights (bool): Generate class weights bar charts
    #   plot_class_counts (bool): Generate class counts bar charts
    #   plot_composite_score (bool): Generate composite score plots
    #   plot_balanced_accuracy (bool): Generate balanced accuracy plots
    #   plot_class_std (bool): Generate class accuracy standard deviation plots
    #   plot_min_class_acc (bool): Generate minimum class accuracy plots
    #   plot_gpu_memory (bool): Generate GPU memory usage plots
    #   plot_roc_curves (bool): Generate ROC curve plots
    #   plot_pr_curves (bool): Generate PR curve plots
    def __init__(
        self,
        logdir,
        prob_dir=None,
        output_folder=None,
        output_format='tiff',
        raster_dpi=300,
        mode='auto',
        roc_curve_epoch='balanced_accuracy',
        pr_curve_epoch='balanced_accuracy',
        checkpoint_map=None,
        use_fixed_axes_height=True,
        fixed_axes_height=5.0,
        master_font_size=22,
        line_width=2.0,
        show_markers=True,
        marker_size=5,
        marker_frequency=1,
        smoothing=0.0,
        font_family='Arial',
        axis_label_size=12,
        title_font_size=14,
        tick_label_size=10,
        legend_font_size=10,
        show_legend=True,
        show_grid=True,
        grid_alpha=0.3,
        grid_linestyle='--',
        grid_color='gray',
        min_class_acc_threshold=0.65,
        loss_legend_loc='upper right',
        accuracy_legend_loc='lower right',
        f1_legend_loc='lower right',
        per_class_legend_loc='best',
        roc_legend_loc='lower right',
        pr_legend_loc='lower left',
        per_class_legend_outside=True,
        roc_legend_outside=True,
        pr_legend_outside=True,
        plot_loss=True,
        plot_accuracy=True,
        plot_f1=True,
        plot_lr=False,
        plot_auc=False,
        plot_ap=False,
        plot_per_class_accuracy=True,
        plot_class_weights=False,
        plot_class_counts=False,
        plot_composite_score=False,
        plot_balanced_accuracy=True,
        plot_class_std=False,
        plot_min_class_acc=False,
        plot_gpu_memory=False,
        plot_roc_curves=True,
        plot_pr_curves=True
    ) -> None:
        
        # Get script directory
        self.script_dir = Path(__file__).parent.absolute()

        # Input/Output settings
        self.logdir = Path(logdir)
        self.prob_dir = Path(prob_dir) if prob_dir else self.logdir / "probabilities"
        self.output_folder = Path(output_folder) if output_folder else setting['pth_output']
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.output_format = output_format.lower()
        self.raster_dpi = raster_dpi
        self.mode = mode
        self.run_type = None

        # ROC/PR settings
        self.roc_curve_epoch = roc_curve_epoch
        self.pr_curve_epoch = pr_curve_epoch
        self.checkpoint_map = checkpoint_map or {}

        # Figure size settings
        self.use_fixed_axes_height = use_fixed_axes_height
        self.fixed_axes_height = fixed_axes_height

        # Font settings
        self.font_family = font_family
        self.master_font_size = master_font_size

        if master_font_size is not None:
            self.axis_label_size = master_font_size
            self.title_font_size = master_font_size
            self.tick_label_size = master_font_size
            self.legend_font_size = master_font_size
        else:
            self.axis_label_size = axis_label_size
            self.title_font_size = title_font_size
            self.tick_label_size = tick_label_size
            self.legend_font_size = legend_font_size

        # Line styling
        self.line_width = line_width
        self.show_markers = show_markers
        self.marker_size = marker_size
        self.marker_frequency = marker_frequency
        self.smoothing = smoothing

        # Legend settings
        self.show_legend = show_legend
        self.loss_legend_loc = loss_legend_loc
        self.accuracy_legend_loc = accuracy_legend_loc
        self.f1_legend_loc = f1_legend_loc
        self.per_class_legend_loc = per_class_legend_loc
        self.roc_legend_loc = roc_legend_loc
        self.pr_legend_loc = pr_legend_loc
        self.per_class_legend_outside = per_class_legend_outside
        self.roc_legend_outside = roc_legend_outside
        self.pr_legend_outside = pr_legend_outside

        # Grid settings
        self.show_grid = show_grid
        self.grid_alpha = grid_alpha
        self.grid_linestyle = grid_linestyle
        self.grid_color = grid_color

        # Threshold settings
        self.min_class_acc_threshold = min_class_acc_threshold

        # Plot flags
        self.plot_loss = plot_loss
        self.plot_accuracy = plot_accuracy
        self.plot_f1 = plot_f1
        self.plot_lr = plot_lr
        self.plot_auc = plot_auc
        self.plot_ap = plot_ap
        self.plot_per_class_accuracy = plot_per_class_accuracy
        self.plot_class_weights = plot_class_weights
        self.plot_class_counts = plot_class_counts
        self.plot_composite_score = plot_composite_score
        self.plot_balanced_accuracy = plot_balanced_accuracy
        self.plot_class_std = plot_class_std
        self.plot_min_class_acc = plot_min_class_acc
        self.plot_gpu_memory = plot_gpu_memory
        self.plot_roc_curves = plot_roc_curves
        self.plot_pr_curves = plot_pr_curves

        # Colors for fixed-line plots
        self.train_color = '#2E86AB'
        self.val_color = '#A23B72'
        self.standard_train_color = '#2E86AB'
        self.standard_val_color = '#A23B72'
        self.lr_color = '#2E86AB'
        self.f1_weighted_color = '#2E86AB'
        self.f1_macro_color = '#F39C12'
        self.auc_color = '#27AE60'
        self.ap_color = '#8E44AD'
        self.composite_score_color = '#E74C3C'
        self.balanced_accuracy_color = '#1ABC9C'
        self.class_std_color = '#E67E22'
        self.min_class_acc_color = '#1ABC9C'
        self.gpu_memory_color = '#95A5A6'

        # Colormap for variable-line plots
        self.per_class_colormap = 'tab10'

        # Validate settings
        self._validate_settings()

        # Setup plot style
        self._setup_plot_style()

        # Data storage
        self.runs_data = {}
        self.per_class_data = {}
        self.class_weight_data = {}
        self.class_count_data = {}
        self.roc_data = {}
        self.pr_data = {}

        self._print_configuration()

    #############################################################################################################
    # METHODS

    # Validate that settings are valid.
    def _validate_settings(self) -> None:
        valid_formats = ['png', 'tiff', 'tif', 'svg', 'pdf']
        if self.output_format not in valid_formats:
            raise ValueError(f"output_format must be one of {valid_formats}, got '{self.output_format}'")

        if self.raster_dpi < 150:
            print(f"  ⚠️ Warning: {self.raster_dpi} DPI is low for publication. 300 DPI minimum recommended.")

        if self.fixed_axes_height <= 0:
            raise ValueError(f"fixed_axes_height must be positive, got {self.fixed_axes_height}")

        if self.marker_frequency < 1:
            raise ValueError(f"marker_frequency must be >= 1, got {self.marker_frequency}")

        if self.smoothing < 0 or self.smoothing > 1:
            raise ValueError(f"smoothing must be between 0 and 1, got {self.smoothing}")

    # Set publication-ready matplotlib defaults.
    def _setup_plot_style(self) -> None:
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': [self.font_family, 'Arial', 'Helvetica'],
            'font.size': self.tick_label_size,
            'axes.labelsize': self.axis_label_size,
            'axes.titlesize': self.title_font_size,
            'axes.titleweight': 'bold',
            'xtick.labelsize': self.tick_label_size,
            'ytick.labelsize': self.tick_label_size,
            'figure.dpi': self.raster_dpi,
            'savefig.dpi': self.raster_dpi,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.color': self.grid_color,
            'grid.linestyle': self.grid_linestyle,
            'grid.alpha': 1.0,
        })

    # Print current configuration.
    def _print_configuration(self) -> None:
        print("=" * 60)
        print("COMPLETE TENSORBOARD PUBLICATION PLOTTER")
        print("NOTE: All epochs in plots start at 1 (shifted from internal 0-index)")
        print("NOTE: All axes start at 0 (x_min=0, y_min=0)")
        if self.use_fixed_axes_height:
            print(f"NOTE: Fixed axes height = {self.fixed_axes_height} inches")
        if self.checkpoint_map:
            print(f"Using pre-selected checkpoints: {len(self.checkpoint_map)} run(s)")
        print("=" * 60)
        print(f"Log directory:       {self.logdir}")
        print(f"Probabilities dir:   {self.prob_dir}")
        print(f"Output folder:       {self.output_folder}")
        print(f"Output format:       {self.output_format.upper()}")
        if self.output_format in ['png', 'tiff', 'tif']:
            print(f"Resolution:          {self.raster_dpi} DPI")
        if self.use_fixed_axes_height:
            print(f"Plot axes height:    {self.fixed_axes_height} inches")
        if self.master_font_size is not None:
            print(f"Font size:           {self.master_font_size} points (uniform)")
        print(f"Smoothing factor:    {self.smoothing}" + (" (no smoothing)" if self.smoothing == 0 else ""))
        print(f"Axis limits:         x_min=0, y_min=0")
        print("=" * 60)

    # Apply smoothing to values.
    def _apply_smoothing(self, values: list, alpha: float = 0.95) -> list:
        if alpha <= 0 or len(values) == 0:
            return values

        smoothed = np.zeros_like(values, dtype=float)
        last = values[0]
        for i, val in enumerate(values):
            smoothed_val = alpha * last + (1 - alpha) * val
            smoothed[i] = smoothed_val
            last = smoothed_val
        return smoothed

    # Convert 0-indexed steps to 1-indexed.
    def _get_shifted_steps(self, steps_0indexed: list) -> list:
        return [s + 1 for s in steps_0indexed]

    # Check if run name follows cross-validation pattern (ds01, ds02, etc.).
    def _is_cross_validation_run(self, run_name: str) -> bool:
        return bool(re.match(r'^ds\d{2}$', run_name))

    # Check if run name follows timestamp pattern (YYYYMMDD-HHMMSS).
    def _is_timestamp_run(self, run_name: str) -> bool:
        return bool(re.match(r'^\d{8}-\d{6}$', run_name))

    # Auto-detect run type.
    def _detect_run_type(self, runs: list) -> str:
        if self.mode == 'crossval':
            print("✓ Mode forced: CROSS-VALIDATION")
            return 'crossval'
        elif self.mode == 'single':
            print("✓ Mode forced: SINGLE TRAINING")
            return 'single'

        crossval_count = 0
        timestamp_count = 0

        for run_path, run_name, _ in runs:
            if self._is_cross_validation_run(run_name):
                crossval_count += 1
            elif self._is_timestamp_run(run_name):
                timestamp_count += 1

        if crossval_count > 0 and timestamp_count == 0:
            print(f"✓ Auto-detected: CROSS-VALIDATION mode ({crossval_count} dsXX folders)")
            return 'crossval'
        elif timestamp_count > 0 and crossval_count == 0:
            print(f"✓ Auto-detected: SINGLE TRAINING mode ({timestamp_count} timestamp folders)")
            return 'single'
        elif crossval_count > 0 and timestamp_count > 0:
            print(f"⚠ Mixed run types detected! Using CROSS-VALIDATION mode as default.")
            return 'crossval'
        else:
            if list(self.logdir.glob("events.out.tfevents.*")):
                print(f"✓ Auto-detected: SINGLE TRAINING mode (direct event files)")
                return 'single'
            print(f"⚠ Could not determine run type. Using CROSS-VALIDATION mode as default.")
            return 'crossval'

    # Get the probability subdirectory based on run type.
    def _get_probability_subdir(self, run_name: str):
        if not self.prob_dir:
            return None

        if self.run_type == 'crossval':
            # Convert dsXX to dataset_XX
            dataset_num = run_name.replace("ds", "")
            dataset_name = f"dataset_{dataset_num}"

            # NEW structure: dataset_X/logs/probabilities/
            prob_subdir = self.prob_dir / dataset_name / "logs" / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            # OLD structure: logs/dsXX/probabilities/
            prob_subdir = self.logdir / run_name / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            # Alternative: self.logdir is the logs folder with dataset_X
            prob_subdir = self.logdir / dataset_name / "logs" / "probabilities"
            if prob_subdir.exists():
                return prob_subdir

            return None
        else:
            # Single training: same as before
            if self.prob_dir.name == "probabilities":
                return self.prob_dir
            prob_subdir = self.logdir / "probabilities"
            if prob_subdir.exists():
                return prob_subdir
            if run_name and self.logdir.name != run_name:
                prob_subdir = self.logdir / run_name / "probabilities"
            else:
                prob_subdir = self.logdir / "probabilities"
            return prob_subdir

    # Find runs in the log directory (supports both old and new structures)
    def find_runs(self) -> list:
        all_event_files = list(self.logdir.rglob("events.out.tfevents.*"))

        if not all_event_files:
            print(f"❌ No event files found in {self.logdir}")
            return []

        folders = defaultdict(list)
        
        for ef in all_event_files:
            parent = ef.parent
            
            # NEW structure: dataset_X/logs/
            if parent.name == "logs" and parent.parent.name.startswith("dataset_"):
                run_name = parent.parent.name.replace("dataset_", "ds")
                folders[parent.parent].append(ef)
            
            # NEW structure (alternative): dataset_X/logs/ with run_name already
            elif parent.name == "logs" and parent.parent.name.startswith("ds"):
                run_name = parent.parent.name
                folders[parent.parent].append(ef)
            
            # OLD structure: logs/dsXX/
            elif parent.name.startswith("ds") and parent.parent.name == "logs":
                run_name = parent.name
                folders[parent].append(ef)
            
            # Single training: direct event files
            elif parent.name == "logs" and parent.parent.name not in ["cross_validation", "acv_results"]:
                # Single training - use the timestamp folder name
                run_name = parent.parent.name
                folders[parent.parent].append(ef)
            
            # Fallback: use the folder name
            else:
                run_name = parent.name
                folders[parent].append(ef)

        runs = []
        for folder_path, files in folders.items():
            # Determine run name from folder structure
            if folder_path.name.startswith("dataset_"):
                run_name = folder_path.name.replace("dataset_", "ds")
            elif folder_path.name.startswith("ds") and folder_path.parent.name == "logs":
                run_name = folder_path.name
            else:
                run_name = folder_path.name
            runs.append((folder_path, run_name, files))

        if runs:
            self.run_type = self._detect_run_type(runs)

        return runs

    # Extract metric from events.
    @staticmethod
    def _extract_metric_from_events(events, metric_name: str, target_value: str = None):
        steps, values = [], []
        for e in events:
            if isinstance(e.value, dict):
                if target_value and target_value in e.value:
                    steps.append(e.step)
                    values.append(e.value[target_value])
                elif metric_name in e.value:
                    steps.append(e.step)
                    values.append(e.value[metric_name])
            else:
                steps.append(e.step)
                values.append(e.value)
        if not steps:
            return None, None
        d = defaultdict(list)
        for s, v in zip(steps, values):
            d[s].append(v)
        steps_sorted = sorted(d.keys())
        return steps_sorted, [np.mean(d[s]) for s in steps_sorted]

    # Calculate balanced accuracy from per-class accuracies (fallback).
    def _calculate_balanced_accuracy_from_per_class(self, run_name: str, epoch_1indexed: int) -> float:
        if run_name not in self.per_class_data:
            return None

        per_class_data = self.per_class_data[run_name]
        if not per_class_data:
            return None

        class_accs = []
        for class_name, df in per_class_data.items():
            epoch_row = df[df['step'] == epoch_1indexed]
            if not epoch_row.empty:
                class_accs.append(epoch_row['value'].values[0])

        if not class_accs:
            return None

        return np.mean(class_accs)

    # Get best epoch by a specific metric.
    def _get_best_epoch_by_metric(self, run_name: str, metric_name: str) -> int:
        if run_name not in self.runs_data:
            return None

        run_data = self.runs_data[run_name]

        if metric_name == 'composite_score':
            if 'composite_score' not in run_data:
                return None
            scores = run_data['composite_score']['value']
            epochs = run_data['composite_score']['step']
            if len(scores) == 0:
                return None
            valid = ~np.isnan(scores)
            if not np.any(valid):
                return None
            best_idx = np.argmax(scores[valid])
            return int(epochs[valid][best_idx])

        elif metric_name == 'balanced_accuracy':
            if 'balanced_accuracy' in run_data:
                scores = run_data['balanced_accuracy']['value']
                epochs = run_data['balanced_accuracy']['step']
                if len(scores) > 0:
                    valid = ~np.isnan(scores)
                    if np.any(valid):
                        best_idx = np.argmax(scores[valid])
                        return int(epochs[valid][best_idx])

            if run_name not in self.per_class_data:
                return None

            per_class_data = self.per_class_data[run_name]
            if not per_class_data:
                return None

            first_class = next(iter(per_class_data.keys()))
            epochs = per_class_data[first_class]['step'].values

            best_epoch = None
            best_score = -float('inf')

            for epoch in epochs:
                bal_acc = self._calculate_balanced_accuracy_from_per_class(run_name, epoch)
                if bal_acc is not None and bal_acc > best_score:
                    best_score = bal_acc
                    best_epoch = int(epoch)

            return best_epoch

        return None

    # Extract all scalar data from a run's event files.
    def extract_run_data(self, run_path: Path, event_files: list):
        all_metrics = defaultdict(list)

        for event_file in event_files:
            try:
                ea = event_accumulator.EventAccumulator(
                    str(event_file),
                    size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE
                )
                ea.Reload()

                folder_name = event_file.parent.name

                for tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)

                    if folder_name in ['Metrics_F1_Macro', 'Metrics_F1_Weighted']:
                        steps = [e.step for e in events]
                        values = [e.value for e in events]
                        if steps:
                            key = 'f1_macro' if 'Macro' in folder_name else 'f1_weighted'
                            all_metrics[(key, folder_name)].append((steps, values))
                        continue

                    if tag == 'Metrics/F1':
                        s, v = self._extract_metric_from_events(events, 'Macro', 'Macro')
                        if s:
                            all_metrics[('f1_macro', folder_name)].append((s, v))
                        s, v = self._extract_metric_from_events(events, 'Weighted', 'Weighted')
                        if s:
                            all_metrics[('f1_weighted', folder_name)].append((s, v))
                        continue

                    steps = [e.step for e in events]
                    values = [e.value for e in events]
                    if steps:
                        all_metrics[(tag, folder_name)].append((steps, values))

            except Exception:
                continue

        if not all_metrics:
            return None, None, None, None

        run_data = {}
        per_class_data = {}
        class_weight_data = {}
        class_count_data = {}

        for (tag, folder_name), metric_lists in all_metrics.items():
            all_steps = []
            all_values = []
            for steps, values in metric_lists:
                all_steps.extend(steps)
                all_values.extend(values)

            if not all_steps:
                continue

            step_dict = defaultdict(list)
            for step, val in zip(all_steps, all_values):
                step_dict[step].append(val)

            steps_0indexed = sorted(step_dict.keys())
            values = [np.mean(step_dict[step]) for step in steps_0indexed]
            steps_1indexed = self._get_shifted_steps(steps_0indexed)

            tag_lower = tag.lower()

            if tag == 'Metrics/F1':
                continue

            if 'loss' in tag_lower:
                if 'train' in tag_lower:
                    metric_name = 'loss_train'
                elif 'val' in tag_lower or 'valid' in tag_lower:
                    metric_name = 'loss_val'
                else:
                    continue

            elif 'accuracy' in tag_lower or 'acc' in tag_lower:
                if 'train' in tag_lower:
                    if 'standard' in tag_lower:
                        metric_name = 'acc_train_standard'
                    else:
                        metric_name = 'acc_train'
                elif 'val' in tag_lower or 'valid' in tag_lower:
                    if 'standard' in tag_lower:
                        metric_name = 'acc_val_standard'
                    else:
                        metric_name = 'acc_val'
                elif 'class' in tag_lower:
                    if 'std' in tag_lower or 'stddev' in tag_lower:
                        metric_name = 'class_std'
                    elif 'min' in tag_lower:
                        metric_name = 'min_class_acc'
                    else:
                        parts = tag.split('/')
                        class_name = parts[-1] if parts else tag
                        metric_name = f'per_class_{class_name}'
                else:
                    continue

            elif 'balanced' in tag_lower and 'accuracy' in tag_lower:
                metric_name = 'balanced_accuracy'

            elif 'f1' in tag_lower:
                if 'weighted' in tag_lower:
                    metric_name = 'f1_weighted'
                elif 'macro' in tag_lower:
                    metric_name = 'f1_macro'
                else:
                    continue

            elif 'auc' in tag_lower:
                if 'class' in tag_lower:
                    parts = tag.split('/')
                    class_name = parts[-1] if parts else 'unknown'
                    metric_name = f'auc_class_{class_name}'
                else:
                    metric_name = 'auc'

            elif 'ap' in tag_lower or 'average_precision' in tag_lower:
                if 'class' in tag_lower:
                    parts = tag.split('/')
                    class_name = parts[-1] if parts else 'unknown'
                    metric_name = f'ap_class_{class_name}'
                else:
                    metric_name = 'ap'

            elif 'lr' in tag_lower or 'learning_rate' in tag_lower:
                metric_name = 'lr'

            elif 'composite' in tag_lower:
                metric_name = 'composite_score'

            elif 'gpu_memory' in tag_lower:
                metric_name = 'gpu_memory'

            elif 'class_count' in tag_lower and 'data' in tag_lower:
                parts = tag.split('/')
                class_name = parts[-1] if parts else 'unknown'
                class_count_data[class_name] = {
                    'step': steps_1indexed,
                    'step_0indexed': steps_0indexed,
                    'value': values
                }
                continue

            elif 'class_weight' in tag_lower and 'data' in tag_lower:
                parts = tag.split('/')
                class_name = parts[-1] if parts else 'unknown'
                class_weight_data[class_name] = {
                    'step': steps_1indexed,
                    'step_0indexed': steps_0indexed,
                    'value': values
                }
                continue

            else:
                continue

            if metric_name in ['lr', 'class_std', 'min_class_acc', 'composite_score', 'balanced_accuracy']:
                smoothed = values
            else:
                smoothed = self._apply_smoothing(values, self.smoothing)

            df = pd.DataFrame({
                'step': steps_1indexed,
                'step_0indexed': steps_0indexed,
                'value': values,
                'smoothed': smoothed
            })

            if metric_name.startswith('per_class_'):
                class_name = metric_name.replace('per_class_', '')
                per_class_data[class_name] = df
            elif metric_name.startswith('auc_class_'):
                class_name = metric_name.replace('auc_class_', '')
                if 'per_class_auc' not in run_data:
                    run_data['per_class_auc'] = {}
                run_data['per_class_auc'][class_name] = df
            elif metric_name.startswith('ap_class_'):
                class_name = metric_name.replace('ap_class_', '')
                if 'per_class_ap' not in run_data:
                    run_data['per_class_ap'] = {}
                run_data['per_class_ap'][class_name] = df
            else:
                if metric_name not in run_data or len(df) > len(run_data[metric_name]):
                    run_data[metric_name] = df

        # Merge F1 data from subfolders
        for (key, fname), metric_list in all_metrics.items():
            if key == 'f1_macro' and fname == 'Metrics_F1_Macro':
                all_steps = []
                all_values = []
                for steps, values in metric_list:
                    all_steps.extend(steps)
                    all_values.extend(values)
                if all_steps:
                    step_dict = defaultdict(list)
                    for step, val in zip(all_steps, all_values):
                        step_dict[step].append(val)
                    steps_0indexed = sorted(step_dict.keys())
                    values = [np.mean(step_dict[step]) for step in steps_0indexed]
                    steps_1indexed = self._get_shifted_steps(steps_0indexed)
                    run_data['f1_macro'] = pd.DataFrame({
                        'step': steps_1indexed,
                        'step_0indexed': steps_0indexed,
                        'value': values,
                        'smoothed': values
                    })

            elif key == 'f1_weighted' and fname == 'Metrics_F1_Weighted':
                all_steps = []
                all_values = []
                for steps, values in metric_list:
                    all_steps.extend(steps)
                    all_values.extend(values)
                if all_steps:
                    step_dict = defaultdict(list)
                    for step, val in zip(all_steps, all_values):
                        step_dict[step].append(val)
                    steps_0indexed = sorted(step_dict.keys())
                    values = [np.mean(step_dict[step]) for step in steps_0indexed]
                    steps_1indexed = self._get_shifted_steps(steps_0indexed)
                    run_data['f1_weighted'] = pd.DataFrame({
                        'step': steps_1indexed,
                        'step_0indexed': steps_0indexed,
                        'value': values,
                        'smoothed': values
                    })

        return run_data, per_class_data, class_weight_data, class_count_data

    # Load probability data for a run.
    def _load_probability_data(self, run_name: str, epoch_choice) -> tuple:
        npz_files = []

        prob_subdir = self._get_probability_subdir(run_name)

        if prob_subdir and prob_subdir.exists():
            npz_files = sorted(prob_subdir.glob("probabilities_epoch_*.npz"))

        if not npz_files and self.prob_dir.exists():
            npz_files = sorted(self.prob_dir.glob(f"*{run_name}*_probabilities_epoch_*.npz"))

        if not npz_files and self.prob_dir.exists():
            npz_files = sorted(self.prob_dir.glob("probabilities_epoch_*.npz"))

        if not npz_files:
            return None, None, None

        epoch_files = []
        for f in npz_files:
            try:
                match = re.search(r'epoch_(\d+)', f.stem)
                if match:
                    internal_epoch = int(match.group(1))
                    epoch_files.append((internal_epoch, f))
            except:
                continue

        if not epoch_files:
            return None, None, None

        epoch_files.sort(key=lambda x: x[0])

        selected_internal_epoch = None
        selected_file = None

        map_key = run_name
        if self.run_type == 'single' and self._is_timestamp_run(run_name):
            if run_name in self.checkpoint_map:
                map_key = run_name
            elif 'single_run' in self.checkpoint_map:
                map_key = 'single_run'

        if self.checkpoint_map and map_key in self.checkpoint_map:
            target_epoch_1indexed = self.checkpoint_map[map_key]
            target_internal_epoch = target_epoch_1indexed - 1
            for internal_epoch, f in epoch_files:
                if internal_epoch == target_internal_epoch:
                    selected_internal_epoch, selected_file = internal_epoch, f
                    break
            if selected_internal_epoch is None:
                print(f"  WARNING: Pre-selected epoch {target_epoch_1indexed} not found")

        if selected_internal_epoch is None:
            if epoch_choice == 'last':
                selected_internal_epoch, selected_file = epoch_files[-1]
            elif epoch_choice == 'composite_score':
                best_epoch = self._get_best_epoch_by_metric(run_name, 'composite_score')
                if best_epoch is not None:
                    best_internal_epoch = best_epoch - 1
                    for internal_epoch, f in epoch_files:
                        if internal_epoch == best_internal_epoch:
                            selected_internal_epoch, selected_file = internal_epoch, f
                            break
                if selected_internal_epoch is None:
                    selected_internal_epoch, selected_file = epoch_files[-1]
            elif epoch_choice == 'balanced_accuracy':
                best_epoch = self._get_best_epoch_by_metric(run_name, 'balanced_accuracy')
                if best_epoch is not None:
                    best_internal_epoch = best_epoch - 1
                    for internal_epoch, f in epoch_files:
                        if internal_epoch == best_internal_epoch:
                            selected_internal_epoch, selected_file = internal_epoch, f
                            break
                if selected_internal_epoch is None:
                    selected_internal_epoch, selected_file = epoch_files[-1]
            else:
                try:
                    target_epoch_1indexed = int(epoch_choice)
                    target_internal_epoch = target_epoch_1indexed - 1
                    for internal_epoch, f in epoch_files:
                        if internal_epoch == target_internal_epoch:
                            selected_internal_epoch, selected_file = internal_epoch, f
                            break
                    if selected_internal_epoch is None:
                        selected_internal_epoch, selected_file = epoch_files[-1]
                except:
                    selected_internal_epoch, selected_file = epoch_files[-1]

        selected_epoch_1indexed = selected_internal_epoch + 1
        print(f"      Using epoch {selected_epoch_1indexed} from {selected_file.name}")

        data = np.load(selected_file)
        probabilities = data['probabilities']
        labels = data['labels']

        if 'classes' in data:
            class_names = data['classes'].tolist()
        else:
            class_names = [f'Class_{i}' for i in range(probabilities.shape[1])]

        return probabilities, labels, class_names

    # Process all runs.
    def process_runs(self) -> bool:
        runs = self.find_runs()

        if not runs:
            print("\n❌ No training runs found.")
            return False

        print("\n📊 Extracting scalar data from TensorBoard...")

        for run_path, run_name, event_files in runs:
            print(f"  Processing: {run_name} ({len(event_files)} event files)")
            run_data, per_class_data, class_weight_data, class_count_data = self.extract_run_data(run_path, event_files)

            if run_data and len(run_data) > 0:
                self.runs_data[run_name] = run_data
                print(f"    ✓ Metrics: {list(run_data.keys())}")

            if per_class_data and len(per_class_data) > 0:
                self.per_class_data[run_name] = per_class_data
                print(f"    ✓ Per-class: {list(per_class_data.keys())}")

            if class_weight_data and len(class_weight_data) > 0:
                self.class_weight_data[run_name] = class_weight_data
                print(f"    ✓ Class weights: {list(class_weight_data.keys())}")

            if class_count_data and len(class_count_data) > 0:
                self.class_count_data[run_name] = class_count_data
                print(f"    ✓ Class counts: {list(class_count_data.keys())}")

        print(f"\n✓ Loaded {len(self.runs_data)} run(s)")

        if self.plot_roc_curves or self.plot_pr_curves:
            print("\n📊 Loading probability data for ROC/PR curves...")
            for run_name in self.runs_data.keys():
                print(f"  {run_name}:")
                probs_roc, labels_roc, class_names = self._load_probability_data(run_name, self.roc_curve_epoch)
                if probs_roc is not None:
                    self.roc_data[run_name] = (probs_roc, labels_roc, class_names)
                else:
                    self.roc_data[run_name] = None

                if self.roc_curve_epoch == self.pr_curve_epoch:
                    self.pr_data[run_name] = self.roc_data[run_name]
                else:
                    probs_pr, labels_pr, class_names = self._load_probability_data(run_name, self.pr_curve_epoch)
                    self.pr_data[run_name] = (probs_pr, labels_pr, class_names) if probs_pr is not None else None

        return True

    # Create figure and axes with consistent sizing.
    def _create_figure_and_ax(self, has_legend: bool = False, has_colorbar: bool = False,
                             has_bbox_legend: bool = False, legend_outside: bool = False):
        if self.use_fixed_axes_height:
            axes_height = self.fixed_axes_height

            top_margin = 0.6
            bottom_margin = 0.8

            fig_height = axes_height + top_margin + bottom_margin

            if legend_outside:
                fig_width = 12.5
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                plt.subplots_adjust(left=0.12, right=0.78, bottom=0.12, top=0.92)
            elif has_bbox_legend:
                fig_width = 11.0
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                plt.subplots_adjust(left=0.12, right=0.85, bottom=0.12, top=0.92)
            elif has_colorbar:
                fig_width = 9.0
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                plt.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)
            else:
                fig_width = 8.0
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                plt.subplots_adjust(left=0.12, right=0.92, bottom=0.12, top=0.92)

            return fig, ax
        else:
            fig, ax = plt.subplots(figsize=(10, 6.5))
            return fig, ax

    # Add legend to plot.
    def _add_legend(self, ax, loc: str = 'best', outside: bool = False, outside_bbox: tuple = (1.05, 0.5)):
        if not self.show_legend:
            return None

        if outside:
            legend = ax.legend(
                fontsize=self.legend_font_size,
                frameon=True,
                framealpha=1.0,
                edgecolor=self.grid_color,
                loc='center left',
                bbox_to_anchor=outside_bbox
            )
        else:
            legend = ax.legend(
                loc=loc,
                fontsize=self.legend_font_size,
                frameon=True,
                framealpha=1.0,
                edgecolor=self.grid_color
            )

        if legend:
            legend.get_frame().set_linewidth(0.8)
            legend.get_frame().set_linestyle(self.grid_linestyle)

        return legend

    # Save figure with appropriate settings.
    def _save_figure(self, fig, output_path):
        save_kwargs = {'dpi': self.raster_dpi, 'bbox_inches': 'tight'}

        if self.output_format in ['png', 'jpg', 'jpeg']:
            fig.savefig(output_path, **save_kwargs)
        elif self.output_format in ['tiff', 'tif']:
            fig.savefig(output_path, format='tiff', **save_kwargs)
        else:
            fig.savefig(output_path, format=self.output_format, **save_kwargs)

        file_size_kb = output_path.stat().st_size / 1024
        file_size_mb = file_size_kb / 1024

        if file_size_mb >= 1:
            print(f"    ✓ {output_path.name} ({file_size_mb:.1f} MB)")
        else:
            print(f"    ✓ {output_path.name} ({file_size_kb:.0f} KB)")

        plt.close(fig)

    #############################################################################################################
    # PLOTTING METHODS

    # Plot loss curves for a run.
    def _plot_loss_curves_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=False)

        if 'loss_train' in metrics:
            x = metrics['loss_train']['step']
            y = metrics['loss_train']['smoothed']
            ax.plot(x, y, linestyle='-', linewidth=self.line_width,
                   color=self.train_color, label='Training')
            if self.show_markers:
                self._add_markers(ax, x, y, color=self.train_color)

        if 'loss_val' in metrics:
            x = metrics['loss_val']['step']
            y = metrics['loss_val']['smoothed']
            ax.plot(x, y, linestyle='-', linewidth=self.line_width,
                   color=self.val_color, label='Validation')
            if self.show_markers:
                self._add_markers(ax, x, y, color=self.val_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Loss Curves{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.loss_legend_loc, outside=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_loss_curves.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Add markers to plot.
    def _add_markers(self, ax, x, y, color=None):
        if not self.show_markers:
            return

        marker_indices = slice(None, None, self.marker_frequency)
        x_markers = np.array(x)[marker_indices]
        y_markers = np.array(y)[marker_indices]

        ax.plot(x_markers, y_markers, 'o', markersize=self.marker_size,
               color=color, linestyle='none', label='_nolegend_')

    # Clean run name for title.
    def _clean_run_name_for_title(self, run_name: str) -> str:
        if self.run_type == 'single' and self._is_timestamp_run(run_name):
            return ""
        return run_name

    # Get colors for bar charts.
    def _get_bar_colors(self, class_names, color_dict, colormap_name):
        if color_dict is not None:
            return [color_dict.get(cls, '#808080') for cls in class_names]
        else:
            cmap = plt.cm.get_cmap(colormap_name)
            return [cmap(i / len(class_names)) for i in range(len(class_names))]

    # Check if a color is white (for edge detection).
    def _is_white_color(self, color) -> bool:
        if isinstance(color, str):
            color_lower = color.lower()
            if color_lower in ['white', '#ffffff', '#fff', 'w']:
                return True
            if color_lower in ['whitesmoke', 'snow', 'ivory', 'floralwhite', 'oldlace']:
                return True
            if color.startswith('#'):
                if len(color) == 7:
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    if r > 240 and g > 240 and b > 240:
                        return True
            return False
        return False

    # Plot accuracy curves for a run.
    def _plot_accuracy_curves_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=False)

        if 'acc_train' in metrics:
            x = metrics['acc_train']['step']
            y = metrics['acc_train']['smoothed']
            ax.plot(x, y, linestyle='-', linewidth=self.line_width,
                   color=self.train_color, label='Training (weighted)')
            if self.show_markers:
                self._add_markers(ax, x, y, color=self.train_color)

        if 'acc_train_standard' in metrics:
            x = metrics['acc_train_standard']['step']
            y = metrics['acc_train_standard']['smoothed']
            ax.plot(x, y, linestyle='--', linewidth=self.line_width,
                   color=self.standard_train_color, alpha=0.7, label='Training (standard)')
            if self.show_markers:
                self._add_markers(ax, x, y, color=self.standard_train_color)

        if 'acc_val' in metrics:
            x = metrics['acc_val']['step']
            y = metrics['acc_val']['smoothed']
            ax.plot(x, y, linestyle='-', linewidth=self.line_width,
                   color=self.val_color, label='Validation (weighted)')
            if self.show_markers:
                self._add_markers(ax, x, y, color=self.val_color)

        if 'acc_val_standard' in metrics:
            x = metrics['acc_val_standard']['step']
            y = metrics['acc_val_standard']['smoothed']
            ax.plot(x, y, linestyle='--', linewidth=self.line_width,
                   color=self.standard_val_color, alpha=0.7, label='Validation (standard)')
            if self.show_markers:
                self._add_markers(ax, x, y, color=self.standard_val_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Accuracy Curves{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.accuracy_legend_loc, outside=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_accuracy_curves.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot F1 scores for a run.
    def _plot_f1_scores_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=False)

        f1_weighted_found = False
        f1_macro_found = False

        if 'f1_weighted' in metrics:
            x = metrics['f1_weighted']['step']
            y = metrics['f1_weighted']['smoothed']
            ax.plot(x, y, linestyle='-', linewidth=self.line_width,
                   color=self.f1_weighted_color, label='Weighted F1')
            if self.show_markers:
                self._add_markers(ax, x, y, color=self.f1_weighted_color)
            f1_weighted_found = True

        if 'f1_macro' in metrics:
            x = metrics['f1_macro']['step']
            y = metrics['f1_macro']['smoothed']
            ax.plot(x, y, linestyle='--', linewidth=self.line_width,
                   color=self.f1_macro_color, label='Macro F1')
            if self.show_markers:
                self._add_markers(ax, x, y, color=self.f1_macro_color)
            f1_macro_found = True

        if not f1_weighted_found and not f1_macro_found:
            ax.text(0.5, 0.5, 'No F1 data available',
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=self.title_font_size)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'F1 Scores{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend and (f1_weighted_found or f1_macro_found):
            self._add_legend(ax, loc=self.f1_legend_loc, outside=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_f1_scores.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot learning rate for a run.
    def _plot_learning_rate_for_run(self, run_name: str, metrics: dict):
        if 'lr' not in metrics:
            return None

        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=False)

        x = metrics['lr']['step']
        y = metrics['lr']['value']
        ax.plot(x, y, linestyle='-', linewidth=self.line_width,
               color=self.lr_color, label='Learning Rate')
        if self.show_markers:
            self._add_markers(ax, x, y, color=self.lr_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Learning Rate Schedule{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc='best', outside=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_learning_rate.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot AUC for a run.
    def _plot_auc_for_run(self, run_name: str, metrics: dict):
        if 'auc' not in metrics:
            return None

        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=False)

        x = metrics['auc']['step']
        y = metrics['auc']['smoothed']
        ax.plot(x, y, linestyle='-', linewidth=self.line_width,
               color=self.auc_color, label='AUC')
        if self.show_markers:
            self._add_markers(ax, x, y, color=self.auc_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'AUC over Training{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc='lower right', outside=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_auc_over_epochs.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot Average Precision for a run.
    def _plot_ap_for_run(self, run_name: str, metrics: dict):
        if 'ap' not in metrics:
            return None

        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=False)

        x = metrics['ap']['step']
        y = metrics['ap']['smoothed']
        ax.plot(x, y, linestyle='-', linewidth=self.line_width,
               color=self.ap_color, label='Average Precision')
        if self.show_markers:
            self._add_markers(ax, x, y, color=self.ap_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Precision')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Average Precision over Training{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc='lower right', outside=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_ap_over_epochs.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot per-class accuracy for a run.
    def _plot_per_class_accuracy_for_run(self, run_name: str, per_class_data: dict):
        if not per_class_data:
            return None

        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.per_class_legend_outside)

        cmap = plt.get_cmap(self.per_class_colormap)
        class_idx = 0

        for class_name, df in per_class_data.items():
            color = cmap(class_idx % cmap.N)
            class_idx += 1
            x = df['step']
            y = df['smoothed']
            ax.plot(x, y, linestyle='-', linewidth=self.line_width,
                   color=color, label=class_name)
            if self.show_markers:
                self._add_markers(ax, x, y, color=color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Per-Class Accuracy{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc=self.per_class_legend_loc, outside=self.per_class_legend_outside)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_per_class_accuracy.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot composite score for a run.
    def _plot_composite_score_for_run(self, run_name: str, metrics: dict):
        if 'composite_score' not in metrics:
            return None

        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=False)

        x = metrics['composite_score']['step']
        y = metrics['composite_score']['value']
        ax.plot(x, y, linestyle='-', linewidth=self.line_width,
               color=self.composite_score_color, label='Composite Score')
        if self.show_markers:
            self._add_markers(ax, x, y, color=self.composite_score_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Composite Score')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Composite Score{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc='lower right', outside=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_composite_score.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot balanced accuracy for a run.
    def _plot_balanced_accuracy_for_run(self, run_name: str, metrics: dict):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=False)

        if 'balanced_accuracy' in metrics:
            x = metrics['balanced_accuracy']['step']
            y = metrics['balanced_accuracy']['value']
        else:
            if run_name not in self.per_class_data:
                return None

            per_class_data = self.per_class_data[run_name]
            if not per_class_data:
                return None

            first_class = next(iter(per_class_data.keys()))
            epochs = per_class_data[first_class]['step'].values

            bal_acc_values = []
            for epoch in epochs:
                bal_acc = self._calculate_balanced_accuracy_from_per_class(run_name, epoch)
                if bal_acc is not None:
                    bal_acc_values.append(bal_acc)
                else:
                    bal_acc_values.append(np.nan)

            x = epochs
            y = np.array(bal_acc_values)

        ax.plot(x, y, linestyle='-', linewidth=self.line_width,
               color=self.balanced_accuracy_color, label='Balanced Accuracy')
        if self.show_markers:
            self._add_markers(ax, x, y, color=self.balanced_accuracy_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Balanced Accuracy')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Balanced Accuracy over Training{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc='lower right', outside=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_balanced_accuracy.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot class accuracy standard deviation for a run.
    def _plot_class_std_for_run(self, run_name: str, metrics: dict):
        if 'class_std' not in metrics:
            return None

        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=False)

        x = metrics['class_std']['step']
        y = metrics['class_std']['value']
        ax.plot(x, y, linestyle='-', linewidth=self.line_width,
               color=self.class_std_color, label='Class Accuracy Std Dev')
        if self.show_markers:
            self._add_markers(ax, x, y, color=self.class_std_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Standard Deviation')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Class Accuracy Variation{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc='upper right', outside=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_class_accuracy_std.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot minimum class accuracy for a run.
    def _plot_min_class_acc_for_run(self, run_name: str, metrics: dict):
        if 'min_class_acc' not in metrics:
            return None

        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=False)

        x = metrics['min_class_acc']['step']
        y = metrics['min_class_acc']['value']
        ax.plot(x, y, linestyle='-', linewidth=self.line_width,
               color=self.min_class_acc_color, label='Minimum Class Accuracy')
        if self.show_markers:
            self._add_markers(ax, x, y, color=self.min_class_acc_color)

        if self.min_class_acc_threshold is not None:
            ax.axhline(y=self.min_class_acc_threshold, color='red',
                      linestyle='--', alpha=0.5, label=f'Threshold ({self.min_class_acc_threshold:.0%})')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Minimum Class Accuracy')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Worst-Performing Class Accuracy{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc='lower right', outside=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_min_class_accuracy.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot GPU memory usage for a run.
    def _plot_gpu_memory_for_run(self, run_name: str, metrics: dict):
        if 'gpu_memory' not in metrics:
            return None

        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=False)

        x = metrics['gpu_memory']['step']
        y = metrics['gpu_memory']['value']
        ax.plot(x, y, linestyle='-', linewidth=self.line_width,
               color=self.gpu_memory_color, label='GPU Memory Usage')
        if self.show_markers:
            self._add_markers(ax, x, y, color=self.gpu_memory_color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Memory Usage Ratio')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'GPU Memory Usage{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            self._add_legend(ax, loc='best', outside=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        filename = f"{run_name}_gpu_memory.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot class weights bar chart for a run.
    def _plot_class_weights_for_run(self, run_name: str, class_weight_data: dict):
        if not class_weight_data:
            return None

        fig, ax = self._create_figure_and_ax()

        class_names = sorted(class_weight_data.keys())
        weights = [class_weight_data[cls]['value'][-1] for cls in class_names]
        weights_pct = np.array(weights) * 100

        fill_colors = self._get_bar_colors(class_names, None, 'viridis')

        max_val = max(weights_pct)
        label_offset = max_val * 0.05

        for i, (class_name, val, fill_color) in enumerate(zip(class_names, weights_pct, fill_colors)):
            if self._is_white_color(fill_color):
                edge_color = '#000000'
                edge_width = 0.5
            else:
                edge_color = fill_color
                edge_width = 0

            bar = ax.bar(class_name, val,
                       color=fill_color,
                       edgecolor=edge_color,
                       linewidth=edge_width)

            ax.text(bar[0].get_x() + bar[0].get_width()/2, val + label_offset,
                   f'{val:.1f}%', ha='center', va='bottom',
                   fontsize=10, color='black')

        ax.set_xlabel('Class')
        ax.set_ylabel('Loss Weight (%)')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Class Weights for Loss Function{(" - " + clean_name) if clean_name else ""}')
        ax.set_ylim(bottom=0, top=max_val * 1.10)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color, axis='y')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()

        filename = f"{run_name}_class_weights.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Plot class counts bar chart for a run.
    def _plot_class_counts_for_run(self, run_name: str, class_count_data: dict):
        if not class_count_data:
            return None

        fig, ax = self._create_figure_and_ax()

        class_names = sorted(class_count_data.keys())
        counts = [class_count_data[cls]['value'][0] for cls in class_names]

        fill_colors = self._get_bar_colors(class_names, None, 'plasma')

        max_count = max(counts)
        label_offset = max_count * 0.05

        bars = []
        for i, (class_name, count, fill_color) in enumerate(zip(class_names, counts, fill_colors)):
            if self._is_white_color(fill_color):
                edge_color = '#000000'
                edge_width = 0.5
            else:
                edge_color = fill_color
                edge_width = 0

            bar = ax.bar(class_name, count,
                       color=fill_color,
                       edgecolor=edge_color,
                       linewidth=edge_width)
            bars.append(bar[0])

        total = sum(counts)
        for bar, count, fill_color in zip(bars, counts, fill_colors):
            pct = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2, count + label_offset,
                   f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                   fontsize=10, color='black')

        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Images')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Class Distribution in Training Set{(" - " + clean_name) if clean_name else ""}')
        ax.set_ylim(bottom=0, top=max_count * 1.10)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color, axis='y')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()

        filename = f"{run_name}_class_counts.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    # Generate ROC curve for a run.
    def _generate_roc_curve_for_run(self, run_name: str, probabilities, labels, class_names, epoch_info: str):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.roc_legend_outside)

        cmap = plt.get_cmap(self.per_class_colormap)
        colors = [cmap(i % cmap.N) for i in range(len(class_names))]

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            binary_labels = (labels == i).astype(int)
            class_probs = probabilities[:, i]

            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            roc_auc_val = auc(fpr, tpr)

            ax.plot(fpr, tpr, color=color, lw=self.line_width,
                   label=f'{class_name} (AUC = {roc_auc_val:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='Random (AUC=0.5)')

        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=self.axis_label_size)
        ax.set_ylabel('True Positive Rate', fontsize=self.axis_label_size)

        title_display = run_name if self.run_type == 'crossval' else "Single Run"

        # Extract epoch number from epoch_info if present
        if "Best Epoch" in epoch_info:
            match = re.search(r'Best Epoch (\d+)', epoch_info)
            if match:
                epoch_num = match.group(1)
                ax.set_title(f'ROC Curves - {title_display} (Best Epoch {epoch_num})',
                            fontsize=self.title_font_size, fontweight='bold')
            else:
                ax.set_title(f'ROC Curves - {title_display}',
                            fontsize=self.title_font_size, fontweight='bold')
        else:
            ax.set_title(f'ROC Curves - {title_display} ({epoch_info})',
                        fontsize=self.title_font_size, fontweight='bold')

        if self.show_legend:
            self._add_legend(ax, loc=self.roc_legend_loc, outside=self.roc_legend_outside)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)

        plt.tight_layout()

        filename = f"{run_name}_roc_curves.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)

        # Save AUC values as CSV
        auc_df = pd.DataFrame([{class_name: auc_val} for class_name, auc_val in zip(class_names, [auc(fpr, tpr) for _, _, fpr, tpr in [self._compute_roc(probabilities, labels, class_names)[i] for i in range(len(class_names))]])])
        # Simplified: just save the values
        try:
            auc_values = {}
            for i, cls in enumerate(class_names):
                binary_labels = (labels == i).astype(int)
                class_probs = probabilities[:, i]
                fpr, tpr, _ = roc_curve(binary_labels, class_probs)
                auc_values[cls] = auc(fpr, tpr)
            auc_df = pd.DataFrame([auc_values]).T
            auc_df.columns = ['AUC']
            auc_df.index.name = 'Class'
            auc_df.to_csv(self.output_folder / f"{run_name}_roc_auc_values.csv")
        except:
            pass

        return filename

    # Helper to compute ROC for CSV export.
    def _compute_roc(self, probs, labels, class_names):
        curves = []
        for i, cls in enumerate(class_names):
            y_true = (labels == i).astype(int)
            y_score = probs[:, i]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            curves.append((cls, roc_auc, fpr, tpr))
        return curves

    # Generate PR curve for a run.
    def _generate_pr_curve_for_run(self, run_name: str, probabilities, labels, class_names, epoch_info: str):
        fig, ax = self._create_figure_and_ax(has_legend=self.show_legend, legend_outside=self.pr_legend_outside)

        cmap = plt.get_cmap(self.per_class_colormap)
        colors = [cmap(i % cmap.N) for i in range(len(class_names))]

        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            binary_labels = (labels == i).astype(int)
            class_probs = probabilities[:, i]

            precision, recall, _ = precision_recall_curve(binary_labels, class_probs)
            ap_val = average_precision_score(binary_labels, class_probs)

            ax.plot(recall, precision, color=color, lw=self.line_width,
                   label=f'{class_name} (AP = {ap_val:.3f})')

        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('Recall', fontsize=self.axis_label_size)
        ax.set_ylabel('Precision', fontsize=self.axis_label_size)

        title_display = run_name if self.run_type == 'crossval' else "Single Run"

        if "Best Epoch" in epoch_info:
            match = re.search(r'Best Epoch (\d+)', epoch_info)
            if match:
                epoch_num = match.group(1)
                ax.set_title(f'Precision-Recall Curves - {title_display} (Best Epoch {epoch_num})',
                            fontsize=self.title_font_size, fontweight='bold')
            else:
                ax.set_title(f'Precision-Recall Curves - {title_display}',
                            fontsize=self.title_font_size, fontweight='bold')
        else:
            ax.set_title(f'Precision-Recall Curves - {title_display} ({epoch_info})',
                        fontsize=self.title_font_size, fontweight='bold')

        if self.show_legend:
            self._add_legend(ax, loc=self.pr_legend_loc, outside=self.pr_legend_outside)
        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)

        plt.tight_layout()

        filename = f"{run_name}_pr_curves.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)

        # Save AP values as CSV
        try:
            ap_values = {}
            for i, cls in enumerate(class_names):
                binary_labels = (labels == i).astype(int)
                class_probs = probabilities[:, i]
                ap_values[cls] = average_precision_score(binary_labels, class_probs)
            ap_df = pd.DataFrame([ap_values]).T
            ap_df.columns = ['Average Precision']
            ap_df.index.name = 'Class'
            ap_df.to_csv(self.output_folder / f"{run_name}_pr_ap_values.csv")
        except:
            pass

        return filename

    # Plot per-class AUC for a run.
    def _plot_per_class_auc_for_run(self, run_name: str, metrics: dict):
        if 'per_class_auc' not in metrics:
            return None

        fig, ax = self._create_figure_and_ax(has_bbox_legend=True)

        cmap = plt.get_cmap(self.per_class_colormap)
        class_idx = 0

        for class_name, df in metrics['per_class_auc'].items():
            color = cmap(class_idx % cmap.N)
            class_idx += 1
            x = df['step']
            y = df['smoothed']
            ax.plot(x, y, linestyle='-', linewidth=self.line_width,
                   color=color, label=class_name)
            if self.show_markers:
                self._add_markers(ax, x, y, color=color)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        clean_name = self._clean_run_name_for_title(run_name)
        ax.set_title(f'Per-Class AUC{(" - " + clean_name) if clean_name else ""}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

        if self.show_grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle=self.grid_linestyle, color=self.grid_color)
        if self.show_legend:
            legend = ax.legend(fontsize=self.legend_font_size - 1, frameon=True,
                             framealpha=1.0, edgecolor=self.grid_color,
                             loc='center left', bbox_to_anchor=(1, 0.5))
            legend.get_frame().set_linewidth(0.8)
            legend.get_frame().set_linestyle(self.grid_linestyle)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        filename = f"{run_name}_per_class_auc.{self.output_format}".replace('.tiff', '.tif')
        self._save_figure(fig, self.output_folder / filename)
        return filename

    #############################################################################################################
    # MAIN GENERATION METHOD

    # Generate all plots for all runs.
    def generate_all_plots(self) -> list:
        print("\n" + "=" * 60)
        print("GENERATING PLOTS FOR EACH RUN")
        print("=" * 60)

        all_generated = []

        for run_name in self.runs_data.keys():
            print(f"\n📁 Processing run: {run_name}")
            metrics = self.runs_data[run_name]
            generated = []

            if self.plot_loss:
                if self._plot_loss_curves_for_run(run_name, metrics):
                    generated.append("loss_curves")

            if self.plot_accuracy:
                if self._plot_accuracy_curves_for_run(run_name, metrics):
                    generated.append("accuracy_curves")

            if self.plot_f1:
                if self._plot_f1_scores_for_run(run_name, metrics):
                    generated.append("f1_scores")

            if self.plot_lr:
                if self._plot_learning_rate_for_run(run_name, metrics):
                    generated.append("learning_rate")

            if self.plot_auc:
                if self._plot_auc_for_run(run_name, metrics):
                    generated.append("auc_over_epochs")
                if self._plot_per_class_auc_for_run(run_name, metrics):
                    generated.append("per_class_auc")

            if self.plot_ap:
                if self._plot_ap_for_run(run_name, metrics):
                    generated.append("ap_over_epochs")

            if self.plot_per_class_accuracy and run_name in self.per_class_data:
                if self._plot_per_class_accuracy_for_run(run_name, self.per_class_data[run_name]):
                    generated.append("per_class_accuracy")

            if self.plot_class_weights and run_name in self.class_weight_data:
                if self._plot_class_weights_for_run(run_name, self.class_weight_data[run_name]):
                    generated.append("class_weights")

            if self.plot_class_counts and run_name in self.class_count_data:
                if self._plot_class_counts_for_run(run_name, self.class_count_data[run_name]):
                    generated.append("class_counts")

            if self.plot_composite_score:
                if self._plot_composite_score_for_run(run_name, metrics):
                    generated.append("composite_score")

            if self.plot_balanced_accuracy:
                if self._plot_balanced_accuracy_for_run(run_name, metrics):
                    generated.append("balanced_accuracy")

            if self.plot_class_std:
                if self._plot_class_std_for_run(run_name, metrics):
                    generated.append("class_accuracy_std")

            if self.plot_min_class_acc:
                if self._plot_min_class_acc_for_run(run_name, metrics):
                    generated.append("min_class_accuracy")

            if self.plot_gpu_memory:
                if self._plot_gpu_memory_for_run(run_name, metrics):
                    generated.append("gpu_memory")

            # ROC and PR curves
            if self.plot_roc_curves and run_name in self.roc_data and self.roc_data[run_name] is not None:
                probabilities, labels, class_names = self.roc_data[run_name]

                # Determine epoch info for title
                if self.checkpoint_map:
                    map_key = run_name
                    if self.run_type == 'single' and self._is_timestamp_run(run_name):
                        if run_name in self.checkpoint_map:
                            map_key = run_name
                        elif 'single_run' in self.checkpoint_map:
                            map_key = 'single_run'

                    if map_key in self.checkpoint_map:
                        epoch_num = self.checkpoint_map[map_key]
                        epoch_info = f"Best Epoch {epoch_num} (by Conf. Analyzer)"
                    else:
                        epoch_info = self._get_epoch_info(run_name, self.roc_curve_epoch)
                else:
                    epoch_info = self._get_epoch_info(run_name, self.roc_curve_epoch)

                if self._generate_roc_curve_for_run(run_name, probabilities, labels, class_names, epoch_info):
                    generated.append("roc_curves")

            if self.plot_pr_curves and run_name in self.pr_data and self.pr_data[run_name] is not None:
                probabilities, labels, class_names = self.pr_data[run_name]

                if self.checkpoint_map:
                    map_key = run_name
                    if self.run_type == 'single' and self._is_timestamp_run(run_name):
                        if run_name in self.checkpoint_map:
                            map_key = run_name
                        elif 'single_run' in self.checkpoint_map:
                            map_key = 'single_run'

                    if map_key in self.checkpoint_map:
                        epoch_num = self.checkpoint_map[map_key]
                        epoch_info = f"Best Epoch {epoch_num} (by Conf. Analyzer)"
                    else:
                        epoch_info = self._get_epoch_info(run_name, self.pr_curve_epoch)
                else:
                    epoch_info = self._get_epoch_info(run_name, self.pr_curve_epoch)

                if self._generate_pr_curve_for_run(run_name, probabilities, labels, class_names, epoch_info):
                    generated.append("pr_curves")

            print(f"  ✓ Generated {len(generated)} plot(s)")
            all_generated.extend([f"{run_name}/{g}" for g in generated])

        print("\n" + "=" * 60)
        print(f"✅ Complete: {len(all_generated)} plot(s) generated for {len(self.runs_data)} run(s)")
        print(f"📁 {self.output_folder.absolute()}")
        print("=" * 60)

        return all_generated

    # Get epoch info string for titles.
    def _get_epoch_info(self, run_name: str, epoch_choice) -> str:
        if epoch_choice == 'composite_score':
            best_epoch = self._get_best_epoch_by_metric(run_name, 'composite_score')
            if best_epoch is not None:
                return f"Best Epoch {best_epoch} (by Composite Score)"
            return "Best Epoch (by Composite Score)"
        elif epoch_choice == 'balanced_accuracy':
            best_epoch = self._get_best_epoch_by_metric(run_name, 'balanced_accuracy')
            if best_epoch is not None:
                return f"Best Epoch {best_epoch} (by Balanced Accuracy)"
            return "Best Epoch (by Balanced Accuracy)"
        elif epoch_choice == 'last':
            if run_name in self.runs_data and 'acc_val' in self.runs_data[run_name]:
                epochs = self.runs_data[run_name]['acc_val']['step']
                if len(epochs) > 0:
                    return f"Last Epoch {int(epochs[-1])}"
            return "Last Epoch"
        else:
            return f"Epoch {epoch_choice}"

    #############################################################################################################
    # CALL

    # Run the complete plotting pipeline.
    # Returns:
    #   list: All generated plot filenames
    def __call__(self) -> list:
        if self.process_runs():
            return self.generate_all_plots()
        else:
            print("\n❌ Failed to process runs.")
            return []