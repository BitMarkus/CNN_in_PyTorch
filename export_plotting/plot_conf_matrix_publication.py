# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Standard Library Imports =====
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
# ===== Third-Party Imports =====
import json
import numpy as np
import matplotlib.pyplot as plt
# ===== Own Modules =====
from settings import setting

class ConfusionMatrixPlotter:

    # Publication-ready confusion matrix plotter for CNN classification results.

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the confusion matrix plotter.
    # Args:
    #   input_folder (Path): Folder containing JSON files
    #   output_folder (Path): Folder for output plots
    #   output_format (str): 'png', 'tiff', 'svg', 'pdf'
    #   raster_dpi (int): DPI for raster formats
    #   normalize (str): 'rows', 'columns', or 'none'
    #   show_counts (bool): Show raw counts in annotations
    #   combined_mode (bool): Side-by-side raw + normalized
    #   use_fixed_axes_height (bool): Use consistent axes height
    #   fixed_axes_height (float): Height of matrix in inches
    #   cmap (str): Colormap for heatmap
    #   show_title (bool): Show/hide title
    #   show_axis_labels (bool): Show/hide axis labels
    #   show_colorbar (bool): Show/hide colorbar
    #   show_per_class_accuracy (bool): Show accuracy on y-axis labels
    #   show_overall_accuracy (bool): Show overall accuracy in title
    #   font_family (str): Font family
    #   master_font_size (int): Master font size (overrides all below)
    #   axis_label_size (int): Axis label font size (points)
    #   title_font_size (int): Title font size (points)
    #   tick_label_size (int): Tick label font size (points)
    #   annotation_font_size (int): Annotation font size (points)
    #   legend_font_size (int): Colorbar font size (points)
    #   xtick_rotation (int): X-axis label rotation
    #   ytick_rotation (int): Y-axis label rotation
    #   annotation_decimal_places (int): Number of decimal places for annotations
    def __init__(
        self,
        input_folder=None,
        output_folder=None,
        output_format='tiff',
        raster_dpi=600,
        normalize='rows',
        show_counts=True,
        combined_mode=False,
        use_fixed_axes_height=True,
        fixed_axes_height=6.0,
        cmap='Blues',
        show_title=True,
        show_axis_labels=True,
        show_colorbar=True,
        show_per_class_accuracy=False,
        show_overall_accuracy=True,
        font_family='Arial',
        master_font_size=None,
        axis_label_size=30,
        title_font_size=30,
        tick_label_size=25,
        annotation_font_size=16,
        legend_font_size=30,
        xtick_rotation=45,
        ytick_rotation=0,
        annotation_decimal_places=3
    ) -> None:

        # Folder settings
        self.input_folder = Path(input_folder) if input_folder else setting['pth_input']
        self.output_folder = Path(output_folder) if output_folder else setting['pth_output']

        # Create folders if they don't exist
        self.input_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)

        # Output format settings
        self.output_format = output_format.lower()
        self.raster_dpi = raster_dpi

        # Matrix settings
        self.normalize = normalize
        self.show_counts = show_counts
        self.combined_mode = combined_mode

        # Size settings
        self.use_fixed_axes_height = use_fixed_axes_height
        self.fixed_axes_height = fixed_axes_height

        # Figure settings
        self.cmap = cmap

        # Text visibility settings
        self.show_title = show_title
        self.show_axis_labels = show_axis_labels
        self.show_colorbar = show_colorbar
        self.show_per_class_accuracy = show_per_class_accuracy
        self.show_overall_accuracy = show_overall_accuracy

        # Font settings
        self.font_family = font_family
        self.master_font_size = master_font_size

        if master_font_size is not None:
            self.axis_label_size = master_font_size
            self.title_font_size = master_font_size
            self.tick_label_size = master_font_size
            self.annotation_font_size = master_font_size
            self.legend_font_size = master_font_size
        else:
            self.axis_label_size = axis_label_size
            self.title_font_size = title_font_size
            self.tick_label_size = tick_label_size
            self.annotation_font_size = annotation_font_size
            self.legend_font_size = legend_font_size

        # Tick settings
        self.xtick_rotation = xtick_rotation
        self.ytick_rotation = ytick_rotation

        # Annotation settings
        self.annotation_decimal_places = annotation_decimal_places

        # Validate settings
        self._validate_settings()

        # Setup plot style
        self._setup_plot_style()

        # Print configuration
        self._print_configuration()

    #############################################################################################################
    # METHODS

    # Validate that settings are valid.
    def _validate_settings(self) -> None:
        valid_formats = ['png', 'tiff', 'tif', 'svg', 'pdf']
        if self.output_format not in valid_formats:
            raise ValueError(f"output_format must be one of {valid_formats}, got '{self.output_format}'")

        valid_normalize = ['rows', 'columns', 'none']
        if self.normalize not in valid_normalize:
            raise ValueError(f"normalize must be one of {valid_normalize}, got '{self.normalize}'")

        if self.raster_dpi < 150:
            print(f"  ⚠️ Warning: {self.raster_dpi} DPI is low for publication. 300 DPI minimum recommended.")

        if self.fixed_axes_height <= 0:
            raise ValueError(f"fixed_axes_height must be positive, got {self.fixed_axes_height}")

        if self.annotation_decimal_places < 0:
            raise ValueError(f"annotation_decimal_places must be >= 0, got {self.annotation_decimal_places}")

    # Set publication-ready matplotlib defaults.
    def _setup_plot_style(self) -> None:
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': [self.font_family, 'Arial', 'Helvetica'],
            'font.size': self.tick_label_size,
            'axes.labelsize': self.axis_label_size,
            'axes.titlesize': self.title_font_size,
            'xtick.labelsize': self.tick_label_size,
            'ytick.labelsize': self.tick_label_size,
            'figure.dpi': self.raster_dpi,
            'savefig.dpi': self.raster_dpi,
            'axes.spines.top': False,
            'axes.spines.right': False
        })

    # Print current configuration.
    def _print_configuration(self) -> None:
        print("=" * 60)
        print("CONFUSION MATRIX PLOTTER CONFIGURATION")
        print("=" * 60)
        print(f"Input folder:        {self.input_folder}")
        print(f"Output folder:       {self.output_folder}")
        print(f"Output format:       {self.output_format.upper()}")
        if self.output_format in ['png', 'tiff', 'tif']:
            print(f"Resolution:          {self.raster_dpi} DPI")
        print(f"Normalization:       {self.normalize}")
        print(f"Show counts:         {self.show_counts}")
        print(f"Combined mode:       {self.combined_mode}")
        print(f"Matrix height:       {self.fixed_axes_height} inches (fixed)")
        if self.master_font_size is not None:
            print(f"Font sizes:          MASTER CONTROL = {self.master_font_size} points (all fonts)")
        else:
            print(f"Font sizes (points): axis={self.axis_label_size}, "
                  f"title={self.title_font_size}, ticks={self.tick_label_size}, "
                  f"annotations={self.annotation_font_size}, legend={self.legend_font_size}")
        print(f"X-tick rotation:     {self.xtick_rotation}°")
        print("=" * 60)

    # Load confusion matrix data from JSON file.
    def _load_json(self, json_path: Path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        cm = np.array(data['confusion_matrix'])
        classes = data['classes']
        class_acc = data.get('class_accuracy', {})
        overall_acc = data.get('overall_accuracy', None)

        if not class_acc:
            class_acc = {}
            for i, cls in enumerate(classes):
                correct = cm[i, i]
                total = np.sum(cm[i, :])
                class_acc[cls] = correct / total if total > 0 else 0

        if overall_acc is None:
            correct = np.trace(cm)
            total = np.sum(cm)
            overall_acc = correct / total if total > 0 else 0

        return cm, classes, class_acc, overall_acc

    # Normalize confusion matrix based on setting.
    def _normalize_matrix(self, cm: np.ndarray):
        if self.normalize == 'rows':
            row_sums = cm.sum(axis=1, keepdims=True)
            data = np.where(row_sums > 0, cm / row_sums, 0)
            fmt = '.3f'
            cbar_label = 'Recall'
        elif self.normalize == 'columns':
            col_sums = cm.sum(axis=0, keepdims=True)
            data = np.where(col_sums > 0, cm / col_sums, 0)
            fmt = '.3f'
            cbar_label = 'Precision'
        else:
            data = cm
            fmt = 'd'
            cbar_label = 'Count'

        return data, fmt, cbar_label

    # Format a number with controlled decimal places.
    def _format_number(self, value, fmt: str) -> str:
        if fmt == 'd':
            return str(int(value))
        else:
            format_str = f'{{:.{self.annotation_decimal_places}f}}'
            formatted = format_str.format(value)
            if '.' in formatted:
                formatted = formatted.rstrip('0').rstrip('.')
            return formatted

    # Create annotation text for each cell.
    def _create_annotation(self, cm: np.ndarray, data: np.ndarray, fmt: str):
        if fmt == 'd':
            annot = data.astype(int)
        elif self.show_counts:
            annot = np.empty_like(data, dtype=object)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    norm_val = data[i, j]
                    count_val = int(cm[i, j])
                    norm_str = self._format_number(norm_val, fmt)
                    annot[i, j] = f'{norm_str}\n({count_val})'
        else:
            annot = np.empty_like(data, dtype=object)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    annot[i, j] = self._format_number(data[i, j], fmt)

        return annot

    # Create class labels with optional accuracy display.
    def _get_class_labels(self, classes: list, class_acc: dict) -> list:
        if self.show_per_class_accuracy:
            labels = []
            for cls in classes:
                acc = class_acc.get(cls, 0)
                labels.append(f'{cls}\n({acc*100:.1f}%)')
            return labels
        return classes

    # Create title with optional overall accuracy.
    def _get_title(self, original_title: str = None) -> str:
        if not self.show_title:
            return None

        if self.show_overall_accuracy and hasattr(self, '_overall_acc'):
            base = original_title if original_title else 'Confusion Matrix'
            return f'{base}\nOverall Accuracy: {self._overall_acc*100:.1f}%'

        return original_title if original_title else 'Confusion Matrix'

    # Calculate figure size based on settings.
    def _calculate_figure_size(self, n_classes: int):
        if self.use_fixed_axes_height:
            axes_size = self.fixed_axes_height
            fig_width = self.left_margin + axes_size + self.right_margin
            fig_height = self.bottom_margin + axes_size + self.top_margin
            return fig_width, fig_height, axes_size, axes_size
        else:
            return self.figsize[0], self.figsize[1], self.figsize[0], self.figsize[1]

    # Create a single confusion matrix plot.
    def _plot_single(self, cm, classes, class_acc, overall_acc, output_path, title=None):
        self._overall_acc = overall_acc

        data, fmt, cbar_label = self._normalize_matrix(cm)
        annot = self._create_annotation(cm, data, fmt)
        class_labels = self._get_class_labels(classes, class_acc)

        fig_width, fig_height, axes_width, axes_height = self._calculate_figure_size(len(classes))

        fig = plt.figure(figsize=(fig_width, fig_height))
        left_pos = self.left_margin / fig_width
        bottom_pos = self.bottom_margin / fig_height
        width_pos = axes_width / fig_width
        height_pos = axes_height / fig_height
        ax = fig.add_axes([left_pos, bottom_pos, width_pos, height_pos])

        im = ax.imshow(data, cmap=self.cmap, aspect='auto')

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = annot[i, j]
                color = 'white' if data[i, j] > 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=self.annotation_font_size, color=color)

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=self.xtick_rotation, ha='right')
        ax.set_yticklabels(class_labels, rotation=self.ytick_rotation)

        if self.show_colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
            cbar.set_label(cbar_label, fontsize=self.legend_font_size)
            cbar.ax.tick_params(labelsize=self.tick_label_size)

        if self.show_axis_labels:
            ax.set_xlabel('Predicted Class')
            ax.set_ylabel('True Class')

        plot_title = self._get_title(title)
        if plot_title:
            ax.set_title(plot_title, fontweight='bold')

        self._save_figure(fig, output_path)
        plt.close()

        print(f"  Physical size: {fig_width:.2f} × {fig_height:.2f} inches")
        return output_path

    # Create side-by-side raw and normalized confusion matrices.
    def _plot_combined(self, cm, classes, class_acc, overall_acc, output_path):
        self._overall_acc = overall_acc

        axes_size = self.fixed_axes_height
        single_width = self.left_margin + axes_size + self.right_margin
        fig_width = single_width * 2 + 0.5
        fig_height = self.bottom_margin + axes_size + self.top_margin

        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

        class_labels = self._get_class_labels(classes, class_acc)

        # Plot 1: Raw counts
        data_raw = cm
        vmax_raw = np.max(data_raw)
        im1 = axes[0].imshow(data_raw, cmap=self.cmap, vmin=0, vmax=vmax_raw, aspect='auto')

        for i in range(data_raw.shape[0]):
            for j in range(data_raw.shape[1]):
                color = 'white' if data_raw[i, j] > vmax_raw / 2 else 'black'
                axes[0].text(j, i, str(int(data_raw[i, j])), ha='center', va='center',
                           fontsize=self.annotation_font_size, color=color)

        axes[0].set_xticks(np.arange(len(classes)))
        axes[0].set_yticks(np.arange(len(classes)))
        axes[0].set_xticklabels(classes, rotation=self.xtick_rotation, ha='right')
        axes[0].set_yticklabels(class_labels, rotation=self.ytick_rotation)

        if self.show_axis_labels:
            axes[0].set_xlabel('Predicted Class')
            axes[0].set_ylabel('True Class')
        axes[0].set_title('Raw Counts', fontweight='bold')

        if self.show_colorbar:
            cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.05)
            cbar1.set_label('Count', fontsize=self.legend_font_size)
            cbar1.ax.tick_params(labelsize=self.tick_label_size)

        # Plot 2: Normalized
        data_norm, _, cbar_label = self._normalize_matrix(cm)
        annot_norm = self._create_annotation(cm, data_norm, '.3f')

        im2 = axes[1].imshow(data_norm, cmap=self.cmap, vmin=0, vmax=1, aspect='auto')

        for i in range(data_norm.shape[0]):
            for j in range(data_norm.shape[1]):
                color = 'white' if data_norm[i, j] > 0.5 else 'black'
                axes[1].text(j, i, annot_norm[i, j], ha='center', va='center',
                           fontsize=self.annotation_font_size, color=color)

        axes[1].set_xticks(np.arange(len(classes)))
        axes[1].set_yticks(np.arange(len(classes)))
        axes[1].set_xticklabels(classes, rotation=self.xtick_rotation, ha='right')
        axes[1].set_yticklabels(class_labels, rotation=self.ytick_rotation)

        if self.show_axis_labels:
            axes[1].set_xlabel('Predicted Class', fontweight='bold')
            axes[1].set_ylabel('True Class', fontweight='bold')

        plot_title = self._get_title('Normalized')
        if plot_title:
            axes[1].set_title(plot_title, fontweight='bold')

        if self.show_colorbar:
            cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.05)
            cbar2.set_label(cbar_label, fontsize=self.legend_font_size)
            cbar2.ax.tick_params(labelsize=self.tick_label_size)

        self._save_figure(fig, output_path)
        plt.close()

        print(f"  Physical size: {fig_width:.2f} × {fig_height:.2f} inches")
        return output_path

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
            print(f"  ✓ Saved: {output_path} ({file_size_mb:.1f} MB)")
        else:
            print(f"  ✓ Saved: {output_path} ({file_size_kb:.0f} KB)")

    # Process a single JSON file.
    def process_file(self, json_filename: str, output_filename: str = None):
        json_path = Path(json_filename)
        if not json_path.is_absolute():
            json_path = self.input_folder / json_filename

        if not json_path.exists():
            print(f"✗ File not found: {json_path}")
            return None

        print(f"\n📁 {json_path.name}")

        try:
            cm, classes, class_acc, overall_acc = self._load_json(json_path)
            print(f"  Loaded {len(classes)} classes ({len(classes)}×{len(classes)} matrix)")
            print(f"  Overall accuracy: {overall_acc*100:.2f}%")

            if output_filename is None:
                base_name = json_path.stem
                if self.combined_mode:
                    output_filename = f"{base_name}_combined.{self.output_format}"
                else:
                    suffix = f"_{self.normalize}" if self.normalize != 'none' else "_raw"
                    output_filename = f"{base_name}_confusion{suffix}.{self.output_format}"

            if self.output_format == 'tiff':
                output_filename = output_filename.replace('.tiff', '.tif')

            output_path = self.output_folder / output_filename

            if self.combined_mode:
                self._plot_combined(cm, classes, class_acc, overall_acc, output_path)
            else:
                title = f"Confusion Matrix ({self.normalize.capitalize()})" if self.normalize != 'none' else "Confusion Matrix (Raw Counts)"
                self._plot_single(cm, classes, class_acc, overall_acc, output_path, title)

            return output_path

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Process all JSON files in the input folder.
    def process_all_files(self, file_pattern: str = "*.json") -> list:
        files = list(self.input_folder.glob(file_pattern))

        if not files:
            print(f"\nNo files found in {self.input_folder} matching '{file_pattern}'")
            return []

        print(f"\nFound {len(files)} files to process")

        results = []
        for json_file in files:
            result = self.process_file(json_file.name)
            if result:
                results.append(result)

        print(f"\n✅ Completed: {len(results)}/{len(files)} files processed")
        return results

    #############################################################################################################
    # CALL

    def __call__(self) -> list:
        return self.process_all_files()