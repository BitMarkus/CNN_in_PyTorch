# Diffusion-Based Phenotypic Extrapolation
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: MIT

# ===== Standard Library Imports =====
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
# ===== Third-Party Imports =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
# ===== Own Modules =====
from settings import setting

class UMAPPlotter:
   
    # Publication-ready dimensionality reduction plotter with consistent axis heights.
    # All plots will have the same physical y-axis height and font sizes,
    # while width adjusts automatically based on data aspect ratio.

    #############################################################################################################
    # CONSTRUCTOR

    # Initialize the plotter with all configuration options.
    # Args:
    #   input_folder (Path): Folder containing CSV files
    #   output_folder (Path): Folder for output plots
    #   output_format (str): 'png', 'tiff', 'svg', 'pdf'
    #   raster_dpi (int): DPI for raster formats
    #   axes_height_inches (float): Physical height of the y-axis in inches
    #   fixed_aspect_ratio (bool): True = preserve data aspect ratio, False = stretch to fit
    #   point_size (int): Size of each point in points (1/72 inch)
    #   point_alpha (float): Transparency: 0=invisible, 1=opaque
    #   show_ellipses (bool): Show 95% confidence ellipses
    #   ellipse_alpha (float): Ellipse transparency
    #   palette (str): Color scheme
    #   show_legend (bool): Show/hide legend
    #   legend_position (str): 'auto', 'inside', 'outside', or location string
    #   legend_marker_size (int): Legend marker size in points
    #   font_family (str): Font family
    #   axis_label_size (int): Font size for axis labels
    #   legend_font_size (int): Font size for legend text
    #   tick_label_size (int): Font size for tick labels
    #   show_grid (bool): Show background grid
    #   grid_alpha (float): Grid line transparency
    #   method_name (str, optional): Manual override for method name
    #   left_margin (float): Space for y-axis label and tick labels (inches)
    #   right_margin (float): Space on right side of axes (inches)
    #   bottom_margin (float): Space for x-axis label and tick labels (inches)
    #   top_margin (float): Space above axes (inches)
    #   legend_margin_extra (float): Extra right margin when legend is outside (inches)
    def __init__(
        self,
        input_folder=None,
        output_folder=None,
        output_format='tiff',
        raster_dpi=600,
        axes_height_inches=5.0,
        fixed_aspect_ratio=True,
        point_size=30,
        point_alpha=0.5,
        show_ellipses=False,
        ellipse_alpha=0.15,
        palette='jet',
        show_legend=True,
        legend_position='outside',
        legend_marker_size=100,
        font_family='Arial',
        axis_label_size=22,
        legend_font_size=22,
        tick_label_size=22,
        show_grid=True,
        grid_alpha=0.3,
        method_name=None,
        left_margin=1.2,
        right_margin=1.0,
        bottom_margin=0.9,
        top_margin=0.5,
        legend_margin_extra=3.0
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

        # Axis size settings
        self.axes_height_inches = axes_height_inches
        self.fixed_aspect_ratio = fixed_aspect_ratio

        # Point rendering settings
        self.point_size = point_size
        self.point_alpha = point_alpha

        # Ellipse settings
        self.show_ellipses = show_ellipses
        self.ellipse_alpha = ellipse_alpha

        # Color settings
        self.palette = palette

        # Legend settings
        self.show_legend = show_legend
        self.legend_position = legend_position
        self.legend_marker_size = legend_marker_size

        # Font settings
        self.font_family = font_family
        self.axis_label_size = axis_label_size
        self.legend_font_size = legend_font_size
        self.tick_label_size = tick_label_size

        # Grid settings
        self.show_grid = show_grid
        self.grid_alpha = grid_alpha

        # Method settings
        self.method_name = method_name

        # Margin settings
        self.left_margin = left_margin
        self.right_margin = right_margin
        self.bottom_margin = bottom_margin
        self.top_margin = top_margin
        self.legend_margin_extra = legend_margin_extra

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

        if self.raster_dpi < 150:
            print(f"  ⚠️ Warning: {self.raster_dpi} DPI is low for publication. 300 DPI minimum recommended.")

        if self.axes_height_inches <= 0:
            raise ValueError(f"axes_height_inches must be positive, got {self.axes_height_inches}")

        if self.point_alpha < 0 or self.point_alpha > 1:
            raise ValueError(f"point_alpha must be between 0 and 1, got {self.point_alpha}")

    # Set publication-ready matplotlib defaults.
    def _setup_plot_style(self) -> None:
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': [self.font_family, 'Arial', 'Helvetica', 'DejaVu Sans'],
            'font.size': self.tick_label_size,
            'axes.labelsize': self.axis_label_size,
            'axes.titlesize': self.axis_label_size,
            'axes.titleweight': 'normal',
            'legend.fontsize': self.legend_font_size,
            'legend.title_fontsize': self.legend_font_size + 1,
            'legend.handletextpad': 0.3,
            'legend.handlelength': 1.0,
            'legend.borderaxespad': 0.5,
            'xtick.labelsize': self.tick_label_size,
            'ytick.labelsize': self.tick_label_size,
            'figure.dpi': self.raster_dpi,
            'savefig.dpi': self.raster_dpi,
            'savefig.bbox': 'standard',
            'axes.spines.top': False,
            'axes.spines.right': False
        })

    # Print current configuration.
    def _print_configuration(self) -> None:
        print("=" * 60)
        print("DIMENSIONALITY REDUCTION PLOTTER CONFIGURATION")
        print("=" * 60)
        print(f"Input folder:        {self.input_folder}")
        print(f"Output folder:       {self.output_folder}")
        print(f"Output format:       {self.output_format.upper()}")
        if self.output_format in ['png', 'tiff', 'tif']:
            print(f"Resolution:          {self.raster_dpi} DPI")
        print(f"Axes height:         {self.axes_height_inches} inches (CONSISTENT across plots)")
        print(f"Fixed aspect ratio:  {self.fixed_aspect_ratio}")
        print(f"Point size:          {self.point_size} points")
        print(f"Point alpha:         {self.point_alpha}")
        print(f"Show ellipses:       {self.show_ellipses}")
        print(f"Palette:             {self.palette}")
        print(f"Show legend:         {self.show_legend}")
        if self.show_legend:
            print(f"Legend position:     {self.legend_position}")
        print(f"Font sizes (points): axis={self.axis_label_size}, "
              f"legend={self.legend_font_size}, ticks={self.tick_label_size}")
        print("=" * 60)

    # Detect the dimensionality reduction method from filename.
    def _detect_method(self, filename: str) -> str:
        if self.method_name:
            return self.method_name

        filename_lower = filename.lower()

        if 'pacmap' in filename_lower:
            return 'PaCMAP'
        elif 'tsne' in filename_lower or 't-sne' in filename_lower:
            return 't-SNE'
        elif 'trimap' in filename_lower or 'tri-map' in filename_lower:
            return 'TriMAP'
        elif 'umap' in filename_lower:
            return 'UMAP'
        else:
            print(f"  ⚠️ Could not detect method from filename '{filename}'. Defaulting to 'UMAP'.")
            return 'UMAP'

    # Get colors for groups.
    def _get_colors(self, group_names: list) -> dict:
        n_groups = len(group_names)

        try:
            cmap = plt.cm.get_cmap(self.palette)
            colors = cmap(np.linspace(0, 1, n_groups))
        except (ValueError, AttributeError):
            try:
                colors = sns.color_palette(self.palette, n_colors=n_groups)
            except (ValueError, AttributeError):
                print(f"  Warning: Palette/colormap '{self.palette}' not found. Using 'colorblind' instead.")
                colors = sns.color_palette('colorblind', n_colors=n_groups)

        return {group: colors[i] for i, group in enumerate(sorted(group_names))}

    # Add 95% confidence ellipse to axes.
    def _add_confidence_ellipse(self, ax, x: np.ndarray, y: np.ndarray, color, n_std: float = 2.0) -> None:
        if len(x) < 5:
            return

        cov = np.cov(x, y)
        if np.linalg.det(cov) <= 0:
            return

        lambda_, v = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
        width, height = 2 * np.sqrt(n_std * lambda_)

        ellipse = Ellipse(
            xy=(np.mean(x), np.mean(y)),
            width=width, height=height, angle=angle,
            facecolor=color, edgecolor='none',
            alpha=self.ellipse_alpha, zorder=0
        )
        ax.add_patch(ellipse)

    # Load and validate embedding CSV file.
    def _load_data(self, csv_path: Path):
        df = pd.read_csv(csv_path)

        dim1_candidates = [col for col in df.columns if 'dim1' in col or col.endswith('_1')]
        dim2_candidates = [col for col in df.columns if 'dim2' in col or col.endswith('_2')]

        if not dim1_candidates or not dim2_candidates:
            raise ValueError(f"No dimension columns found. Available: {list(df.columns)}")

        dim1 = dim1_candidates[0]
        dim2 = dim2_candidates[0]

        df = df.rename(columns={dim1: 'dim1', dim2: 'dim2'})

        if 'label_name' not in df.columns:
            if 'label_numeric' in df.columns:
                df['label_name'] = df['label_numeric'].astype(str)
            else:
                raise ValueError("No 'label_name' or 'label_numeric' column found")

        df = df.dropna(subset=['dim1', 'dim2', 'label_name'])

        return df

    # Calculate figure size based on fixed axes height and data aspect ratio.
    def _calculate_figure_size(self, df, has_outside_legend: bool = False):
        # Calculate data ranges
        x_min, x_max = df['dim1'].min(), df['dim1'].max()
        y_min, y_max = df['dim2'].min(), df['dim2'].max()
        x_range = x_max - x_min
        y_range = y_max - y_min

        if y_range == 0:
            y_range = 1
        if x_range == 0:
            x_range = 1

        data_aspect = x_range / y_range

        print(f"  Data ranges: X [{x_min:.2f}, {x_max:.2f}] (width={x_range:.2f}), "
              f"Y [{y_min:.2f}, {y_max:.2f}] (height={y_range:.2f})")
        print(f"  Data aspect ratio (width/height): {data_aspect:.3f}")

        axes_height = self.axes_height_inches

        if self.fixed_aspect_ratio:
            axes_width = axes_height * data_aspect
        else:
            axes_width = axes_height

        print(f"  Axes size: {axes_width:.2f} inches wide × {axes_height:.2f} inches tall")

        legend_width = 0
        if has_outside_legend:
            n_groups = len(df['label_name'].unique())
            max_label_length = max([len(str(label)) for label in df['label_name'].unique()])
            legend_width = 0.6 + (max_label_length * 0.08)
            legend_width = max(legend_width, 0.8)
            print(f"  Estimated legend width: {legend_width:.2f} inches")

        extra_right = legend_width if has_outside_legend else 0

        fig_width = self.left_margin + axes_width + self.right_margin + extra_right
        fig_height = self.bottom_margin + axes_height + self.top_margin

        print(f"  Figure size: {fig_width:.2f} × {fig_height:.2f} inches")

        return fig_width, fig_height, axes_width, axes_height, legend_width

    # Setup axes at the correct position within the figure.
    def _setup_axes(self, fig, axes_width: float, axes_height: float, legend_width: float = 0):
        total_width = self.left_margin + axes_width + self.right_margin + legend_width
        total_height = self.bottom_margin + axes_height + self.top_margin

        left_pos = self.left_margin / total_width
        bottom_pos = self.bottom_margin / total_height
        width_pos = axes_width / total_width
        height_pos = axes_height / total_height

        ax = fig.add_axes([left_pos, bottom_pos, width_pos, height_pos])

        return ax

    # Create plot with consistent axis height.
    def _create_plot(self, df, output_filename: str, method_name: str = "UMAP"):
        output_path = self.output_folder / output_filename
        if not output_path.suffix:
            output_path = output_path.with_suffix(f'.{self.output_format}')

        groups = sorted(df['label_name'].unique())
        n_groups = len(groups)
        color_dict = self._get_colors(groups)

        has_outside_legend = False
        if self.show_legend:
            if self.legend_position == 'outside':
                has_outside_legend = True
            elif self.legend_position == 'auto' and n_groups > 8:
                has_outside_legend = True

        fig_width, fig_height, axes_width, axes_height, legend_width = self._calculate_figure_size(
            df, has_outside_legend
        )

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.raster_dpi)

        ax = self._setup_axes(fig, axes_width, axes_height, legend_width)

        use_raster = False
        if self.output_format in ['svg', 'pdf']:
            use_raster = True
        elif self.output_format in ['png', 'tiff', 'tif']:
            use_raster = True

        for group in groups:
            subset = df[df['label_name'] == group]
            color = color_dict.get(group)

            ax.scatter(
                subset['dim1'], subset['dim2'],
                c=color, s=self.point_size, alpha=self.point_alpha,
                label=group if self.show_legend else None,
                edgecolors='none',
                rasterized=use_raster,
                linewidth=0
            )

            if self.show_ellipses and color is not None:
                self._add_confidence_ellipse(ax, subset['dim1'], subset['dim2'], color)

        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=self.axis_label_size)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=self.axis_label_size)
        ax.tick_params(axis='both', labelsize=self.tick_label_size)

        if self.fixed_aspect_ratio:
            ax.set_aspect('equal')
        else:
            ax.set_aspect('auto')

        ax.autoscale(enable=True, tight=True)

        if self.show_grid:
            ax.grid(True, linestyle='--', alpha=self.grid_alpha, linewidth=0.5)
            ax.set_axisbelow(True)

        if self.show_legend:
            self._add_legend(ax, groups, n_groups, legend_width)

        save_kwargs = {
            'dpi': self.raster_dpi,
            'bbox_inches': None,
            'pad_inches': 0
        }

        if self.output_format in ['png', 'jpg', 'jpeg']:
            plt.savefig(output_path, **save_kwargs)
        elif self.output_format in ['tiff', 'tif']:
            plt.savefig(output_path, format='tiff', **save_kwargs)
        else:
            plt.savefig(output_path, format=self.output_format, **save_kwargs)

        plt.close()

        file_size_kb = output_path.stat().st_size / 1024
        file_size_mb = file_size_kb / 1024

        if file_size_mb >= 1:
            print(f"  ✓ Saved: {output_path} ({file_size_mb:.1f} MB)")
        else:
            print(f"  ✓ Saved: {output_path} ({file_size_kb:.0f} KB)")

        print(f"  Physical size: {fig_width:.2f} × {fig_height:.2f} inches "
              f"(axes: {axes_width:.2f} × {axes_height:.2f} inches)")

        return output_path

    # Add legend to the plot.
    def _add_legend(self, ax, groups: list, n_groups: int, legend_width: float = 0) -> None:
        legend_font_size = self.legend_font_size
        if n_groups > 10:
            legend_font_size = max(6, self.legend_font_size - 2)

        if self.legend_position == 'auto':
            if n_groups <= 8:
                legend = ax.legend(loc='best', frameon=True, fancybox=False,
                                   facecolor='white', edgecolor='#CCCCCC',
                                   framealpha=0.9, fontsize=legend_font_size)
            else:
                legend = ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
                                   frameon=True, facecolor='white', edgecolor='#CCCCCC',
                                   framealpha=0.9, fontsize=legend_font_size,
                                   borderaxespad=0)
        elif self.legend_position == 'inside':
            legend = ax.legend(loc='best', frameon=True, fancybox=False,
                               facecolor='white', edgecolor='#CCCCCC',
                               framealpha=0.9, fontsize=legend_font_size)
        elif self.legend_position == 'outside':
            legend = ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
                               frameon=True, facecolor='white', edgecolor='#CCCCCC',
                               framealpha=0.9, fontsize=legend_font_size,
                               borderaxespad=0)
        else:
            legend = ax.legend(loc=self.legend_position, frameon=True, fancybox=False,
                               facecolor='white', edgecolor='#CCCCCC',
                               framealpha=0.9, fontsize=legend_font_size)

        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_linestyle('--')

        if hasattr(legend, 'legend_handles'):
            legend_handles = legend.legend_handles
        else:
            legend_handles = legend.legendHandles

        for handle in legend_handles:
            handle.set_sizes([self.legend_marker_size])

    # Process a single CSV file.
    def process_file(self, csv_filename: str, output_filename: str = None):
        csv_path = Path(csv_filename)
        if not csv_path.is_absolute():
            csv_path = self.input_folder / csv_filename

        if not csv_path.exists():
            print(f"✗ File not found: {csv_path}")
            return None

        print(f"\n📁 {csv_path.name}")

        try:
            df = self._load_data(csv_path)
            print(f"  Loaded {len(df):,} points")
            print(f"  Groups: {df['label_name'].nunique()}")

            method_name = self._detect_method(csv_path.name)
            print(f"  Method: {method_name}")

            if output_filename is None:
                output_filename = f"{csv_path.stem}.{self.output_format}"

            if self.output_format == 'tiff':
                output_filename = output_filename.replace('.tiff', '.tif')
                if not output_filename.endswith('.tif'):
                    output_filename = output_filename.replace(f'.{self.output_format}', '.tif')

            return self._create_plot(df, output_filename, method_name)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Process all CSV files in the input folder.
    def process_all_files(self, file_pattern: str = "*.csv") -> list:
        files = list(self.input_folder.glob(file_pattern))

        if not files:
            print(f"\nNo files found in {self.input_folder} matching '{file_pattern}'")
            print("Please place your CSV files in the input folder.")
            return []

        print(f"\nFound {len(files)} files to process")

        results = []
        for csv_file in files:
            result = self.process_file(csv_file.name)
            if result:
                results.append(result)

        print(f"\n✅ Completed: {len(results)}/{len(files)} files processed")
        return results

    #############################################################################################################
    # CALL

    def __call__(self) -> list:
        return self.process_all_files()