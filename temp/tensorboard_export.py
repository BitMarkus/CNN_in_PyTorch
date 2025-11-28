import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns
from typing import Dict, List, Optional, Union
import json

class TensorBoardExporter:
    """
    A comprehensive class for exporting and analyzing TensorBoard log data.
    Provides methods to load, plot, export, and compare training metrics.
    """
    
    def __init__(self, log_dir: Union[str, Path]):
        """
        Initialize the TensorBoardExporter with a log directory.
        
        Args:
            log_dir: Path to the TensorBoard log directory
        """
        self.log_dir = Path(log_dir)
        self.data = None
        self._verify_log_dir()
    
    def _verify_log_dir(self):
        """Verify that the log directory contains TensorBoard event files."""
        event_files = list(self.log_dir.glob("events.out.tfevents.*"))
        if not event_files:
            raise ValueError(f"No TensorBoard event files found in {self.log_dir}")
        print(f"Found {len(event_files)} event file(s) in {self.log_dir}")
    
    def load_data(self, reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load TensorBoard logs into a dictionary of pandas DataFrames.
        
        Args:
            reload: If True, reload data even if already loaded
            
        Returns:
            Dictionary mapping metric names to DataFrames
        """
        if self.data is not None and not reload:
            return self.data
        
        # Create event accumulator
        event_acc = EventAccumulator(str(self.log_dir))
        event_acc.Reload()
        
        # Get all available tags (metric names)
        tags = event_acc.Tags()['scalars']
        print(f"Available metrics: {tags}")
        
        # Convert to pandas DataFrames
        self.data = {}
        for tag in tags:
            try:
                events = event_acc.Scalars(tag)
                if not events:
                    continue
                    
                steps = [e.step for e in events]
                values = [e.value for e in events]
                wall_times = [e.wall_time for e in events]
                
                self.data[tag] = pd.DataFrame({
                    'step': steps,
                    'value': values,
                    'wall_time': wall_times,
                    'epoch': [s + 1 for s in steps]  # Assuming steps are epochs
                })
                
                print(f"Loaded {len(events)} data points for {tag}")
                
            except Exception as e:
                print(f"Warning: Could not load metric {tag}: {e}")
        
        return self.data
    
    def get_metric_names(self) -> List[str]:
        """Get list of all available metric names."""
        if self.data is None:
            self.load_data()
        return list(self.data.keys())
    
    def get_metric_data(self, metric_name: str) -> pd.DataFrame:
        """Get DataFrame for a specific metric."""
        if self.data is None:
            self.load_data()
        
        if metric_name not in self.data:
            available = self.get_metric_names()
            raise ValueError(f"Metric '{metric_name}' not found. Available metrics: {available}")
        
        return self.data[metric_name]
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get the final values of all metrics."""
        if self.data is None:
            self.load_data()
        
        final_metrics = {}
        for metric_name, df in self.data.items():
            if len(df) > 0:
                final_metrics[metric_name] = df['value'].iloc[-1]
        
        return final_metrics
    
    def plot_comprehensive_metrics(self, output_dir: Optional[Union[str, Path]] = None, 
                                 show_plot: bool = True) -> plt.Figure:
        """
        Create comprehensive plots from TensorBoard logs.
        
        Args:
            output_dir: Directory to save plots (optional)
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if self.data is None:
            self.load_data()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Accuracy Plot
        if 'Accuracy/train' in self.data and 'Accuracy/val' in self.data:
            ax = axes[0]
            ax.plot(self.data['Accuracy/train']['epoch'], self.data['Accuracy/train']['value'], 
                   'g-', label='Training', linewidth=2, alpha=0.8)
            ax.plot(self.data['Accuracy/val']['epoch'], self.data['Accuracy/val']['value'], 
                   'r-', label='Validation', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training vs Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Loss Plot
        if 'Loss/train' in self.data and 'Loss/val' in self.data:
            ax = axes[1]
            ax.plot(self.data['Loss/train']['epoch'], self.data['Loss/train']['value'], 
                   'g-', label='Training', linewidth=2, alpha=0.8)
            ax.plot(self.data['Loss/val']['epoch'], self.data['Loss/val']['value'], 
                   'r-', label='Validation', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training vs Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Learning Rate
        if 'Metrics/LR' in self.data:
            ax = axes[2]
            ax.plot(self.data['Metrics/LR']['epoch'], self.data['Metrics/LR']['value'], 
                   'b-', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        # 4. F1 Scores
        if 'Metrics/F1/Macro' in self.data and 'Metrics/F1/Weighted' in self.data:
            ax = axes[3]
            ax.plot(self.data['Metrics/F1/Macro']['epoch'], self.data['Metrics/F1/Macro']['value'], 
                   'b-', label='Macro', linewidth=2, alpha=0.8)
            ax.plot(self.data['Metrics/F1/Weighted']['epoch'], self.data['Metrics/F1/Weighted']['value'], 
                   'orange', label='Weighted', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Score')
            ax.set_title('F1 Scores')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        # 5. AUC and AP
        if 'Metrics/AUC' in self.data or 'Metrics/AP' in self.data:
            ax = axes[4]
            if 'Metrics/AUC' in self.data:
                ax.plot(self.data['Metrics/AUC']['epoch'], self.data['Metrics/AUC']['value'], 
                       'purple', label='AUC', linewidth=2, alpha=0.8)
            if 'Metrics/AP' in self.data:
                ax.plot(self.data['Metrics/AP']['epoch'], self.data['Metrics/AP']['value'], 
                       'red', label='AP', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title('AUC and Average Precision')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        # 6. GPU Memory
        if 'System/GPU_Memory' in self.data:
            ax = axes[5]
            ax.plot(self.data['System/GPU_Memory']['epoch'], 
                   self.data['System/GPU_Memory']['value'] * 100, 
                   'red', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('GPU Memory Usage (%)')
            ax.set_title('GPU Memory Usage')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / "comprehensive_metrics.png", dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / "comprehensive_metrics.pdf", bbox_inches='tight')
            print(f"Plots saved to {output_dir}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def export_to_csv(self, output_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        Export all TensorBoard metrics to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
            
        Returns:
            Dictionary of DataFrames
        """
        if self.data is None:
            self.load_data()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export individual metrics
        for metric_name, df in self.data.items():
            # Clean metric name for filename
            filename = metric_name.replace('/', '_') + '.csv'
            df.to_csv(output_dir / filename, index=False)
            print(f"Exported {metric_name} to {filename}")
        
        # Create a summary CSV
        summary_data = []
        for metric_name, df in self.data.items():
            if len(df) > 0:
                summary_data.append({
                    'metric': metric_name,
                    'final_value': df['value'].iloc[-1],
                    'max_value': df['value'].max(),
                    'min_value': df['value'].min(),
                    'mean_value': df['value'].mean(),
                    'std_value': df['value'].std(),
                    'num_epochs': len(df),
                    'first_epoch': df['epoch'].min(),
                    'last_epoch': df['epoch'].max()
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'training_summary.csv', index=False)
        print(f"Exported training summary to training_summary.csv")
        
        # Export final metrics as JSON for easy reading
        final_metrics = self.get_final_metrics()
        with open(output_dir / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Exported final metrics to final_metrics.json")
        
        return self.data
    
    def plot_metric(self, metric_name: str, output_dir: Optional[Union[str, Path]] = None,
                   show_plot: bool = True) -> plt.Figure:
        """
        Plot a single metric.
        
        Args:
            metric_name: Name of the metric to plot
            output_dir: Directory to save plot (optional)
            show_plot: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        df = self.get_metric_data(metric_name)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['epoch'], df['value'], 'b-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title(f'{metric_name} over Time')
        ax.grid(True, alpha=0.3)
        
        # Add final value annotation
        final_value = df['value'].iloc[-1]
        ax.annotate(f'Final: {final_value:.4f}', 
                   xy=(df['epoch'].iloc[-1], final_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = metric_name.replace('/', '_') + '.png'
            plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_dir / filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return fig

    @classmethod
    def compare_runs(cls, log_dirs: List[Union[str, Path]], 
                    run_names: Optional[List[str]] = None,
                    output_dir: Optional[Union[str, Path]] = None,
                    metrics: List[str] = None) -> plt.Figure:
        """
        Compare multiple training runs.
        
        Args:
            log_dirs: List of log directories to compare
            run_names: Names for each run (optional)
            output_dir: Directory to save comparison plot (optional)
            metrics: Specific metrics to compare (default: common metrics)
            
        Returns:
            matplotlib Figure object
        """
        if run_names is None:
            run_names = [f"Run_{i+1}" for i in range(len(log_dirs))]
        
        if metrics is None:
            metrics = ['Accuracy/val', 'Loss/val', 'Metrics/F1/Weighted', 'Metrics/AUC']
        
        # Load all runs
        exporters = [cls(log_dir) for log_dir in log_dirs]
        for exporter in exporters:
            exporter.load_data()
        
        # Create comparison plot
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(log_dirs)))
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            for j, (exporter, run_name, color) in enumerate(zip(exporters, run_names, colors)):
                if metric in exporter.data:
                    df = exporter.data[metric]
                    ax.plot(df['epoch'], df['value'], 
                           color=color, label=run_name, linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / "run_comparison.png", dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / "run_comparison.pdf", bbox_inches='tight')
            print(f"Comparison plot saved to {output_dir}")
        
        plt.show()
        return fig

    def generate_report(self, output_dir: Union[str, Path]):
        """
        Generate a comprehensive report with all plots and data.
        
        Args:
            output_dir: Directory to save the report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Generating comprehensive report...")
        
        # 1. Load data
        self.load_data()
        
        # 2. Export all data to CSV
        self.export_to_csv(output_dir / "csv_data")
        
        # 3. Generate comprehensive plots
        self.plot_comprehensive_metrics(output_dir / "plots", show_plot=False)
        
        # 4. Generate individual metric plots
        individual_plot_dir = output_dir / "plots" / "individual"
        individual_plot_dir.mkdir(exist_ok=True)
        
        for metric_name in self.get_metric_names():
            try:
                self.plot_metric(metric_name, individual_plot_dir, show_plot=False)
            except Exception as e:
                print(f"Could not plot {metric_name}: {e}")
        
        # 5. Print summary
        final_metrics = self.get_final_metrics()
        print("\n" + "="*50)
        print("TRAINING REPORT SUMMARY")
        print("="*50)
        for metric, value in final_metrics.items():
            print(f"{metric:25}: {value:.4f}")
        
        print(f"\nReport generated in: {output_dir}")

# Usage examples:
if __name__ == "__main__":
    # Example 1: Basic usage
    exporter = TensorBoardExporter("logs/your_training_run")
    exporter.generate_report("analysis_results")
    
    # Example 2: Compare multiple runs
    TensorBoardExporter.compare_runs(
        log_dirs=["logs/run1", "logs/run2", "logs/run3"],
        run_names=["Baseline", "With Augmentation", "Weighted Loss"],
        output_dir="comparison_results"
    )
    
    # Example 3: Custom analysis
    exporter = TensorBoardExporter("logs/your_training_run")
    data = exporter.load_data()
    final_metrics = exporter.get_final_metrics()
    
    # Plot specific metric
    exporter.plot_metric("Accuracy/val", output_dir="custom_plots")