import pandas as pd
import numpy as np
from pathlib import Path

def select_best_checkpoints(metrics_files, output_dir, top_k=3):
    """
    Selects checkpoints with best balanced accuracy.
    
    Args:
        metrics_files: List of paths to CSV files containing validation metrics
        output_dir: Directory to save selected checkpoints
        top_k: Number of best checkpoints to select
    """
    # Load all metrics
    all_metrics = []
    for file in metrics_files:
        df = pd.read_csv(file)
        df['file_path'] = file
        all_metrics.append(df)
    
    metrics_df = pd.concat(all_metrics)
    
    # Calculate balance score (you can adjust this formula)
    metrics_df['balance_score'] = (
        metrics_df['ko_accuracy'] + metrics_df['wt_accuracy'] -  # sum of accuracies
        np.abs(metrics_df['ko_accuracy'] - metrics_df['wt_accuracy'])  # penalty for imbalance
    )
    
    # Sort by balance score
    sorted_df = metrics_df.sort_values('balance_score', ascending=False)
    
    # Select top k checkpoints
    best_checkpoints = sorted_df.head(top_k)
    
    # Save selection
    best_checkpoints.to_csv(Path(output_dir) / 'best_checkpoints.csv', index=False)
    
    return best_checkpoints