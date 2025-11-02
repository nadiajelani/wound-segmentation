#!/usr/bin/env python3
"""
Capture Fold-wise Metrics During Cross-Validation Training

This script provides a callback class that you can add to your existing
cross-validation training to capture and save fold-wise metrics.

Usage:
    Add this callback to your training script's callbacks list.
"""

import json
import os
from typing import Dict, List
import numpy as np
from datetime import datetime

class FoldMetricsCallback:
    """
    Callback to capture fold-wise metrics during cross-validation training.
    """
    
    def __init__(self, fold_number: int, save_dir: str = "fold_metrics"):
        """
        Initialize the callback.
        
        Args:
            fold_number: Current fold number (0-4 for 5-fold CV)
            save_dir: Directory to save metrics
        """
        self.fold_number = fold_number
        self.save_dir = save_dir
        self.metrics_history = {
            'fold': fold_number,
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch: int, logs: Dict):
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number
            logs: Dictionary containing metrics
        """
        self.metrics_history['epochs'].append(epoch)
        
        # Extract metrics from logs
        for key in ['loss', 'dice_coefficient', 'iou_score', 'precision', 'recall', 'accuracy']:
            train_key = key
            val_key = f'val_{key}'
            
            if train_key in logs:
                self.metrics_history[f'train_{key}'].append(float(logs[train_key]))
            if val_key in logs:
                self.metrics_history[f'val_{key}'].append(float(logs[val_key]))
    
    def on_train_end(self, logs: Dict = None):
        """
        Called at the end of training for this fold.
        """
        # Save fold metrics
        filename = f"fold_{self.fold_number}_metrics.json"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"✅ Fold {self.fold_number} metrics saved to {filepath}")
    
    def get_best_metrics(self) -> Dict:
        """
        Get the best validation metrics for this fold.
        
        Returns:
            Dictionary with best metrics
        """
        best_metrics = {}
        
        # Find epoch with best validation loss
        if self.metrics_history['val_loss']:
            best_epoch_idx = np.argmin(self.metrics_history['val_loss'])
            
            for key in ['val_dice', 'val_iou', 'val_precision', 'val_recall', 'val_accuracy']:
                if key in self.metrics_history and self.metrics_history[key]:
                    best_metrics[key] = self.metrics_history[key][best_epoch_idx]
        
        return best_metrics

def collect_all_fold_metrics(save_dir: str = "fold_metrics") -> Dict:
    """
    Collect metrics from all fold files.
    
    Args:
        save_dir: Directory containing fold metric files
    
    Returns:
        Dictionary with all fold metrics
    """
    all_fold_metrics = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'accuracy': []
    }
    
    if not os.path.exists(save_dir):
        print(f"⚠️ Directory {save_dir} not found")
        return all_fold_metrics
    
    # Find all fold metric files
    fold_files = [f for f in os.listdir(save_dir) if f.startswith('fold_') and f.endswith('_metrics.json')]
    
    if not fold_files:
        print(f"⚠️ No fold metric files found in {save_dir}")
        return all_fold_metrics
    
    print(f"📊 Found {len(fold_files)} fold metric files")
    
    for fold_file in sorted(fold_files):
        filepath = os.path.join(save_dir, fold_file)
        
        try:
            with open(filepath, 'r') as f:
                fold_data = json.load(f)
            
            # Extract best validation metrics
            best_metrics = {}
            if fold_data['val_loss']:
                best_epoch_idx = np.argmin(fold_data['val_loss'])
                
                for key in ['val_dice', 'val_iou', 'val_precision', 'val_recall', 'val_accuracy']:
                    if key in fold_data and fold_data[key]:
                        best_metrics[key] = fold_data[key][best_epoch_idx]
            
            # Add to collection
            for metric, values in all_fold_metrics.items():
                val_key = f'val_{metric}'
                if val_key in best_metrics:
                    values.append(best_metrics[val_key])
            
            print(f"  ✅ {fold_file}: Best val_dice = {best_metrics.get('val_dice', 'N/A'):.3f}")
            
        except Exception as e:
            print(f"⚠️ Error reading {fold_file}: {e}")
    
    return all_fold_metrics

def calculate_fold_confidence_intervals(fold_metrics: Dict) -> Dict:
    """
    Calculate confidence intervals from fold metrics.
    
    Args:
        fold_metrics: Dictionary with fold-wise metrics
    
    Returns:
        Dictionary with confidence intervals
    """
    from scipy import stats
    
    results = {}
    
    for metric, values in fold_metrics.items():
        if len(values) < 2:
            continue
            
        values = np.array(values)
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        std_err = std / np.sqrt(n)
        
        # Calculate 95% confidence interval
        alpha = 0.05
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        margin_error = t_critical * std_err
        
        lower_bound = mean - margin_error
        upper_bound = mean + margin_error
        
        results[metric] = {
            'fold_values': values.tolist(),
            'mean': mean,
            'std': std,
            'lower_ci': lower_bound,
            'upper_ci': upper_bound,
            'n_folds': n
        }
    
    return results

def main():
    """
    Main function to collect and analyze fold metrics.
    """
    print("🔬 Fold Metrics Collector and Analyzer")
    print("=" * 40)
    
    # Collect fold metrics
    fold_metrics = collect_all_fold_metrics()
    
    if not any(fold_metrics.values()):
        print("\n💡 To capture fold metrics during training:")
        print("1. Add FoldMetricsCallback to your training script")
        print("2. Run cross-validation training")
        print("3. Re-run this script to analyze results")
        
        # Show example integration
        print("\n📝 Example integration in your training script:")
        print("""
# In your train_model function, add this callback:
from capture_fold_metrics import FoldMetricsCallback

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    # ... your existing code ...
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint(f"{model_save_path}_fold{fold}", save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        FoldMetricsCallback(fold_number=fold),  # Add this line
        TqdmCallback(verbose=1)
    ]
        """)
        return
    
    # Calculate confidence intervals
    print("\n📈 Calculating confidence intervals...")
    results = calculate_fold_confidence_intervals(fold_metrics)
    
    # Display results
    print("\n📊 Fold-wise Results:")
    print("=" * 25)
    
    for metric, data in results.items():
        print(f"\n{metric.upper()}:")
        print(f"  Fold values: {[f'{v:.3f}' for v in data['fold_values']]}")
        print(f"  Mean: {data['mean']:.3f}")
        print(f"  Std:  {data['std']:.3f}")
        print(f"  95% CI: [{data['lower_ci']:.3f}, {data['upper_ci']:.3f}]")
    
    # Save results
    output_file = 'fold_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("✅ Fold analysis complete!")

if __name__ == "__main__":
    main()

