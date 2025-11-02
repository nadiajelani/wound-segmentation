#!/usr/bin/env python3
"""
Confidence Interval Calculator for Model Performance Metrics

This script calculates confidence intervals for Dice and IoU metrics
without modifying your existing working codebase.

Usage:
    python calculate_confidence_intervals.py

The script will:
1. Generate sample performance data (replace with your actual results)
2. Calculate 95% confidence intervals for Dice and IoU
3. Output formatted results for your paper
"""

import numpy as np
import pandas as pd
from scipy import stats
import json
from typing import Dict, List, Tuple
import os

def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for a given dataset.
    
    Args:
        data: Array of metric values
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # Standard error of the mean
    
    # Calculate confidence interval
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_error = t_critical * std_err
    
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    
    return mean, lower_bound, upper_bound

def bootstrap_confidence_interval(data: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 1000) -> Tuple[float, float, float]:
    """
    Calculate confidence interval using bootstrap method.
    
    Args:
        data: Array of metric values
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    mean = np.mean(data)
    
    # Calculate percentiles for confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return mean, lower_bound, upper_bound

def generate_sample_performance_data(n_samples: int = 50) -> Dict[str, np.ndarray]:
    """
    Generate sample performance data for demonstration.
    Replace this with your actual validation results.
    
    Args:
        n_samples: Number of validation samples
    
    Returns:
        Dictionary with metric arrays
    """
    # Sample data based on typical wound segmentation performance
    # Replace these with your actual validation results
    
    # Dice scores (typically 0.7-0.9 for good wound segmentation)
    dice_scores = np.random.normal(0.82, 0.08, n_samples)
    dice_scores = np.clip(dice_scores, 0, 1)  # Ensure valid range
    
    # IoU scores (typically 0.6-0.8 for good wound segmentation)
    iou_scores = np.random.normal(0.72, 0.09, n_samples)
    iou_scores = np.clip(iou_scores, 0, 1)  # Ensure valid range
    
    # Precision (typically 0.8-0.95)
    precision_scores = np.random.normal(0.88, 0.06, n_samples)
    precision_scores = np.clip(precision_scores, 0, 1)
    
    # Recall (typically 0.75-0.9)
    recall_scores = np.random.normal(0.84, 0.07, n_samples)
    recall_scores = np.clip(recall_scores, 0, 1)
    
    # Accuracy (typically 0.9-0.98)
    accuracy_scores = np.random.normal(0.94, 0.03, n_samples)
    accuracy_scores = np.clip(accuracy_scores, 0, 1)
    
    return {
        'dice': dice_scores,
        'iou': iou_scores,
        'precision': precision_scores,
        'recall': recall_scores,
        'accuracy': accuracy_scores
    }

def load_actual_results_if_available() -> Dict[str, np.ndarray]:
    """
    Try to load actual validation results from CSV files.
    Returns sample data if no actual results found.
    """
    # Check for common validation result files
    possible_files = [
        'unet_simclr_validation_results.csv',
        'medsam_validation_results.csv',
        'validation_results.csv',
        'model_performance.csv'
    ]
    
    for filename in possible_files:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                print(f"✅ Found validation results: {filename}")
                
                results = {}
                if 'Dice Score' in df.columns:
                    results['dice'] = df['Dice Score'].dropna().values
                if 'IoU Score' in df.columns:
                    results['iou'] = df['IoU Score'].dropna().values
                if 'Precision' in df.columns:
                    results['precision'] = df['Precision'].dropna().values
                if 'Recall' in df.columns:
                    results['recall'] = df['Recall'].dropna().values
                if 'Accuracy' in df.columns:
                    results['accuracy'] = df['Accuracy'].dropna().values
                
                if results:
                    return results
            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")
    
    print("📊 No validation results found, using sample data for demonstration")
    return generate_sample_performance_data()

def format_metric_with_ci(mean: float, lower: float, upper: float, decimals: int = 3) -> str:
    """
    Format metric with confidence interval for LaTeX table.
    
    Args:
        mean: Mean value
        lower: Lower bound of CI
        upper: Upper bound of CI
        decimals: Number of decimal places
    
    Returns:
        Formatted string for LaTeX
    """
    return f"{mean:.{decimals}f} ({lower:.{decimals}f}-{upper:.{decimals}f})"

def format_percentage_with_ci(mean: float, lower: float, upper: float, decimals: int = 1) -> str:
    """
    Format percentage metric with confidence interval for LaTeX table.
    
    Args:
        mean: Mean value (0-1 range)
        lower: Lower bound of CI
        upper: Upper bound of CI
        decimals: Number of decimal places
    
    Returns:
        Formatted string for LaTeX
    """
    mean_pct = mean * 100
    lower_pct = lower * 100
    upper_pct = upper * 100
    return f"{mean_pct:.{decimals}f}\\% ({lower_pct:.{decimals}f}-{upper_pct:.{decimals}f})"

def main():
    """
    Main function to calculate confidence intervals and generate results.
    """
    print("🔬 Confidence Interval Calculator for Model Performance Metrics")
    print("=" * 60)
    
    # Load performance data
    performance_data = load_actual_results_if_available()
    
    # Calculate confidence intervals for each metric
    results = {}
    methods = ['t_test', 'bootstrap']
    
    for method in methods:
        print(f"\n📊 Calculating {method} confidence intervals...")
        results[method] = {}
        
        for metric_name, values in performance_data.items():
            if len(values) < 2:
                print(f"⚠️ Skipping {metric_name}: insufficient data")
                continue
                
            if method == 't_test':
                mean, lower, upper = calculate_confidence_interval(values)
            else:  # bootstrap
                mean, lower, upper = bootstrap_confidence_interval(values)
            
            results[method][metric_name] = {
                'mean': mean,
                'lower': lower,
                'upper': upper,
                'n_samples': len(values)
            }
            
            print(f"  {metric_name.upper()}: {mean:.3f} [{lower:.3f}, {upper:.3f}] (n={len(values)})")
    
    # Generate LaTeX table format
    print("\n📝 LaTeX Table Format:")
    print("=" * 40)
    
    # Use t-test results (more standard for academic papers)
    t_results = results['t_test']
    
    latex_table = """
\\begin{table}[h]
\\centering
\\caption{Model Performance Metrics with 95\\% Confidence Intervals}
\\begin{tabular}{|l|c|}
\\hline
\\textbf{Metric} & \\textbf{Value (95\\% CI)} \\\\
\\hline"""
    
    if 'accuracy' in t_results:
        acc_str = format_percentage_with_ci(
            t_results['accuracy']['mean'],
            t_results['accuracy']['lower'],
            t_results['accuracy']['upper']
        )
        latex_table += f"\nAccuracy & {acc_str} \\\\"
    
    if 'dice' in t_results:
        dice_str = format_metric_with_ci(
            t_results['dice']['mean'],
            t_results['dice']['lower'],
            t_results['dice']['upper']
        )
        latex_table += f"\nDice Coefficient & {dice_str} \\\\"
    
    if 'iou' in t_results:
        iou_str = format_metric_with_ci(
            t_results['iou']['mean'],
            t_results['iou']['lower'],
            t_results['iou']['upper']
        )
        latex_table += f"\nIoU (Intersection over Union) & {iou_str} \\\\"
    
    if 'precision' in t_results:
        prec_str = format_percentage_with_ci(
            t_results['precision']['mean'],
            t_results['precision']['lower'],
            t_results['precision']['upper']
        )
        latex_table += f"\nPrecision & {prec_str} \\\\"
    
    if 'recall' in t_results:
        rec_str = format_percentage_with_ci(
            t_results['recall']['mean'],
            t_results['recall']['lower'],
            t_results['recall']['upper']
        )
        latex_table += f"\nRecall & {rec_str} \\\\"
    
    latex_table += """
\\hline
\\end{tabular}
\\end{table}"""
    
    print(latex_table)
    
    # Save results to JSON for reference
    output_file = 'confidence_interval_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Generate summary statistics
    print("\n📈 Summary Statistics:")
    print("=" * 30)
    for metric_name, values in performance_data.items():
        print(f"{metric_name.upper()}:")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std:  {np.std(values):.4f}")
        print(f"  Min:  {np.min(values):.4f}")
        print(f"  Max:  {np.max(values):.4f}")
        print(f"  N:    {len(values)}")
        print()
    
    print("✅ Confidence interval calculation complete!")
    print("\n📋 Next steps:")
    print("1. Copy the LaTeX table above into your paper")
    print("2. Replace sample data with your actual validation results")
    print("3. Run your validation scripts to generate real performance data")

if __name__ == "__main__":
    main()

