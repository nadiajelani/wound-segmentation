# Confidence Interval Integration Guide

## Overview
This guide explains how to integrate confidence interval calculations with your existing wound segmentation validation scripts without modifying your working codebase.

## What Was Created

### 1. Standalone Calculator Script
- **File**: `calculate_confidence_intervals.py`
- **Purpose**: Calculates 95% confidence intervals for Dice, IoU, and other metrics
- **Methods**: Both t-test and bootstrap confidence intervals
- **Output**: LaTeX-formatted table ready for your paper

### 2. Updated Paper Table
- **File**: `ieee_paper/paper.tex`
- **Change**: Updated performance metrics table to include confidence intervals
- **Format**: `Metric: Value (Lower-Upper)` e.g., `0.820 (0.800-0.839)`

## How to Use with Your Existing Validation Scripts

### Option 1: Run Validation Scripts First (Recommended)

1. **Run your existing validation scripts** to generate CSV results:
   ```bash
   # Run your U-Net validation
   python check_mask_unet.py
   
   # Run your MedSAM validation  
   python check_mask_medsam.py
   ```

2. **The calculator will automatically detect** the CSV files:
   - `unet_simclr_validation_results.csv`
   - `medsam_validation_results.csv`
   - Any CSV with columns: `Dice Score`, `IoU Score`, `Precision`, `Recall`, `Accuracy`

3. **Run the confidence interval calculator**:
   ```bash
   python calculate_confidence_intervals.py
   ```

### Option 2: Manual Integration (If you want to modify validation scripts)

Add this code to your existing validation scripts (e.g., `check_mask_unet.py`):

```python
# Add at the end of your validation script
from calculate_confidence_intervals import calculate_confidence_interval, format_metric_with_ci

# After saving your results CSV
if len(results) > 1:  # Need at least 2 samples for CI
    dice_values = [r["Dice Score"] for r in results if r["Dice Score"] != "Error"]
    iou_values = [r["IoU Score"] for r in results if r["IoU Score"] != "Error"]
    
    if dice_values and iou_values:
        dice_mean, dice_lower, dice_upper = calculate_confidence_interval(np.array(dice_values))
        iou_mean, iou_lower, iou_upper = calculate_confidence_interval(np.array(iou_values))
        
        print(f"\n📊 Confidence Intervals (95%):")
        print(f"Dice: {format_metric_with_ci(dice_mean, dice_lower, dice_upper)}")
        print(f"IoU:  {format_metric_with_ci(iou_mean, iou_lower, iou_upper)}")
```

## Understanding the Results

### Confidence Interval Interpretation
- **95% CI**: We are 95% confident that the true population mean lies within this range
- **Narrower CI**: More precise estimate (larger sample size or lower variance)
- **Wider CI**: Less precise estimate (smaller sample size or higher variance)

### Example Output
```
Dice Coefficient: 0.820 (0.800-0.839)
```
- **Mean**: 0.820 (best estimate)
- **95% CI**: 0.800 to 0.839 (range of plausible values)
- **Interpretation**: We're 95% confident the true Dice score is between 0.800 and 0.839

## Statistical Methods Used

### 1. T-Test Confidence Intervals (Default)
- **Formula**: `mean ± t_critical × standard_error`
- **Assumption**: Data is approximately normally distributed
- **Best for**: Academic papers, standard practice

### 2. Bootstrap Confidence Intervals
- **Method**: Resampling with replacement (1000 iterations)
- **Assumption**: Minimal assumptions about data distribution
- **Best for**: Non-normal data, robust estimates

## File Structure After Running

```
/Users/nadiajelani/projects/wound-segmentation/
├── calculate_confidence_intervals.py          # Main calculator
├── confidence_interval_results.json          # Detailed results
├── unet_simclr_validation_results.csv        # Your validation data
├── medsam_validation_results.csv             # Your validation data
└── ieee_paper/
    └── paper.tex                             # Updated with CI table
```

## Customization Options

### Change Confidence Level
```python
# In calculate_confidence_intervals.py, modify:
mean, lower, upper = calculate_confidence_interval(values, confidence=0.99)  # 99% CI
```

### Add More Metrics
```python
# Add to the performance_data dictionary:
results['f1_score'] = f1_values
results['specificity'] = specificity_values
```

### Different Output Format
```python
# Modify format_metric_with_ci() for different formatting:
# Current: "0.820 (0.800-0.839)"
# Custom:  "0.820 ± 0.020" (mean ± margin of error)
```

## Troubleshooting

### "No validation results found"
- **Cause**: No CSV files with expected column names
- **Solution**: Run your validation scripts first, or check CSV column names

### "Insufficient data" warning
- **Cause**: Less than 2 samples for a metric
- **Solution**: Increase your validation dataset size

### Confidence intervals too wide
- **Cause**: High variance in your metrics
- **Solution**: 
  - Increase sample size
  - Check for outliers in your data
  - Consider stratified sampling

## Academic Best Practices

### 1. Sample Size
- **Minimum**: 30 samples for reliable CI
- **Recommended**: 50+ samples for wound segmentation
- **Large studies**: 100+ samples for publication

### 2. Reporting Standards
- Always report both mean and CI
- Specify confidence level (95% is standard)
- Include sample size (n=X)
- Report method used (t-test vs bootstrap)

### 3. LaTeX Formatting
```latex
% Good format
Dice Coefficient & 0.820 (0.800-0.839) \\

% Also acceptable
Dice Coefficient & 0.820 ± 0.020 \\
```

## Next Steps

1. **Run your validation scripts** to get real performance data
2. **Execute the calculator** to get confidence intervals
3. **Copy the LaTeX table** into your paper
4. **Validate the results** make sense for your dataset
5. **Consider additional metrics** if needed (F1-score, specificity, etc.)

## Support

If you encounter issues:
1. Check that your validation CSV files have the expected column names
2. Ensure you have at least 2 samples per metric
3. Verify your data doesn't contain "Error" values
4. Check the generated `confidence_interval_results.json` for detailed statistics

The calculator is designed to work with your existing codebase without modifications, ensuring your working system remains unchanged.

