# Wound Segmentation CLI

Professional command-line interface for wound analysis and training.

## Quick Start

### Basic Analysis
```bash
./ws analyze /path/to/wound/image.jpg
```

### Full Analysis with Reports
```bash
./ws analyze /path/to/wound/image.jpg --name "John Doe" --age 45 --report --voice
```

### System Information
```bash
./ws info
```

## Commands

### `analyze` - Wound Analysis
Analyze a wound image and generate comprehensive reports.

**Usage:**
```bash
ws analyze [OPTIONS] IMAGE
```

**Options:**
- `--use-medsam` - Use MedSAM model instead of U-Net
- `--report/--no-report` - Generate PDF reports (default: report)
- `--voice/--no-voice` - Generate voice summaries (default: voice)
- `--name TEXT` - Patient name
- `--age INTEGER` - Patient age
- `--output-dir PATH` - Custom output directory
- `--confidence FLOAT` - Confidence threshold (0.0-1.0, default: 0.5)
- `--verbose, -v` - Verbose output

**Examples:**
```bash
# Basic analysis
ws analyze wound.jpg

# Full analysis with patient info
ws analyze wound.jpg --name "Jane Smith" --age 30 --report --voice

# High confidence analysis
ws analyze wound.jpg --confidence 0.8

# Use MedSAM model
ws analyze wound.jpg --use-medsam
```

### `train-unet` - U-Net Training
Train a U-Net model for wound segmentation.

**Usage:**
```bash
ws train-unet [OPTIONS] DATA_DIR
```

**Options:**
- `--epochs INTEGER` - Number of training epochs (default: 100)
- `--batch-size INTEGER` - Training batch size (default: 8)
- `--lr FLOAT` - Learning rate (default: 0.001)
- `--output-dir PATH` - Model output directory
- `--verbose, -v` - Verbose output

### `train-classifier` - Classifier Training
Train a classifier model for wound classification.

**Usage:**
```bash
ws train-classifier [OPTIONS] DATA_DIR
```

**Options:**
- `--epochs INTEGER` - Number of training epochs (default: 50)
- `--batch-size INTEGER` - Training batch size (default: 16)
- `--lr FLOAT` - Learning rate (default: 0.001)
- `--output-dir PATH` - Model output directory
- `--verbose, -v` - Verbose output

### `info` - System Information
Display system information and configuration.

**Usage:**
```bash
ws info
```

### `version` - Version Information
Display version information.

**Usage:**
```bash
ws version
```

## Output Structure

All analysis results are saved to the `outputs/` directory with the following structure:

```
outputs/
├── sessions/
│   └── YYYYMMDD_HHMMSS/          # Session directory
├── uploads/                       # Original images
├── masks/                         # Segmentation masks
├── visualizations/                # Overlays, heatmaps, trends
├── reports/                       # PDF reports
└── voice_summaries/               # Audio summaries
```

## Features

### Analysis Features
- **AI-Powered Segmentation**: Uses SIMCLR U-Net or MedSAM models
- **Comprehensive Metrics**: Wound area, perimeter, condition assessment
- **Multiple Output Formats**: Images, PDFs, audio, CSV data
- **Professional Reports**: Patient and clinician versions
- **Voice Summaries**: Text-to-speech audio reports

### CLI Features
- **Rich Interface**: Beautiful terminal output with progress indicators
- **Flexible Options**: Customizable analysis parameters
- **Error Handling**: Clear error messages and validation
- **Help System**: Comprehensive help and documentation
- **Verbose Mode**: Detailed output for debugging

## Configuration

The CLI respects all configuration settings from the `woundseg.config` module:

- **Feature Flags**: `ENABLE_VOICE_SUMMARY`, `ENABLE_MEDSAM`, etc.
- **Model Paths**: Automatic model discovery
- **Output Directory**: Configurable output location
- **Device Settings**: Automatic device selection

## Examples

### Basic Workflow
```bash
# Check system info
ws info

# Analyze a wound
ws analyze /path/to/wound.jpg --name "Patient Name" --age 35

# View results
ls outputs/
```

### Advanced Usage
```bash
# High-confidence analysis with MedSAM
ws analyze wound.jpg --use-medsam --confidence 0.9 --verbose

# Analysis without voice/reports (faster)
ws analyze wound.jpg --no-voice --no-report

# Custom output directory
ws analyze wound.jpg --output-dir /custom/path
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure model files are in the correct directory
2. **Voice service unavailable**: Check gTTS installation and `ENABLE_VOICE_SUMMARY` setting
3. **Permission errors**: Ensure write permissions for output directory
4. **Image format issues**: Supported formats: JPG, PNG, BMP, TIFF

### Getting Help

```bash
# General help
ws --help

# Command-specific help
ws analyze --help
ws train-unet --help

# Verbose output for debugging
ws analyze image.jpg --verbose
```

## Integration

The CLI integrates with all Stage 5 services:

- **Storage Service**: Organized file management
- **Voice Service**: Text-to-speech generation
- **Reporting Service**: PDF report creation
- **Validation Service**: Input/output validation
- **Explainability Service**: Model interpretability

## Development

### Adding New Commands

1. Create command module in `cli/commands/`
2. Import and register in `cli/ws_cli.py`
3. Add help documentation
4. Test with various options

### Extending Analysis

The CLI uses the modular pipeline system:
- `AnalysisPipeline` for orchestration
- `ModelProvider` for model management
- Service modules for specialized functionality