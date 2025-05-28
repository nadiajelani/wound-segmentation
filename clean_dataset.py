import os
import shutil
from PIL import Image
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = '/Users/nadiajelani/Desktop/wounds-whisperer/wounds/dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
QUARANTINE_DIR = os.path.join(BASE_DIR, 'quarantine')  # Where to move problematic files
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Supported image formats

# Create quarantine directory
os.makedirs(QUARANTINE_DIR, exist_ok=True)

def scan_directory(directory, subset_name):
    """Scan a directory for images and identify problematic files."""
    logger.info("Scanning %s directory: %s", subset_name, directory)
    problematic_files = []
    total_files = 0
    valid_files = 0

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_files += 1

            # Skip files with unsupported extensions
            if not file.lower().endswith(SUPPORTED_EXTENSIONS):
                logger.warning("Unsupported file found: %s", file_path)
                problematic_files.append((file_path, "Unsupported extension"))
                continue

            # Try to open the image
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify the image is valid
                    img.close()
                    valid_files += 1
            except Exception as e:
                logger.error("Failed to load image %s: %s", file_path, str(e))
                problematic_files.append((file_path, str(e)))

    logger.info("Found %d total files, %d valid images in %s", total_files, valid_files, subset_name)
    return problematic_files

def quarantine_files(problematic_files):
    """Move problematic files to the quarantine directory."""
    if not problematic_files:
        logger.info("No problematic files to quarantine")
        return

    for file_path, reason in problematic_files:
        try:
            relative_path = os.path.relpath(file_path, BASE_DIR)
            quarantine_path = os.path.join(QUARANTINE_DIR, relative_path)
            os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)
            shutil.move(file_path, quarantine_path)
            logger.info("Moved problematic file to %s (Reason: %s)", quarantine_path, reason)
        except Exception as e:
            logger.error("Failed to move file %s: %s", file_path, str(e))

def main():
    logger.info("Starting dataset cleaning at %s", datetime.now().strftime("%I:%M %p %Z, %B %d, %Y"))
    
    # Scan train and validation directories
    train_issues = scan_directory(TRAIN_DIR, "train")
    val_issues = scan_directory(VALIDATION_DIR, "validation")
    
    # Combine problematic files
    all_issues = train_issues + val_issues
    
    # Quarantine problematic files
    quarantine_files(all_issues)
    
    logger.info("Dataset cleaning completed at %s", datetime.now().strftime("%I:%M %p %Z, %B %d, %Y"))
    if all_issues:
        logger.info("Problematic files were moved to %s. Please review them.", QUARANTINE_DIR)
    else:
        logger.info("No problematic files found. Dataset is clean.")

if __name__ == "__main__":
    main()
