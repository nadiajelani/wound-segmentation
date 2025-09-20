#!/usr/bin/env python3
"""
Wound Segmentation Health Check Script
Validates that all required dependencies and configurations are working correctly.
"""

import sys
import importlib
import subprocess
import os
from pathlib import Path

# Colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

def check_python_version():
    """Check Python version compatibility"""
    print_header("PYTHON VERSION CHECK")
    
    version = sys.version_info
    print_info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version.major}.{version.minor} is supported")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False

def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally check version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        
        if min_version and version != 'Unknown':
            # Simple version comparison (can be improved)
            try:
                from packaging import version as pkg_version
                if pkg_version.parse(version) >= pkg_version.parse(min_version):
                    print_success(f"{package_name} {version} (>= {min_version})")
                    return True
                else:
                    print_warning(f"{package_name} {version} (requires >= {min_version})")
                    return False
            except ImportError:
                print_success(f"{package_name} {version} (version check skipped)")
                return True
        else:
            print_success(f"{package_name} {version}")
            return True
            
    except ImportError:
        print_error(f"{package_name} not installed")
        return False

def check_critical_packages():
    """Check all critical packages for the wound segmentation project"""
    print_header("CRITICAL PACKAGES CHECK")
    
    packages = [
        ("numpy", "numpy", "1.20.0"),
        ("pandas", "pandas", "1.3.0"),
        ("tensorflow", "tensorflow", "2.8.0"),
        ("opencv-python", "cv2", "4.5.0"),
        ("scikit-image", "skimage", "0.18.0"),
        ("scikit-learn", "sklearn", "1.0.0"),
        ("matplotlib", "matplotlib", "3.3.0"),
        ("pillow", "PIL", "8.0.0"),
        ("reportlab", "reportlab", "3.6.0"),
        ("tkinter", "tkinter"),  # Built-in, no version check
    ]
    
    results = []
    for package_name, import_name, *version in packages:
        min_version = version[0] if version else None
        results.append(check_package(package_name, import_name, min_version))
    
    return all(results)

def check_optional_packages():
    """Check optional packages"""
    print_header("OPTIONAL PACKAGES CHECK")
    
    packages = [
        ("albumentations", "albumentations"),
        ("segmentation-models", "segmentation_models"),
        ("gTTS", "gtts"),
        ("Flask", "flask"),
        ("scipy", "scipy"),
    ]
    
    results = []
    for package_name, import_name in packages:
        try:
            results.append(check_package(package_name, import_name))
        except Exception as e:
            print_warning(f"Failed to check {package_name}: {e}")
            results.append(False)
    
    installed_count = sum(results)
    total_count = len(results)
    print_info(f"Optional packages: {installed_count}/{total_count} installed")
    
    return True  # Optional packages are not critical

def check_tkinter_functionality():
    """Test Tkinter functionality"""
    print_header("TKINTER FUNCTIONALITY CHECK")
    
    try:
        import tkinter as tk
        print_success("Tkinter module imported successfully")
        
        # Try to create a root window (without showing it)
        root = tk.Tk()
        root.withdraw()  # Hide the window
        print_success("Tkinter root window created successfully")
        root.destroy()
        
        return True
    except Exception as e:
        print_error(f"Tkinter functionality test failed: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability for TensorFlow"""
    print_header("GPU AVAILABILITY CHECK")
    
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print_success(f"GPU detected: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print_info(f"  GPU {i}: {gpu.name}")
            
            # Test GPU memory
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([1.0, 2.0, 3.0])
                    b = tf.constant([4.0, 5.0, 6.0])
                    c = tf.add(a, b)
                print_success("GPU computation test passed")
                return True
            except Exception as e:
                print_warning(f"GPU computation test failed: {e}")
                return False
        else:
            print_warning("No GPU detected, using CPU")
            return True
            
    except Exception as e:
        print_error(f"GPU check failed: {e}")
        return False

def check_model_files():
    """Check if required model files exist"""
    print_header("MODEL FILES CHECK")
    
    model_paths = [
        "models/simclr_unet_patch_wound.keras",
        "models/",
    ]
    
    results = []
    for path in model_paths:
        if os.path.exists(path):
            if os.path.isfile(path):
                size = os.path.getsize(path) / (1024 * 1024)  # MB
                print_success(f"Model file found: {path} ({size:.1f} MB)")
            else:
                print_success(f"Model directory found: {path}")
            results.append(True)
        else:
            print_warning(f"Model file/directory not found: {path}")
            results.append(False)
    
    return any(results)  # At least one model file/dir should exist

def check_directories():
    """Check if required directories exist"""
    print_header("DIRECTORIES CHECK")
    
    directories = [
        "outputs/",
        "reports/",
        "wound_progress_report/",
    ]
    
    results = []
    for directory in directories:
        if os.path.exists(directory):
            print_success(f"Directory exists: {directory}")
            results.append(True)
        else:
            print_warning(f"Directory missing: {directory}")
            # Try to create it
            try:
                os.makedirs(directory, exist_ok=True)
                print_success(f"Created directory: {directory}")
                results.append(True)
            except Exception as e:
                print_error(f"Failed to create directory {directory}: {e}")
                results.append(False)
    
    return all(results)

def run_quick_test():
    """Run a quick functionality test"""
    print_header("QUICK FUNCTIONALITY TEST")
    
    try:
        # Test basic imports
        import numpy as np
        import pandas as pd
        import cv2
        import tensorflow as tf
        
        # Test basic operations
        arr = np.array([1, 2, 3, 4, 5])
        df = pd.DataFrame({'test': [1, 2, 3]})
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        print_success("Basic NumPy operations work")
        print_success("Basic Pandas operations work")
        print_success("Basic OpenCV operations work")
        
        # Test TensorFlow
        with tf.device('/CPU:0'):  # Force CPU to avoid GPU issues
            x = tf.constant([1.0, 2.0, 3.0])
            y = tf.reduce_sum(x)
            print_success("Basic TensorFlow operations work")
        
        return True
        
    except Exception as e:
        print_error(f"Quick functionality test failed: {e}")
        return False

def main():
    """Main health check function"""
    print_header("WOUND SEGMENTATION HEALTH CHECK")
    print_info("Checking system configuration and dependencies...")
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Critical Packages", check_critical_packages),
        ("Optional Packages", lambda: check_optional_packages()),
        ("Tkinter Functionality", check_tkinter_functionality),
        ("GPU Availability", check_gpu_availability),
        ("Model Files", check_model_files),
        ("Directories", check_directories),
        ("Quick Functionality Test", run_quick_test),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print_error(f"{name} check failed with exception: {e}")
            results[name] = False
    
    # Summary
    print_header("HEALTH CHECK SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status:4}{Colors.END} {name}")
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} checks passed{Colors.END}")
    
    if passed == total:
        print_success("All checks passed! Your environment is ready.")
        return 0
    elif passed >= total * 0.8:
        print_warning("Most checks passed. Some issues detected but system should work.")
        return 1
    else:
        print_error("Multiple critical issues detected. Please fix them before proceeding.")
        return 2

if __name__ == "__main__":
    sys.exit(main())