#!/usr/bin/env python3
"""
Test runner for wound segmentation system.

This script provides a convenient way to run all tests with different
configurations and options.
"""

import sys
import os
import argparse
from pathlib import Path
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_tests(test_type="all", verbose=False, coverage=False, markers=None):
    """
    Run tests with specified configuration.
    
    Args:
        test_type: Type of tests to run (all, unit, integration, golden)
        verbose: Enable verbose output
        coverage: Generate coverage report
        markers: Pytest markers to include/exclude
    """
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    cmd.append("tests/")
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add coverage
    if coverage:
        cmd.append("--cov=woundseg")
        cmd.append("--cov-report=html")
        cmd.append("--cov-report=term")
    
    # Add markers
    if markers:
        cmd.extend(["-m", markers])
    
    # Add test type specific options
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "golden":
        cmd.extend(["-m", "golden"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    
    # Add other useful options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings for cleaner output
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run wound segmentation tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_tests.py                    # Run all tests
  python tests/run_tests.py --type unit        # Run only unit tests
  python tests/run_tests.py --type integration # Run only integration tests
  python tests/run_tests.py --type golden      # Run only golden tests
  python tests/run_tests.py --type fast        # Run fast tests only
  python tests/run_tests.py --type slow        # Run slow tests only
  python tests/run_tests.py --coverage         # Run with coverage
  python tests/run_tests.py --verbose          # Verbose output
  python tests/run_tests.py --markers "not slow" # Custom markers
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["all", "unit", "integration", "golden", "fast", "slow"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--markers", "-m",
        help="Pytest markers to include/exclude (e.g., 'not slow')"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies before running tests"
    )
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "pytest", "pytest-cov", "pytest-mock", "pytest-html"
            ], check=True)
            print("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies: {e}")
            return 1
    
    # Run tests
    return run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage,
        markers=args.markers
    )


if __name__ == "__main__":
    sys.exit(main())