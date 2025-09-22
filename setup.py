#!/usr/bin/env python3
"""
Setup script for woundseg package.
This file provides backward compatibility with older pip versions.
For modern Python packaging, use pyproject.toml instead.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "AI-powered wound segmentation and analysis system for medical applications"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="woundseg",
    version="1.0.0",
    description="AI-powered wound segmentation and analysis system for medical applications",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Wound Segmentation Team",
    author_email="contact@woundseg.ai",
    url="https://github.com/woundseg/woundseg",
    project_urls={
        "Homepage": "https://github.com/woundseg/woundseg",
        "Documentation": "https://woundseg.readthedocs.io",
        "Repository": "https://github.com/woundseg/woundseg.git",
        "Issues": "https://github.com/woundseg/woundseg/issues",
        "Changelog": "https://github.com/woundseg/woundseg/blob/main/CHANGELOG.md",
    },
    packages=find_packages(include=["woundseg*", "cli*"]),
    package_data={
        "woundseg": [
            "config/*.yaml",
            "config/*.json",
            "models/*.json",
            "templates/*.html",
            "static/*",
        ]
    },
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        "train": [
            "albumentations>=1.3.0,<2.0.0",
            "tensorboard>=2.10.0,<3.0.0",
            "wandb>=0.13.0,<1.0.0",
            "tqdm>=4.64.0,<5.0.0",
        ],
        "explain": [
            "shap>=0.41.0,<1.0.0",
            "lime>=0.2.0,<1.0.0",
            "grad-cam>=1.4.0,<2.0.0",
        ],
        "synthetic": [
            "diffusers>=0.20.0,<1.0.0",
            "transformers>=4.25.0,<5.0.0",
            "accelerate>=0.20.0,<1.0.0",
            "torch>=1.13.0,<3.0.0",
            "torchvision>=0.14.0,<1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0,<8.0.0",
            "pytest-cov>=4.0.0,<5.0.0",
            "pytest-mock>=3.10.0,<4.0.0",
            "black>=22.0.0,<24.0.0",
            "isort>=5.10.0,<6.0.0",
            "flake8>=5.0.0,<7.0.0",
            "mypy>=1.0.0,<2.0.0",
            "pre-commit>=2.20.0,<4.0.0",
        ],
        "all": [
            "woundseg[train,explain,synthetic,dev]"
        ],
    },
    entry_points={
        "console_scripts": [
            "ws=cli.ws_cli:app",
        ],
        "gui_scripts": [
            "woundseg-gui=woundseg.gui:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords=[
        "medical-ai",
        "wound-segmentation",
        "computer-vision",
        "deep-learning",
        "healthcare",
        "tensorflow",
        "keras",
        "unet",
    ],
    zip_safe=False,
    platforms=["any"],
    license="MIT",
)