#!/usr/bin/env python
"""
Setup script for neurobiomorphic-ai package.

This package implements advanced neurobiomorphic AI systems with 
first-principles reasoning capabilities.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip() 
            for line in f.readlines() 
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = []

setup(
    name="neurobiomorphic-ai",
    version="0.1.0",
    author="Neurobiomorphic AI Team",
    author_email="team@neurobiomorphic-ai.com",
    description="Advanced production-grade neurobiomorphic AI system with first-principles reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aslonv/neurobiomorphic-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11", 
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9,<3.13",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "isort>=5.13.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.6.0",
        ],
        "profiling": [
            "line_profiler>=4.1.0",
            "memory_profiler>=0.61.0",
            "psutil>=5.9.0",
        ],
        "visualization": [
            "matplotlib>=3.8.0",
            "seaborn>=0.13.0", 
            "tensorboard>=2.15.0",
        ],
        "optimization": [
            "optuna>=3.4.0",
            "ray[tune]>=2.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neurobiomorphic-train=neurobiomorphic.cli:train",
            "neurobiomorphic-eval=neurobiomorphic.cli:evaluate",
            "neurobiomorphic-config=neurobiomorphic.cli:config",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
