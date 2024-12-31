"""Script for packaging the vect_gan package."""

from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="vect_gan",
    version="0.1.1",
    description="A package for generating synthetic molecular tabular data using VECT-GAN.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Youssef Abdalla",
    author_email="youssef.abdalla.16@ucl.ac.uk",
    url="https://github.com/y-babdalla/vect_gan",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["vect_gan/gen_model/checkpoints/*"]},
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
