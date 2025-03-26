from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="syntht2i",
    version="0.1.0",
    author="Simo Ryu",
    author_email="simo@fal.ai",
    description="A synthetic text-to-image dataset generator for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fal-ai-community/alphabet-dataset",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "streamlit>=1.0.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=20.8b1",
            "isort>=5.6.0",
            "flake8>=3.8.0",
        ],
    },
)
