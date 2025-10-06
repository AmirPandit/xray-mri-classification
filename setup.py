from setuptools import setup, find_packages

setup(
    name="xray-mri-classification",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.3.0",
        "numpy>=1.21.0",
        "python-dotenv>=0.19.0",
    ],
)