from setuptools import setup, find_packages

setup(
    name="ai-vs-real-classifier",
    version="0.0.1",
    description="Multi-Model AI vs Real Image Classifier",
    author="Allkeeey",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "timm",
        "python-dotenv",
        "pydantic",
        "pyyaml",
        "fastapi",
        "uvicorn",
        "streamlit",
        "mlflow",
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    python_requires=">=3.14",
)