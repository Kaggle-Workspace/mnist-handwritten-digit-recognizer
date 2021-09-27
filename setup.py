from setuptools import setup
from setuptools import find_packages
from tensorflow.python.ops.gen_control_flow_ops import enter


setup(
    name="mnist-handwritten-digit-recognizer",
    version="0.1",
    description="A test package",
    author="Kaustav Ghosh",
    author_email="teetangh@gmail.com",
    packages=find_packages(
        exclude=("tests*", "testing*", "examples*", "docs*", "build*")
    ),  # finds all folders with __init__.py
    install_requires=[
        "numpy", "pandas", "matplotlib",
        "tensorflow", "keras"],

    entry_points={
        "console_scripts": [
            "mnist-ann-cli=src.run_ann.main:main",
            "mnist-cnn-cli=src.run_cnn.main:main"
        ]
    },

)
