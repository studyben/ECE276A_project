from setuptools import setup, find_packages

setup(
    name="ECE2761_Project1",
    version="1.0.0",
    packages=find_packages(include=['bin_detection.*', 'pixel_classification.*'])
)