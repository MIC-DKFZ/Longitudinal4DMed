from setuptools import setup, find_packages

setup(
    name="Longi4DMed",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=open("requirements.txt").read().splitlines(),
)
