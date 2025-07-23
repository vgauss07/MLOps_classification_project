from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS-PROJECT-X",
    version="0.1",
    author="Jeffrey Voke",
    packages=find_packages(),
    install_requires=requirements,

)
