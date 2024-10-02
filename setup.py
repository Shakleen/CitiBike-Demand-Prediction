from setuptools import find_packages, setup
from typing import List


def get_requirements(file_name: str) -> List[str]:
    """This function will return a list of requirements.

    Args:
        file_name (str): Path of requirements.txt

    Returns:
        List[str]: List of required packages
    """
    requiments = None

    with open(file_name, "r") as file:
        requiments = [
            line.replace("\n", "")
            for line in file.readlines()
            if line.find("-e .") == -1
        ]

    return requiments


setup(
    name="CitiBike Demand Prediction",
    version="0.0.6",
    description="An End-to-End Machine Learning project where I predict demand of bikes at citibike stations at hourly level.",
    author="Shakleen Ishfar",
    author_email="shakleenishfar@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
