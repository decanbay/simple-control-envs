from setuptools import setup, find_packages

setup(
    name="control_envs",
    version="0.0.4",
    packages=find_packages(),
    install_requires=[
        "gymnasium",  # Changed from "gym"
        "numpy",
        "matplotlib"
    ],
    python_requires=">=3.7",
    description="Simple control environments for reinforcement learning",
    author="Deniz Ekin Canbay",
    author_email="decanbay@gmail.com",
)