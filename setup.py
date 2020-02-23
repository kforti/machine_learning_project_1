import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="assignment1", # Replace with your own username
    version="0.0.1",
    author="Kevin-Brian-Ryan",
    description="Assignment 1",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/kforti/machine_learning_project_1",
    packages=setuptools.find_packages(),
    install_requires=[
        'flann',
        'sklearn',
        'numpy',
        'pandas',
        'matplotlib',
        'jupyter'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)