# Introduction

Repository of SECOND (Sparsely Embedded CONvolutional Detection), a Deep Learning model for 3D object detection.

This project provides a detailed implementation of the state-of-the-art architecture as explained in the [article](https://www.mdpi.com/1424-8220/18/10/3337).

A major feature of the project is the presence of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) for preprocessing point cloud data into voxels.


### Summary

The whole pipeline passes through different phases starting from a raw point cloud to make detection of eventual objects and regress their bounding box.

### Classes

The defined classes are the following:

| **Class**          | **Label** |
|--------------------|-----------|
| Electrical Cabinet | 64        |
| Technical Room     | 65        |
| GSM-R Antenna      | 66        |
| Track Telephone    | 67        |

The target objects are the aforementioned telecommunication assets.


# Getting Started

The repository is structured as follows:
```
├── docs                        generated documentation
├── evaluation                  python modules
├── notebooks                   jupyter notebooks
├── second                      source folder with SECOND's code
├── test                        unit / integration tests
├── .pre-commit-config.yaml     pre-commit installation file
├── conda_environment.yml       environment's dependencies file
├── pyproject.toml              project settings
├── pytest.ini                  pytest configuration
├── README.md                   readme file
└── setup.py                    python setup file
```

### Evaluation
````
evaluation/
├── training_plots   : training plots losses
└── weights          : stored best weights of the model
````

### Notebooks
````
notebooks/
├── clusters.ipynb   : clustering techniques to obtain the best anchors
└── test.ipynb       : testing module to simulate model training
````

### SECOND
````
second/
├── dataset          : module for data loading
├── model            : main module with SECOND's source code
├── testing          : module for testing function
├── training         : module for training function
├── utils            : module for some utility functions
└── __init__.py      : init module for relative import
````

#### DataLoader
````
dataset/
├── __init__.py      : init module for relative import
└── paris_orleans.py : loaders for clouds/voxels with their labels
````

#### Model
````
model/
├── layers           : step-by-step implementation of SECOND's layers
├── losses           : implementation of SECOND's loss function
├── preprocessing    : voxelization preprocessor using SPCONV library
├── scheduler        : implementation of a special scheduler for SECOND
├── __init__.py      : init module for relative import
└── second.py        : implementation of SECOND gathering its layers
````

#### Training
````
training/
└── train.py         : training function
````

#### Utilities
````
utils/
├── __init__.py      : init module for relative import
├── config.py        : hyperparameters for label encoding
└── model_saver.py   : function for saving the best weights
````

# Build and Test

1. Installation process

Clone the repository:
- `git clone https://github.com/medengessia/second.pytorch.git`

Use Conda to install the dependencies:
- `conda env create --file conda_environment.yml`

Activate the conda environment
- `conda activate second`

Local (editable) installation of the package
- `pip install -e .`

Configure pre-commit
- `pre-commit install`

Run tests on the database
- `pytest test`

Run tests
- `pytest`

# Contribute

Please feel free to raise issues whenever some of the aforementioned guidelines are inefficient or when there are some improvements to do. Pull Requests are welcome as well.

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)
