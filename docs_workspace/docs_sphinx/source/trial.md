# Trial

This section provides an introductory guide for begineers to get
 started with using **laygo2**.

## Colab tutorial

For individuals seeking to exercise the capabilities of **laygo2** 
without local installation, [Colab](https://colab.research.google.com/), 
a cloud-based Jupyter environment, can be utilized. 
The example tutorial can be accessed at this 
**[link](https://colab.research.google.com/drive/1tpuUvqb6BujzZI6RBf2cFdAfMqBsxpep?usp=sharing)**.

## Installation 

There are two available options for installing the **laygo2** package.

### 1. Cloning from **[Github repository](https://github.com/niftylab/laygo2.git)**

Users can obtain the latest version of laygo2 by cloning the Github 
repository by running the following command:

    >>> git clone https://github.com/niftylab/laygo2.git

It is highly recommended that the following command is used periodically to maintain the code to the latest version.

    >>> git pull origin master

After that, update the PHTHONPATH environment variable to point out the laygo2 package path for importing.

    # (csh/tcsh example) add the following command to your .cshrc
    setenv PYTHONPATH ${PYTHONPATH}:[LAYGO2_INSTALLATION_PATH]/laygo2

### 2. **[Pip](https://pypi.org/project/laygo2)** installation

Another option for installation is through the use of **[pip](https://pypi.org)**, the package manager for Python. This method allows for a 
covenient and quick installation process.
Run the following command to pip-install the **laygo2** package:

    >>> pip install laygo2

## Technology setup

The following files in the **laygo2_tech** directory need to be 
prepared for the usage of **laygo2** with a new technology:

    laygo2_tech_templayes.py  # contains the definitions for templates.
    laygo2_tech_grids.py      # contains the definitions for grids.
    laygo2_tech.yaml          # contains technology parameters.

A minimum technology setup, designed for for quick_start.py can be 
found **[here](https://github.com/niftylab/laygo2/tree/master/laygo2_tech_quick_start)**.
A more comprehensive tech setup example, designed for **gpdk045**,
 can be found **[here](https://github.com/niftylab/laygo2_workspace_gpdk045/tree/master/laygo2_tech_example)**.

## Simple gate generation

Running the following command will generate a NAND gate layout.

    # 1. Clone the laygo2 repository to your local machine.
    # 2. Navigate to the laygo2 directory.
    >>> cd laygo2 
    # 3. Run the quick start script by typing the following command:
    >>> python -m quick_start.py
    # 4. Alternatively, you can run ipython and type the following command:
    >>> run 'quick_start.py'
    
The resulting layout of the NAND gate is shown in the figure below:

![laygo2 nand gate](../assets/img/user_guide_nandgate.png "laygo2 NAND gate layout")

## Trial in SKY130 technology

Follow the tutorial available at the this 
**[link](https://laygo2-sky130-docs.readthedocs.io/en/latest/)**.
that covers the use of **laygo2** for generating a D flip-flop layout 
in the **[SKY130](https://skywater-pdk.readthedocs.io/en/main/)** 
technology.

The generated layout of a D flip-flop is shown in the figure below:

![sky130 dff2x](../assets/img/trial_sky130_dff.png "sky130 dff2x")

For those who prefer a Colab version, please check this **[link](https://colab.research.google.com/drive/1dToEQe7500TUNOPN2aPTJGRgcbbNsqhj?usp=sharing)**.

![sky130 dff2x colab](../assets/img/trial_sky130_dff_colab.png "sky130 dff2x colab")
