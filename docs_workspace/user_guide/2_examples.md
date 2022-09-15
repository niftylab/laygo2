# laygo2 Quick Trial

This document introduces setup procedures and several design examples for begineers.

<!--
* **[Quick installation and setup](#Quick-Installation-and-Setup)** describes the installation and set up procedure of 
laygo in linux environments.
* **[Technology setup](#Technology-Setup)** illustrates how to set up laygo2 for new technology nodes.
* **[simple-gates](#Simple-Gates)**: introduces layout generators for simple logic gates.
-->

## Installation 

There are two ways of installing laygo2 on your environment; 1) cloning from github and 2) running pip.

### Installing laygo2 from github

Users can download and install the laygo2 package by cloning its github 
repository by typing the following command:

    >>>> git clone https://github.com/niftylab/laygo2.git

It is highly recommended that the following command is used periodically to maintain the code to the latest version.

    >>>> git pull origin master

After that, update the PHTHONPATH environment variable to point out the laygo2 package path for importing.

    # (csh/tcsh example) add the following command to your .cshrc
    setenv PYTHONPATH ${PYTHONPATH}:[LAYGO2_INSTALLATION_DIR]/laygo2

### From Pypi 

Instead of cloning github repository, laygo2 package can be installed by running the following pip command:

    >>>> pip install laygo2

## Technology setup for laygo2

To be added

## Simple gate generation

Running the following command will generate a NAND gate layout.

    >>>> run ./laygo2/examples/nand_generate.py
    
The resulting layout of the NAND gate is shown in the figure below:

![laygo2 nand gate](../assets/img/user_guide_nandgate.png "laygo2 NAND gate layout")
