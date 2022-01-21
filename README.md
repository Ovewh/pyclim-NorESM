# pyclim-NorESM

Python scripts used for analysis and plotting output from the Norwegian Earth System Model (NorESM)

## Structure

- **pyclim_noresm**: folder which contains functions for reading and analyzing NorESM output
- **examples**: folder with example scripts to show how the function scripts in **pyclim-noresm** can be used

## What can you find
 
- Reading NorESM data. Choose between noresm raw data or cmorized files
 
- Functions used for calculating time, regional and global averages, extract sub-regions, regridding etc.

- Functions and scripts for calculating northwards atmospheric energy transport

- For the regridding package xesmf to work properly, you need to make a new conda environment before installing the packages:

### Installation

1. Clone the git repository and cd into the pyclim-NorESM directory:
```
git clone https://github.com/adagj/pyclim-NorESM.git && cd pyclim-NorESM

```

2. Install the conda enviroment
```
conda env create -f=environment.yaml
```
3. Install pyclim_noresm using pip (use -e for development):

```
pip install -e . 
```

