# Linear Threshold Mmodel seeding for Multi Layer Networks

Repository for experiments on seed selection methods for multilayer linear 
threshold model.

Authors: Michał Czuba, Piotr Bródka  
Affiliation: Wrocław University of Science and Technology, Poland

## Data

Used networks come from repository with experiments on ICM model: 
https://github.com/pbrodka/SQ4MLN.

## Configuration of the runtime

`git submodule update --init`

`conda create --name ltm-seeding-mln python=3.10`
`conda activate ltm-seeding-mln`
`pip install -r submodules/network-diffusion/requirements/production.txt`
`pip install ipykernel seaborn`
`python -m ipykernel install --user --name=ltm-seeding-mln`

### Unix

`ln -s network_diffusion submodules/network-diffusion/network_diffusion`

### Windows

`mklink /J .\network_diffusion .\submodules\network-diffusion\network_diffusion`


## Executing experiments

To run experiments with execute: `run_experiments.py` and provide proper CLI
arguments, i.e. a path to configuration file and runner type. See examples in
'data/example_configs' for inspirations. 

## Generating data visualisations

Please run `result_analysis.ipynb` to obtain analysis of the obtained results.
