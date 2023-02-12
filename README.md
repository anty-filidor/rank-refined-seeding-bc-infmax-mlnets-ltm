# Linear Threshold Mmodel seeding for Multi Layer Networks

Repository for experiments on seed selection methods for multilayer linear 
threshold model.

Authors: Michał Czuba, Piotr Bródka  
Affiliation: Wrocław University of Science and Technology, Poland

## Data

Used networks come from repository with experiments on ICM model: 
https://github.com/pbrodka/SQ4MLN.

## Configuration of the runtime

`git submodule update --init --recursive`

`conda create --name ltm-seeding-mln python=3.10`
`conda activate ltm-seeding-mln`
`pip install -r submodules/network-diffusion/requirements/production.txt`
`pip install ipykernel`
`python -m ipykernel install --user --name=ltm-seeding-mln`

### Unix

`ln -s network_diffusion submodules/network-diffusion/network_diffusion`

### Windows

`mklink /J .\network_diffusion .\submodules\network-diffusion\network_diffusion`


## Executing experiments

To run experiments with "classic" seeding methods: `python runner.py` and update
`SEED_SELECTOR` variable (in the beginning of the script) with seed selection
class that is supported (full list is in `utils.py`).  

Experiments on greedy algorithms are defined in `greedy_runner.py` due to its
"hacky" nature.

These two scripts store results in `./experiments` directory. In order to
concatenate them use `postprocessing.ipynb` notebook that will produce an 
aggregated file: `./experiments/all_results.csv`

If reproducible run is needed please uncomment `set_seed` method invocation in
scripts.

## Generating data visualisations

Please run `result_analysis.ipynb` to obtain analysis of the obtained results.
