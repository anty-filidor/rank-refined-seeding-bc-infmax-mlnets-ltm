# Linear Threshold Model - seed selection methods - Multi Layer Networks

Repository for experiments on seed selection methods for Multilayer Linear 
Threshold Model.

Authors: Michał Czuba, Piotr Bródka  
Affiliation: Wrocław University of Science and Technology, Poland

## Data

Networks used in experiments come from Piotr Bródka's repository with experiments
on ICM model: https://github.com/pbrodka/SQ4MLN and from a website of
Manilo De Domenico: https://manliodedomenico.com/data.php.

## Configuration of the runtime

```
git submodule update --init
conda create --name ltm-seeding-mln python=3.10
conda activate ltm-seeding-mln
pip install -r submodules/network-diffusion/requirements/production.txt
pip install ipykernel seaborn pandas-profiling
python -m ipykernel install --user --name=ltm-seeding-mln
```

This repo works with git LFS, so please install it in order to pull large files!

### Unix

`ln -s submodules/network-diffusion/network_diffusion network_diffusion

### Windows

`mklink /J .\network_diffusion .\submodules\network-diffusion\network_diffusion`


## Structure of the repository
```
.
├── README.md
├── data
│   ├── example_configs   -> example configs to run experiments
│   ├── findings          -> generated results that were used in the paper
│   └── networks          -> networks used in experiments
├── experiments
│   ├── all_results.csv   -> csv with aggregated and processed results
│   ├── degree_centrality -> detailed logs from experiments with deg-c
│   .
│   .
│   .
│   └── vote_rank_mln     -> detailed logs from experiments with v-rnk-m
├── misc                  -> miscellaneous scripts to trigger experiments
├── network_diffusion     -> a backbone library for simulations
├── postprocessing.ipynb  -> a script to genreate `all_results.csv`
├── result_analysis.ipynb -> a script to analyse results
├── robustness_maps.ipynb -> a script to obtain robustness maps
├── run_experiments.py    -> main entrypoint ti trigger simulations
├── runners               -> scripts to execute experiments according to configs
└── submodules            -> git modules
```

## Executing experiments

To run experiments with execute: `run_experiments.py` and provide proper CLI
arguments, i.e. a path to configuration file and runner type. See examples in
'data/example_configs' for inspirations. 

## Generating data visualisations

Please run `result_analysis.ipynb` to obtain analysis from performed 
experiments.
