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
pip install -e submodules/network-diffusion
pip install ipykernel seaborn pandas-profiling
python -m ipykernel install --user --name=ltm-seeding-mln
```

This repo works with git LFS, so please install it in order to pull large files!

## Structure of the repository
```
.
├── README.md
├── _data_set                 -> networks used in experiments
├── _experiments
│   ├── examples              -> example configs to run experiments
│   ├── all_methods           -> raw results of evaluation of all methods on base nets
│   │   ├── all_results.csv   -> csv with aggregated and processed results
│   │   ...
│   │   └── <method name>     -> detailed logs for <method name>
│   └── top_methods           -> raw results of evaluation of top methods on large nets
├── _results                  -> generated results that were used in the paper
├── misc                      -> miscellaneous scripts to trigger experiments
├── runners                   -> scripts to execute experiments according to configs
├── run_experiments.py        -> main entrypoint ti trigger simulations
├── submodules                -> backbone library for simulations as a submodule
├── postprocessing.ipynb      -> script to genreate `all_results.csv`
├── result_analysis.ipynb     -> script to analyse results
└──  robustness_maps.ipynb     -> script to obtain robustness maps
```

## Executing experiments

To run experiments with execute: `run_experiments.py` and provide proper CLI
arguments, i.e. a path to configuration file and runner type. See examples in
'data/example_configs' for inspirations. 

## Generating data visualisations

Please run `result_analysis.ipynb` to obtain analysis from performed 
experiments.
