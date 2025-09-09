# Rank-Refining Seed Selection Methods for Budget Constrained Influence Maximisation in Multilayer Networks under Linear Threshold Model

A repository with the datasets, experimental environment, post-processing pipeline,
and detailed results for the above project.

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
├── experiments.ipynb         -> doodles
├── postprocessing.ipynb      -> script to genreate `all_results.csv`
├── all_result_analysis.ipynb -> script to analyse results form base networks
├── top_result_analysis.ipynb -> script to analyse results from large networks
└── efficiency_maps.ipynb     -> script to obtain efficiency maps
```

## Executing experiments

To run experiments with execute: `run_experiments.py` and provide proper CLI
arguments, i.e. a path to configuration file and runner type. See examples in
`_experiments_/examples` for inspirations. 

## Generating data visualisations

Please run `postprocessing.ipynb` to convert raw results. Then, depending to the
stage of the experiments, execute `all_result_analysis.ipynb` or `top_result_analysis.ipynb`
to obtain analysis from performed experiments.

## Acknowledgment

This work was supported by the National Science Centre, Poland [grant no. 2022/45/B/ST6/04145] (www.multispread.pwr.edu.pl); the Polish Ministry of Science and Higher Education programme “International Projects Co-Funded”; and the EU under the Horizon Europe [grant no. 101086321]. Views and opinions expressed are those of the authors and do not necessarily reflect those of the funding agencies.
