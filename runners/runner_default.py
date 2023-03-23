import itertools
import yaml

import network_diffusion as nd
import pandas as pd
from misc.net_loader import load_network
from misc.utils import *
from pathlib import Path
from tqdm import tqdm


def parameter_space(protocols, seed_budgets, mi_values, networks):
    seed_budgets_full = [(100 - i, i) for i in seed_budgets]
    nets = [(n, load_network(n)) for n in networks]  # network name, network
    return itertools.product(protocols, seed_budgets_full, mi_values, nets)


def run_experiments(config):

    p_space = parameter_space(
        protocols=config["model"]["parameters"]["protocols"],
        seed_budgets=config["model"]["parameters"]["seed_budgets"],
        mi_values=config["model"]["parameters"]["mi_values"],
        networks=config["networks"],
    )
    seed_selector = get_seed_selector(
        config["model"]["parameters"]["ss_method"]["name"]
    )(**config["model"]["parameters"]["ss_method"]["parameters"])

    max_epochs_num = config["run"]["max_epochs_num"]
    patience = config["run"]["patience"]
    repeats_of_each_case = config["run"]["repetitions"]
    logging_freq = config["logging"]["full_output_frequency"] * repeats_of_each_case
    out_dir = Path(config["logging"]["out_dir"]) / config["logging"]["name"]
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Experiments started at {get_current_time()}")

    global_stats_handler = pd.DataFrame(data={})
    p_bar = tqdm(list(p_space), desc="main loop", leave=False, colour="green")

    for idx, investigated_case in enumerate(p_bar):

        # obtain parameters of the propagation scenario
        protocol = investigated_case[0]
        seeding_budget = investigated_case[1]
        mi_value = investigated_case[2]
        net_name, net = investigated_case[3]

        # initialise model
        mltm = nd.models.MLTModel(
            protocol=protocol,
            seed_selector=seed_selector,
            seeding_budget = seeding_budget,
            mi_value=mi_value,
        )

        # perform experiment on given model and network "REPEATS" times
        for repetition in range(1, 1 + repeats_of_each_case):

            # update progress_bar
            case_name = (
                f"proto_{protocol}--a_seeds_{seeding_budget[1]}"
                f"--mi_{round(mi_value, 3)}--net_{net_name}"
                f"--run_{repetition}_{repeats_of_each_case}"
            )
            p_bar.set_description_str(str(case_name))

            try:
                # run experiment on a deep copy of the network!
                experiment = nd.MultiSpreading(model=mltm, network=net.copy())
                logs = experiment.perform_propagation(
                    n_epochs=max_epochs_num, patience=patience
                )

                # obtain global data and if case is even local one as well
                diffusion_len, active_actors, seed_actors = extract_basic_stats(
                    detailed_logs=logs._local_stats, patience=patience
                )
                active_actors_prct = active_actors / net.get_actors_num() * 100
                seed_actors_prct = seed_actors / net.get_actors_num() * 100
                gain = compute_gain(seed_actors_prct, active_actors_prct)
                if idx % logging_freq == 0:
                    case_dir = out_dir.joinpath(f"{idx}-{case_name}")
                    case_dir.mkdir(exist_ok=True)
                    logs.report(path=str(case_dir))

            except KeyboardInterrupt as e:
                raise e

            except BaseException as e:
                diffusion_len = None
                active_actors_prct = None
                seed_actors_prct = None
                gain = None
                print(f"Ooops something went wrong for case: {case_name}: {e}")

            # update global logs
            case = {
                "network": net_name,
                "protocol": protocol,
                "seeding_budget": seeding_budget[1],
                "mi_value": mi_value,
                "repetition_run": repetition,
                "diffusion_len": diffusion_len,
                "active_actors_prct": active_actors_prct,
                "seed_actors_prct": seed_actors_prct,
                "gain": gain,
            }
            global_stats_handler = pd.concat(
                [global_stats_handler, pd.DataFrame.from_records([case])],
                ignore_index=True,
                axis=0,
            )

    # save global logs and config
    global_stats_handler.to_csv(out_dir.joinpath("results.csv"))
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"Experiments finished at {get_current_time()}")
