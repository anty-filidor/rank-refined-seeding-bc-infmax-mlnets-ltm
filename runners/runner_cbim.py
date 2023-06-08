import itertools
import yaml

import network_diffusion as nd
import pandas as pd
from pathlib import Path
from misc.net_loader import load_network
from misc.utils import *
from tqdm import tqdm


def parameter_space(protocols, seed_selector, seed_budgets, probability, networks):
    seed_budgets_full = [(100 - i, i, 0) for i in seed_budgets]
    ss = (get_seed_selector(seed_selector["name"])(**seed_selector["parameters"]))
    net_ranks= [
        (n, _ := load_network(n), ss(_, actorwise=True)) for n in networks
    ]  # network name, network, ranking

    return itertools.product(protocols, seed_budgets_full, probability, net_ranks)


def run_experiments(config):
    max_epochs_num = config["run"]["max_epochs_num"]
    patience = config["run"]["patience"]
    repeats_of_each_case = config["run"]["repetitions"]
    logging_freq = config["logging"]["full_output_frequency"]
    out_dir = Path(config["logging"]["out_dir"]) / config["logging"]["name"]
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Experiments started at {get_current_time()}")

    global_stats_handler = pd.DataFrame(data={})

    for method in config["model"]["parameters"]["ss_method"]:
        p_space = parameter_space(
            protocols=config["model"]["parameters"]["protocols"],
            seed_budgets=config["model"]["parameters"]["seed_budgets"],
            probability=config["model"]["parameters"]["probability"],
            seed_selector=method,
            networks=config["networks"],
        )


        p_bar = tqdm(list(p_space), desc="main loop", leave=False, colour="green")

        for idx, investigated_case in enumerate(p_bar):

            # obtain parameters of the propagation scenario
            protocol = investigated_case[0]
            seeding_budget = investigated_case[1]
            probability = investigated_case[2]
            print(len(investigated_case[3]))
            net_name, net, ranking = investigated_case[3]
            # initialise model - in order to speed up computations we use mocky
            # selector and feed it with apriori computed ranking
            micm = nd.models.MICModel(
                protocol=protocol,
                seed_selector=nd.seeding.MockyActorSelector(ranking),
                seeding_budget=seeding_budget,
                probability=probability,
            )

            for repetition in range(1, 1 + repeats_of_each_case):
                # update progress_bar
                case_name = (
                    f"proto_{protocol}--a_seeds_{seeding_budget[1]}"
                    f"--prob_{round(probability, 3)}--net_{net_name}"
                    f"--run_{repetition}_{repeats_of_each_case}"
                )
                p_bar.set_description_str(str(case_name))

                try:
                    # run experiment on a deep copy of the network!
                    experiment = nd.MultiSpreading(model=micm, network=net.copy())
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
                    "probability": probability,
                    "repetition_run": repetition,
                    "diffusion_len": diffusion_len,
                    "active_actors_prct": active_actors_prct,
                    "seed_actors_prct": seed_actors_prct,
                    "gain": gain,
                    "s_method": f"{method['name']}:{method['parameters']['threshold']}"
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
