from typing import List

import itertools

from loader import *
from tqdm import tqdm
from utils import *

import network_diffusion as nd
import numpy as np
import pandas as pd


PROTOCOLS = ("OR", "AND")
MI_VALUES = np.linspace(0.1, 0.9, num=9)
SEEDING_MAX_BUDGET = 30
NETWORKS = {
    "aucs": get_aucs_network(),
    "ckm_physicians": get_ckm_physicians_network(),
    "lazega": get_lazega_network(),
}

MAX_EPOCHS_NUM = 1000
PATIENCE = 1
REPEATS_OF_EACH_CASE = 1
FULL_LOGS_FREQ = 1
OUT_DIR = Path("./experiments/greedy")

global_stats_handler = pd.DataFrame(data={})
experiments = itertools.product(PROTOCOLS, MI_VALUES, NETWORKS)
p_bar = tqdm(list(experiments), desc="main loop", leave=False, colour="green")

print(f"Experiments started at {get_current_time()}")

for idx, investigated_case in enumerate(p_bar):

    # obtain parameters of the propagation scenario
    protocol = investigated_case[0]
    mi_value = investigated_case[1]
    network = NETWORKS[investigated_case[2]]

    greedy_ranking: List[nd.mln.actor.MLNetworkActor] = []
    actors_num = network.get_actors_num()

    # repeat until spent seeding budget exceeds maximum value
    while (100 * len(greedy_ranking) / actors_num) <= SEEDING_MAX_BUDGET:

        # containers for the best actor in the run and its performance
        best_actor = None
        best_diffusion_len = MAX_EPOCHS_NUM
        best_coverage = 0
        best_logs = None

        # obtain pool of actors and limit of budget in the run
        eval_seed_budget = 100 * (len(greedy_ranking) + 1) / actors_num
        available_actors = set(network.get_actors()).difference(
            set(greedy_ranking)
        )

        # update progress_bar
        case_name = (
            f"proto_{protocol}--a_seeds_{round(eval_seed_budget, 2)}"
            f"--mi_{round(mi_value, 3)}--net_{investigated_case[2]}"
        )
        p_bar.set_description_str(str(case_name))

        # iterate greedly through all avail. actors to find the best combination
        for actor in available_actors:

            # initialise model with "ranking" that prioritises current actor
            apriori_ranking = [
                *greedy_ranking, actor, *available_actors.difference({actor})
            ]
            mltm = nd.models.MLTModel(
                protocol=protocol,
                seed_selector=nd.seeding.MockyActorSelector(apriori_ranking),
                seeding_budget=(100 - eval_seed_budget, eval_seed_budget),
                mi_value=mi_value,
            )

            # run experiment on a deep copy of the network!
            experiment = nd.MultiSpreading(model=mltm, network=network.copy())
            logs = experiment.perform_propagation(MAX_EPOCHS_NUM, PATIENCE)

            # compute boost that current actor provides
            diffusion_len, active_actors, seed_actors = extract_basic_stats(
                detailed_logs=logs._local_stats, patience=PATIENCE
            )
            coverage = active_actors / actors_num * 100

            # if gain is relevant update the best currently actor
            if (
                coverage > best_coverage or 
                (
                    coverage == best_coverage and
                    diffusion_len < best_diffusion_len
                )
            ):
                best_actor = actor
                best_diffusion_len = diffusion_len
                best_coverage = coverage
                best_logs = logs
                print(
                    f"\n\tcurrently best actor '{best_actor.actor_id}' for "
                    f"greedy list: {[i.actor_id for i in greedy_ranking]}, "
                    f"coverage: {round(best_coverage, 2)}"
                )

        # when the best combination is found update table with the best actors
        greedy_ranking.append(best_actor)

        # save logs for further analysis
        case_dir = OUT_DIR.joinpath(f"{idx}-{case_name}")
        case_dir.mkdir(exist_ok=True, parents=True)
        best_logs.report(path=str(case_dir))

        # update global logs
        case = {
            "network": investigated_case[2],
            "protocol": protocol,
            "seeding_budget": eval_seed_budget,
            "mi_value": mi_value,
            "repetition_run": 1,
            "diffusion_len": best_diffusion_len,
            "active_actors_prct": best_coverage,
            "seed_actors_prct": eval_seed_budget,
            "gain": compute_gain(eval_seed_budget, best_coverage),
        }
        global_stats_handler = pd.concat(
            [global_stats_handler, pd.DataFrame.from_records([case])],
            ignore_index=True,
            axis=0,
        )

global_stats_handler.to_csv(OUT_DIR.joinpath("results.csv"))

print(f"Experiments finished at {get_current_time()}")
