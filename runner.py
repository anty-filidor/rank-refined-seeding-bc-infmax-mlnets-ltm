import itertools

import network_diffusion as nd
import numpy as np
import pandas as pd

from loader import *
from tqdm import tqdm
from utils import *


# set_seed(43)  # in order to make reproducible results uncomment it!

SEED_SELECTOR = nd.seeding.RandomSeedSelector()
PROTOCOLS = ("OR", "AND")
SEEDING_BUDGETS = [(100 - i, i) for i in np.logspace(0, 2, num=15).round(2)]
MI_VALUES = np.linspace(0.1, 1, num=10)
NETWORKS = {
    "aucs": get_aucs_network(),
    "ckm_physicians": get_ckm_physicians_network(),
    "eu_transportation": get_eu_transportation_network(),
    "lazega": get_lazega_network(),
    "er2": get_er2_network(),
    "er3": get_er3_network(),
    "er5": get_er5_network(),
    "sf2": get_sf2_network(),
    "sf3": get_sf3_network(),
    "sf5": get_sf5_network(),
}

MAX_EPOCHS_NUM = 1000
PATIENCE = 1
REPEATS_OF_EACH_CASE = determine_repetitions_for_selector(SEED_SELECTOR)
FULL_LOGS_FREQ = 20
OUT_DIR = prepare_out_path_for_selector(SEED_SELECTOR)

global_stats_handler = pd.DataFrame(data={})
experiments = itertools.product(PROTOCOLS, SEEDING_BUDGETS, MI_VALUES, NETWORKS)
p_bar = tqdm(list(experiments), desc="main loop", leave=False, colour="green")

print(f"Experiments started at {get_current_time()}")

for idx, investigated_case in enumerate(p_bar):

    # obtain parameters of the propagation scenario
    protocol = investigated_case[0]
    seeding_budget = investigated_case[1]
    mi_value = investigated_case[2]
    network = NETWORKS[investigated_case[3]]

    # initialise model
    mltm = nd.models.MLTModel(
        protocol=protocol,
        seed_selector=SEED_SELECTOR,
        seeding_budget = seeding_budget,
        mi_value=mi_value,
    )

    # perform experiment on given model and network "REPEATS" times
    for repetition in range(1, 1 + REPEATS_OF_EACH_CASE):

        # update progress_bar
        case_name = (
            f"proto_{protocol}--a_seeds_{seeding_budget[1]}"
            f"--mi_{round(mi_value, 3)}--net_{investigated_case[3]}"
            f"--run_{repetition}_{REPEATS_OF_EACH_CASE}"
        )
        p_bar.set_description_str(str(case_name))

        try:
            # run experiment on a deep copy of the network!
            experiment = nd.MultiSpreading(model=mltm, network=network.copy())
            logs = experiment.perform_propagation(
                n_epochs=MAX_EPOCHS_NUM, patience=PATIENCE
            )

            # obtain global data and if case is even local one as well
            diffusion_len, activ_actors = extract_basic_stats(
                logal_stats=logs._local_stats, patience=PATIENCE
            )
            activ_actors_prct = activ_actors / network.get_actors_num() * 100
            if idx % FULL_LOGS_FREQ == 0:
                case_dir = OUT_DIR.joinpath(f"{idx}-{case_name}")
                case_dir.mkdir(exist_ok=True)
                logs.report(path=str(case_dir))

        except:
            # print corrupted case 
            diffusion_len, activ_actors_prct = None, None
            print(f"Ooops something went wrong for case: {case_name}")

        # update global logs
        case = {
            "network": investigated_case[3],
            "protocol": protocol,
            "seeding_budget": seeding_budget[1],
            "mi_value": mi_value,
            "diffusion_len": diffusion_len,
            "activated_prct_actors": activ_actors_prct,
            "repetition_run": repetition,
        }
        global_stats_handler = pd.concat(
            [global_stats_handler, pd.DataFrame.from_records([case])],
            ignore_index=True,
            axis=0,
        )


global_stats_handler.to_csv(OUT_DIR.joinpath("results.csv"))

print(f"Experiments finished at {get_current_time()}")
