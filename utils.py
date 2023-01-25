import datetime
import os
import random
import sys

from pathlib import Path

import numpy as np
import network_diffusion as nd


def set_seed(seed):
    """Fix seeds for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_basic_stats(detailed_logs, patience):
    """Get length of diffusion, real number of seeds and final coverage."""
    length_of_diffusion = 0
    active_nodes_list = []
    seed_nodes_list = []

    for epoch_num, epoch_changes in detailed_logs.items():
        epoch_num = int(epoch_num)

        if len(epoch_changes) > 0:
            # +1 for seeding epoch
            # -patience for epochs when no propagation was observed
            length_of_diffusion = epoch_num + 1 - patience
    
        acviated_nodes = []
        for node in epoch_changes:
            if node["new_state"] == "1":
                acviated_nodes.append(node['node_name'])
            else:  # sanity check to detect leaks when nodes gets deactivated
                if epoch_num > 0:
                    raise AttributeError("Results contradict themselves!")
        
        active_nodes_list.extend(acviated_nodes)
        if epoch_num == 0:
            seed_nodes_list.extend(acviated_nodes)

    active_actors_num = len(set(active_nodes_list))
    seed_actors_num = len(set(seed_nodes_list))

    return length_of_diffusion, active_actors_num, seed_actors_num


def compute_gain(seeds_prct, coverage_prct):
    max_available_gain = 100 - seeds_prct
    obtained_gain = coverage_prct - seeds_prct
    return 100 * obtained_gain / max_available_gain


def block_prints():
    sys.stdout = open(os.devnull, 'w')


def enable_prints():
    sys.stdout = sys.__stdout__


def get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S")


def prepare_out_path_for_selector(selector):
    if isinstance(selector, nd.seeding.DegreeCentralitySelector):
        out_path = Path("./experiments/degree_centrality")
    elif isinstance(selector, nd.seeding.KShellSeedSelector):
        out_path = Path("./experiments/k_sheel")
    elif isinstance(selector, nd.seeding.KShellMLNSeedSelector):
        out_path = Path("./experiments/k_sheel_mln")
    elif isinstance(selector, nd.seeding.NeighbourhoodSizeSelector):
        out_path = Path("./experiments/neighbourhood_size")
    elif isinstance(selector, nd.seeding.PageRankSeedSelector):
        out_path = Path("./experiments/page_rank")
    elif isinstance(selector, nd.seeding.RandomSeedSelector):
        out_path = Path("./experiments/random")
    elif isinstance(selector, nd.seeding.VoteRankSeedSelector):
        out_path = Path("./experiments/vote_rank")
    elif isinstance(selector, nd.seeding.VoteRankMLNSeedSelector):
        out_path = Path("./experiments/vote_rank_mln")
    else:
        raise ValueError(f"{selector} is not a valid seed selector!")
    out_path.mkdir(exist_ok=True, parents=True)
    return out_path


def determine_repetitions_for_selector(selector):
    if isinstance(selector, nd.seeding.RandomSeedSelector):
        repeats = 20
    elif isinstance(selector, nd.seeding.base_selector.BaseSeedSelector):
        repeats = 1
    else:
        raise ValueError(f"{selector} is not a valid seed selector!")
    return repeats
