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


def extract_basic_stats(logal_stats, patience):
    """Get last epoch when diffusion took place and final coverage."""

    length_of_diffusion = 0
    activated_nodes_list = []

    for epoch_num, epoch_changes in logal_stats.items():
        if len(epoch_changes) > 0:
            # +1 for seeding epoch
            # -patience for epochs when no propagation was observed
            length_of_diffusion = int(epoch_num) + 1 - patience
        for node in epoch_changes:
            if node["new_state"] == "1":
                activated_nodes_list.append(node['node_name'])
            else:  # sanity check to detect leaks when nodes gets deactivated
                if int(epoch_num) > 0:
                    raise AttributeError("Results contradict themselves!")

    activated_actors = len(set(activated_nodes_list))

    return length_of_diffusion, activated_actors

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
        out_path = Path("./experiments/k_sheel_mcz")
    elif isinstance(selector, nd.seeding.NeighbourhoodSizeSelector):
        out_path = Path("./experiments/neighbourhood_size")
    elif isinstance(selector, nd.seeding.PageRankSeedSelector):
        out_path = Path("./experiments/page_rank")
    elif isinstance(selector, nd.seeding.RandomSeedSelector):
        out_path = Path("./experiments/random")
    elif isinstance(selector, nd.seeding.VoteRankSeedSelector):
        out_path = Path("./experiments/vote_rank")
    else:
        raise ValueError(f"{selector} is not a valid seed selector!")
    out_path.mkdir(exist_ok=True, parents=True)
    return out_path
