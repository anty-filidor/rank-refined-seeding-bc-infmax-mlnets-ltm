import datetime
import os
import random
import sys

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
            elif node["new_state"] == "-1":
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


def get_seed_selector(selector_name):
    if selector_name == "degree_centrality":
        return nd.seeding.DegreeCentralitySelector
    elif selector_name == "k_shell":
        return nd.seeding.KShellSeedSelector
    elif selector_name == "k_shell_mln":
        return nd.seeding.KShellMLNSeedSelector
    elif selector_name == "neighbourhood_size":
        return nd.seeding.NeighbourhoodSizeSelector
    elif selector_name == "page_rank":
        return nd.seeding.PageRankSeedSelector
    elif selector_name == "page_rank_mln":
        return nd.seeding.PageRankMLNSeedSelector
    elif selector_name == "random":
        return nd.seeding.RandomSeedSelector
    elif selector_name == "vote_rank":
        return nd.seeding.VoteRankSeedSelector
    elif selector_name == "vote_rank_mln":
        return nd.seeding.VoteRankMLNSeedSelector
    elif selector_name == "closeness":
        return nd.seeding.ClosenessSelector
    elif selector_name == "betweennes":
        return nd.seeding.BetweennessSelector
    elif selector_name == "katz":
        return nd.seeding.KatzSelector
    elif selector_name == "cbim":
        return nd.seeding.CBIMselector

    raise AttributeError(f"{selector_name} is not a valid seed selector name!")
