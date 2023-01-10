import datetime
import os
import random
import sys

import numpy as np


def set_seed(seed):
    """Fix seeds for reproducable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_basic_stats(logal_stats):
    """Get last epoch when diffusion took place and final coverage."""

    length_of_diffusion = 0
    activated_nodes_list = []

    for epoch_num, epoch_changes in logal_stats.items():
        if len(epoch_changes) > 0:
            length_of_diffusion = int(epoch_num) + 1 # because of seeding epoch
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
