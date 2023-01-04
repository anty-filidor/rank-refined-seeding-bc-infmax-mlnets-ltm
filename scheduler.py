import network_diffusion as nd
from pathlib import Path
from loader import *


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


# experiment

net = get_aucs_network()

mltm = nd.models.MLTModel(
    protocol="AND",
    seed_selector=nd.seeding.KShellSeedSelector(),
    seeding_budget = (70, 30),
    mi_value=0.9,
)

out_dir = "./experiment_report"
Path(out_dir).mkdir(exist_ok=True)

experiment = nd.MultiSpreading(model=mltm, network=net)

logger = experiment.perform_propagation(n_epochs=100, stop_on_hold=True)

# obtaining results

logger.plot(True, out_dir)
logger.report(path=out_dir)
diffusion_len, activ_actors = extract_basic_stats(logger._local_stats)
print(f"Length of diffusion {diffusion_len}, activated {activ_actors} actors.")

# protocols = ("OR", "AND")
# seeding_methods = (
#     nd.seeding.DegreeCentralitySelector,
#     nd.seeding.KShellSeedSelector,
#     nd.seeding.NeighbourhoodSizeSelector,
#     nd.seeding.PageRankSeedSelector,
#     nd.seeding.RandomSeedSelector,
#     nd.seeding.VoteRankSeedSelector,
# )
# seeding_budgets = [(i, 100 - i) for i in np.arange(0, 101, 1)]
# mi_values = np.logspace(-2, 0, num=100)
