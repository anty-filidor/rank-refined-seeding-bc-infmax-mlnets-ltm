import network_diffusion as nd
from pathlib import Path
from loader import *

net = get_aucs_network()

mltm = nd.models.MLTModel(
    protocol="OR",
    seed_selector=nd.seeding.KShellSeedSelector(),
    seeding_budget = (70, 30),
    mi_value=0.4,
)

out_dir = "./experiment_report"
Path(out_dir).mkdir(exist_ok=True)

experiment = nd.MultiSpreading(model=mltm, network=net)

logger = experiment.perform_propagation(n_epochs=100, stop_on_hold=True)
logger.plot(True, out_dir)
logger.report(path=out_dir)

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

# ilu aktorów było aktywnych kiedy proces się nie rozprzestrzeniał i kiedy skońćzył się rozprzestrzeniać
