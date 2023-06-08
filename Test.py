import random
import pandas as pd

from misc.utils import set_seed
from network_diffusion import MultilayerNetwork
from network_diffusion.models import MICModel
from network_diffusion.multi_spreading import MultiSpreading
from network_diffusion.seeding import RandomSeedSelector
from network_diffusion.seeding.cbim import CBIMselector
network = MultilayerNetwork.from_mpx("./data/networks/er_3.mpx")
model = MICModel(
    seeding_budget=(100 - 30, 30, 0),
    seed_selector=CBIMselector(0.3),
    protocol="AND",
    probability=0.3,
)
seed = CBIMselector(0.3)
seed.actorwise(network)

experiment = MultiSpreading(model, network.copy())
logs = experiment.perform_propagation(n_epochs=5)
logs.report(True,"./vis")