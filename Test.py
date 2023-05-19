import random
import pandas as pd

from misc.utils import set_seed
from network_diffusion import MultilayerNetwork
from network_diffusion.models import MICModel
from network_diffusion.multi_spreading import MultiSpreading
from network_diffusion.seeding import RandomSeedSelector

rand_probability = random.random()
rand_seed_num = int(random.random() * 100)
rand_protocol = "OR" if random.random() > 0.5 else "AND"
print(
    f"Evaluating params: \
    probability: {round(rand_probability, 3)}, \
    seeds: {rand_seed_num}, \
    protocol: {rand_protocol}"
)

# initialise a container for logs
repetitive_logs = []

# perform 20 times experiments with the same parameters and collect bulk logs
for i in range(20):
    set_seed(0)
    network = MultilayerNetwork.from_mpx("./data/networks/er_3.mpx")
    model = MICModel(
        seeding_budget=(100-rand_seed_num, rand_seed_num, 0),
        seed_selector=RandomSeedSelector(),
        protocol=rand_protocol,
        probability=rand_probability,
    )
    experiment = MultiSpreading(model, network.copy())
    logs = experiment.perform_propagation(n_epochs=5)
    repetitive_logs.append(logs._global_stats_converted.copy())


# compare all collected logs with the first one in order to determine wheter 
# they are identical
for idx, log in enumerate(repetitive_logs[1:]):
    for l_name, l_reference_log in repetitive_logs[0].items():
        try:
            pd.testing.assert_frame_equal(log[l_name], l_reference_log)
        except AssertionError as e:
            print(f"Runs 0 and {idx} in layer {l_name} are not the same!")
            print(log[l_name])
            print(l_reference_log)
            print(e)
            print("\n\n\n")

print("Finished checking if results are repetitive!")
