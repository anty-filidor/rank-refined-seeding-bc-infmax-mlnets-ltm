from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.seeding.kshell_selector import  KShellSeedSelector, KShellMLNSeedSelector
from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork as MLNetwork
from network_diffusion.models.base_model import BaseModel
from network_diffusion.models.base_model import NetworkUpdateBuffer as NUBuff
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE, NumericType
tup = (70,30)

model1 = []
#site = MLNetwork()
ml = MLNetwork.from_mpx("D:/Zajęcia/Studia magisterskie/Do pracy magisterskiej/python/ltm-seeding-mln/data/networks/er_2.mpx")
#print(MLNetwork._get_description_str(ml))
G = nx.Graph()
#nx.draw(G)
#seed1 = BaseSeedSelector()
seed = KShellMLNSeedSelector()
model = MLTModel(tup, seed, "OR", 0.5)
m1=model.set_initial_states(ml)

model1 = model.network_evaluation_step(ml)
m2 = model.get_allowed_states(ml)
#print(list(model1))
print(m2)
