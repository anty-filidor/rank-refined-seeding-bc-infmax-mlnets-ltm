from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from network_diffusion.models.ICM_model2 import ICModel as IC2
from network_diffusion.models.ICM_model import ICModel
from network_diffusion.mln.mlnetwork import MultilayerNetwork as mln
from network_diffusion.models.mlt_model import MLTModel
from network_diffusion.seeding.neighbourhoodsize_selector import NeighbourhoodSizeSelector
from network_diffusion.multi_spreading import  MultiSpreading
from network_diffusion.seeding.kshell_selector import  KShellSeedSelector, KShellMLNSeedSelector
from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork as MLNetwork
from network_diffusion.models.base_model import BaseModel
from network_diffusion.models.base_model import NetworkUpdateBuffer as NUBuff
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE, NumericType
tup = (50,50)

#model1 = []
#site = MLNetwork()
ml = MLNetwork.from_mpx("D:/Zajęcia/Studia magisterskie/Do pracy magisterskiej/python/Independent-Cascade-Model/data/networks/er_2.mpx")
seed = NeighbourhoodSizeSelector()
model = MLTModel(tup, seed, "OR", 0.5)

icm = ICModel(tup,seed,"OR",0.15)
icm2 = IC2(tup,seed,"AND",0.15)
icm2.set_initial_states(ml)
prop = MultiSpreading(icm,ml)
print(prop.perform_propagation(2))
#print(len(icm2.symulacja(ml))) # symulacja zwraca losowe wyniki czyli pr
#print(model.__str__())

print(len(icm2.network_evaluation_step(ml)))