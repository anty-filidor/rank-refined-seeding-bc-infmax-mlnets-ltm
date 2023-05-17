from typing import Dict, List, Tuple
import os
import sys
import networkx as nx
import random
import numpy as np
from network_diffusion.models.mic_model import MICModel
from network_diffusion.experiment_logger import ExperimentLogger
from network_diffusion.seeding.degreecentrality_selector import DegreeCentralitySelector
from network_diffusion.models.mic_model import MICModel
from network_diffusion.seeding.random_selector import RandomSeedSelector
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
tup = (10, 90)

#seed = RandomSeedSelector()

ml = MLNetwork.from_mpx(
    "D:/Zajęcia/Studia magisterskie/Do pracy magisterskiej/python/Independent-Cascade-Model/data/networks/er_3.mpx")

# random seed

random.seed('a28')
seed = RandomSeedSelector()


print(random.random())
model = MICModel(tup,seed,'OR', 0.1)

spred = MultiSpreading(model,ml)
print(spred.perform_propagation(4))


#graph: nx.Graph() = ml.layers['l1']
#print(list(graph.neighbors('a630')))






