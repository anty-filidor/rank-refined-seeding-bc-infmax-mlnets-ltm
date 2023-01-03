import pandas as pd
import network_diffusion as nd
import networkx as nx

def _network_from_pandas(path):
    df = pd.read_csv(path, names=["node_1", "node_2", "layer"])
    net_dict = {l_name: nx.Graph() for l_name in [*df["layer"].unique()]}
    for _, row in df.iterrows():
        net_dict[row["layer"]].add_edge(row["node_1"], row["node_2"])
    return nd.MultilayerNetwork.load_layers_nx(
        layer_names=[*net_dict.keys()], network_list=[*net_dict.values()]
    )

def get_aucs_network():
    return nd.MultilayerNetwork.load_mpx(file_path="data/aucs.mpx")

def get_ckm_physicians_network():
    return _network_from_pandas(
        "data/CKM-Physicians-Innovation_4NoNature.edges"
    )

def get_eu_transportation_network():
    return _network_from_pandas(
        "data/EUAirTransportation_multiplex_4NoNature.edges"
    )

def get_lazega_network():
    return _network_from_pandas(
        "data/Lazega-Law-Firm_4NoNatureNoLoops.edges"
    )

def get_er2_network():
    return nd.MultilayerNetwork.load_mpx(file_path="data/er_2.mpx")

def get_er3_network():
    return nd.MultilayerNetwork.load_mpx(file_path="data/er_3.mpx")

def get_er5_network():
    return nd.MultilayerNetwork.load_mpx(file_path="data/er_5.mpx")

def get_sf2_network():
    return nd.MultilayerNetwork.load_mpx(file_path="data/sf_2.mpx")

def get_sf3_network():
    return nd.MultilayerNetwork.load_mpx(file_path="data/sf_3.mpx")

def get_sf5_network():
    return nd.MultilayerNetwork.load_mpx(file_path="data/sf_5.mpx")
