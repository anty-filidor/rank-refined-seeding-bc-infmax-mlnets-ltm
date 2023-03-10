import pandas as pd
import network_diffusion as nd
import networkx as nx


def _network_from_pandas(path):
    df = pd.read_csv(path, names=["node_1", "node_2", "layer"])
    net_dict = {l_name: nx.Graph() for l_name in [*df["layer"].unique()]}
    for _, row in df.iterrows():
        net_dict[row["layer"]].add_edge(row["node_1"], row["node_2"])
    return nd.MultilayerNetwork.from_nx_layers(
        layer_names=[*net_dict.keys()], network_list=[*net_dict.values()]
    )


def get_aucs_network():
    return nd.MultilayerNetwork.from_mpx(file_path="data/aucs.mpx")


def get_ckm_physicians_network():
    return _network_from_pandas(
        "data/networks/CKM-Physicians-Innovation_4NoNature.edges"
    )


def get_eu_transportation_network():
    return _network_from_pandas(
        "data/networks/EUAirTransportation_multiplex_4NoNature.edges"
    )


def get_lazega_network():
    return _network_from_pandas(
        "data/networks/Lazega-Law-Firm_4NoNatureNoLoops.edges"
    )


def get_er2_network():
    return nd.MultilayerNetwork.from_mpx(file_path="data/networks/er_2.mpx")


def get_er3_network():
    return nd.MultilayerNetwork.from_mpx(file_path="data/networks/er_3.mpx")


def get_er5_network():
    return nd.MultilayerNetwork.from_mpx(file_path="data/networks/er_5.mpx")


def get_sf2_network():
    return nd.MultilayerNetwork.from_mpx(file_path="data/networks/sf_2.mpx")


def get_sf3_network():
    return nd.MultilayerNetwork.from_mpx(file_path="data/networks/sf_3.mpx")


def get_sf5_network():
    return nd.MultilayerNetwork.from_mpx(file_path="data/networks/sf_5.mpx")


def load_network(net_name: str) -> nd.MultilayerNetwork:
    if net_name == "aucs":
        return get_aucs_network()
    elif net_name == "ckm_physicians":
        return get_ckm_physicians_network()
    elif net_name == "eu_transportation":
        return get_eu_transportation_network()
    elif net_name == "lazega":
        return get_lazega_network()
    elif net_name == "er2":
        return get_er2_network()
    elif net_name == "er3":
        return get_er3_network()
    elif net_name == "er5":
        return get_er5_network()
    elif net_name == "sf2":
        return get_sf2_network()
    elif net_name == "sf3":
        return get_sf3_network()
    elif net_name == "sf5":
        return get_sf5_network()
    raise AttributeError(f"Unknown network: {net_name}")
