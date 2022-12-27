import network_diffusion as nd

net = nd.MultilayerNetwork.load_mlx(file_path="data/aucs.mpx")

mltm = nd.models.MLTModel(
    layers=list(net.layers.keys()),
    protocol="AND",
    seed_selector=nd.seeding.KShellSeedSelector(),
    seeding_budget = (70, 30),
    mi_value=0.4,
)

experiment = nd.MultiSpreading(model=mltm, network=net)
logger = experiment.perform_propagation(n_epochs=10)
logger.plot(True, ".")
logger.report(path="./experiment_data")
