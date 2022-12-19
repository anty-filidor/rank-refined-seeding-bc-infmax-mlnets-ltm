import network_diffusion as nd


net = nd.MultilayerNetwork.load_mlx(file_path="data/aucs.mpx")

print(net)
