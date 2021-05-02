import numpy as np
import random
import math
import matplotlib.pyplot as plt
import torch
from pyperlin import FractalPerlin2D
from matplotlib.colors import LinearSegmentedColormap

width = 1024
height = 1024

colour_list = ["midnightblue",
               "royalblue",
               "lightseagreen",
               "gold",
               "darkgoldenrod",
               "forestgreen",
               "forestgreen",
               "darkgreen",
               "saddlebrown",
               "dimgrey",
               "darkgrey",
               "snow"]

nodes = [0.0, 0.12, 0.2, 0.22, 0.24, 0.3, 0.35, 0.65, 0.68, 0.7, 0.90, 1.0]

colour_map = LinearSegmentedColormap.from_list("Terrain cmap", list(zip(nodes, colour_list)))


def generate_noise(w, h, n):
    shape = (n, h, w)
    resolutions = [(2 ** i, 2 ** i) for i in range(1, 7)]  # lacunarity
    factors = [0.7 ** i for i in range(6)]  # persistence
    g_cuda = torch.Generator(device='cuda')
    noise_layer = FractalPerlin2D(shape, resolutions, factors, generator=g_cuda)().cpu().numpy()[0]
    return noise_layer


def generate_poseidon_layer(w, h):
    p_layer = np.ones((h, w), dtype=float)
    conversion_point = random.randint(math.floor(0.7 * w), math.ceil(0.8 * w))
    x = int(0.2 * w)
    shade_fraction = 1 / x

    for row in p_layer:
        conversion_point += random.randint(-2, 1)
        shade_level = 1
        for i in range(w):
            if i < conversion_point:
                row[i] = 1
            elif i > conversion_point + x:
                row[i] = 0
            else:
                shade_level -= shade_fraction
                row[i] = shade_level

    np.save("p_layer.npy", p_layer)
    plt.imsave("p_map.png", p_layer, cmap='Blues')


def apply_poseidon_layer(noise_layer):
    poseidon_layer = np.load("p_layer.npy")
    output_layer = np.multiply(noise_layer, poseidon_layer)

    return output_layer


def complete_map_gen(width, height, colour_map):
    generate_poseidon_layer(width, height)
    noise_layer = generate_noise(width, height, 8)
    normal_layer = (noise_layer / 2) + 0.5
    final_layer = normal_layer ** 0.5
    # final_layer = apply_poseidon_layer(normal_layer)
    np.save(f"layers.npy", final_layer)
    plt.imsave(f"mapped.png", final_layer, cmap=colour_map)


def read_cmap_file(filename):
    opened_cmap = plt.imread(filename)
    print(opened_cmap)

# complete_map_gen(width, height, colour_map)
read_cmap_file("cmap.png")