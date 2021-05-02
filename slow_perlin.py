import numpy as np
import random
import math
import matplotlib.pyplot as plt
from perlin import Perlin
from matplotlib.colors import LinearSegmentedColormap

width = 1920
height = 1080

colour_list = ["midnightblue",
               "royalblue",
               "lightseagreen",
               "gold",
               "darkgoldenrod",
               "forestgreen",
               "forestgreen",
               "darkgreen",
               "forestgreen",
               "darkgoldenrod",
               "dimgrey",
               "darkgrey",
               "snow"]

nodes = [0.0, 0.12, 0.2, 0.22, 0.24, 0.3, 0.35, 0.55, 0.7, 0.74, 0.76, 0.90, 1.0]

colour_map = LinearSegmentedColormap.from_list("Basic cmap", list(zip(nodes, colour_list)))

# seed_list = [1234, 2345, 3456, 5678, 6789, 9876, 8765, 7654, 6543, 5432, 4321]
seed_list = [123]


def open_layer(n):
    f = f"{seed}layer{n}.npy"
    return np.load(f)


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


def generate_layer(w, h, f, seed):
    new_layer = np.empty((h, w), dtype=float)
    noise = Perlin(seed)

    for i, row in enumerate(new_layer):
        for j in range(w):
            row[j] = (noise.two(f * j, f * i) / f)

    return new_layer


def combine_layers(layer_count, w, h):
    layer_list = []
    for i in range(0, layer_count):
        current_layer = open_layer(i)
        layer_list.append(current_layer)

    output_layer = np.zeros((h, w), dtype=float)
    for i, layer in enumerate(layer_list):
        output_layer += layer * (1/(2**i))

    output_layer = output_layer / 2

    return output_layer


def apply_poseidon_layer(normal_layer):
    poseidon_layer = np.load("p_layer.npy")
    output_layer = np.multiply(normal_layer, poseidon_layer)

    return output_layer


def generate_layer_set(w, h, n, seed):
    layer_list = []
    for i in range(0, n):
        new_layer = generate_layer(w, h, 2**i, seed)
        layer_list.append(new_layer)
        np.save(f"{seed}layer{i}.npy", new_layer)
        plt.imsave(f"{seed}\map{i}.png", new_layer, cmap='Greys')

    return layer_list


def complete_map_gen(width, height, seed, colour_map):
    # generate_poseidon_layer(width, height)
    generate_layer_set(width, height, 1, seed)
    raw_layer = combine_layers(1, width, height)
    layer_range = math.ceil(np.amax(raw_layer) - np.amin(raw_layer)) + 2
    normal_layer = (raw_layer/layer_range) + 0.5
    final_layer = apply_poseidon_layer(normal_layer)
    np.save(f"{seed}layers.npy", final_layer)
    plt.imsave(f"{seed}mapped.png", final_layer, cmap=colour_map)


for seed in seed_list:
    complete_map_gen(width, height, seed, colour_map)
