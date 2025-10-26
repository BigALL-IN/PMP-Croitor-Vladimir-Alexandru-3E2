from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

pixel_variables = []
for row in range(5):
    for col in range(5):
        pixel_variables.append(f"X{row}_{col}")

neighbor_edges = []
for row in range(5):
    for col in range(5):
        current_pixel = f"X{row}_{col}"

        if row + 1 < 5:
            neighbor_edges.append((current_pixel, f"X{row + 1}_{col}"))

        if col + 1 < 5:
            neighbor_edges.append((current_pixel, f"X{row}_{col + 1}"))

model = MarkovNetwork()
model.add_nodes_from(pixel_variables)
model.add_edges_from(neighbor_edges)


graph = nx.Graph()
graph.add_nodes_from(model.nodes())
graph.add_edges_from(model.edges())

grid_positions = {}
for row in range(5):
    for col in range(5):
        grid_positions[f"X{row}_{col}"] = (col, -row)

plt.figure(figsize=(6, 6))
nx.draw(graph, grid_positions, node_size=300, node_color='lightblue',
        alpha=0.8, with_labels=False, edge_color='gray')
plt.title("Markov Network Structure (5x5 Grid)")
plt.axis('equal')
plt.show()


rng = np.random.default_rng(42)

original_image = rng.choice([-1, 1], size=(5, 5))

noisy_image = original_image.copy()
num_noisy = rng.choice(5 * 5, size=2, replace=False)

for flat_index in num_noisy:
    row = flat_index // 5
    col = flat_index % 5
    noisy_image[row, col] *= -1


LAMBDA = 2.5

unary_factors = []
for row in range(5):
    for col in range(5):
        variable_name = f"X{row}_{col}"
        observed_value = int(noisy_image[row, col])

        values = [
            math.exp(-LAMBDA * ((-1) - observed_value) ** 2),
            math.exp(-LAMBDA * ((+1) - observed_value) ** 2)
        ]

        factor = DiscreteFactor(
            variables=[variable_name],
            cardinality=[2],
            values=values
        )
        unary_factors.append(factor)

pair_values = []
for xi in [-1, 1]:
    for xj in [-1, 1]:
        pair_values.append(math.exp(-((xi - xj) ** 2)))

pair_factors = []
for pixel1, pixel2 in neighbor_edges:
    factor = DiscreteFactor(
        variables=[pixel1, pixel2],
        cardinality=[2, 2],
        values=pair_values
    )
    pair_factors.append(factor)

all_factors = unary_factors + pair_factors
model.add_factors(*all_factors)
model.check_model()


bp_infer = BeliefPropagation(model)
bp = bp_infer.map_query(variables=pixel_variables)

denoised_image = np.zeros((5, 5), dtype=int)
for row in range(5):
    for col in range(5):
        var = f"X{row}_{col}"
        denoised_image[row, col] = -1 if bp[var] == 0 else 1


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(original_image, cmap="gray", vmin=-1, vmax=1)
axes[0].set_title("Original Image", fontsize=14)
axes[0].axis("off")

axes[1].imshow(noisy_image, cmap="gray", vmin=-1, vmax=1)
axes[1].set_title("Noisy Image", fontsize=14)
axes[1].axis("off")

axes[2].imshow(denoised_image, cmap="gray", vmin=-1, vmax=1)
axes[2].set_title("Denoised (MAP)", fontsize=14)
axes[2].axis("off")

plt.tight_layout()
plt.show()
