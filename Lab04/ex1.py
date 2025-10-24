import numpy as np
from matplotlib import pyplot as plt
from pgmpy.inference import BeliefPropagation
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovNetwork
model = MarkovNetwork([('A1', 'A2'), ('A2', 'A5'), ('A2', 'A4'), ('A5', 'A4'), ('A1', 'A3'), ('A3', 'A4')])


import networkx as nx
pos = nx.circular_layout(model)
nx.draw(model, with_labels=True, pos=pos, alpha=0.5, node_size=2000)

plt.show()

cliques = list(nx.find_cliques(model))
print(cliques)

factors = []
i_values = {'A1' : 1, 'A2' : 2, 'A3' : 3, 'A4' : 4, 'A5' : 5}
for clique in cliques:
    clique_sorted = sorted(clique)
    cardinality = [2] * len(clique_sorted)

    n_atts = len(clique_sorted)
    n_configs = 2 ** n_atts
    probs = np.zeros(n_configs)

    for i in range(n_configs):
        config = []
        temp = i
        for j in range(n_atts):
            config.append(1 if temp % 2 == 1 else -1)
            temp //= 2

        exponent = sum(i_values[var] * val for var, val in zip(clique_sorted, config))
        probs[i] = np.exp(exponent)

    factor = DiscreteFactor(variables=clique_sorted,cardinality=cardinality,values=probs)
    factors.append(factor)

model.add_factors(*factors)

bp_infer = BeliefPropagation(model)
marginals = bp_infer.map_query(variables=['A1','A2','A3','A4','A5'])
print(marginals)
