import networkx as nx
import numpy as np
from hmmlearn import hmm

states = ["Difficult", "Medium", "Easy"]
n_states = len(states)

observations = ["FB", "B", "S", "NS"]
n_observations = len(observations)

start_probability = np.array([1/3, 1/3, 1/3])

transition_probability = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25],
])

emission_probability = np.array([
    [0.1, 0.2, 0.4, 0.3],
    [0.15, 0.25, 0.5, 0.1],
    [0.2, 0.3, 0.4, 0.1],
])

model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

observations_sequence = np.array([0, 0, 2, 1, 1, 2, 1, 1, 3, 1, 1]).reshape(-1, 1)
prob = model.score(observations_sequence)
print(np.exp(prob))
log_probability, hidden_states = model.decode(observations_sequence,
                                              lengths = len(observations_sequence),
                                              algorithm ='viterbi')
print("Most likely hidden states:", hidden_states)
print("probability: ", np.exp(log_probability))

G = nx.DiGraph()
for s in states:
    G.add_node(s)
for i in range(n_states):
    for j in range(n_states):
        if transition_probability[i, j] > 0:
            G.add_edge(states[i], states[j], weight=transition_probability[i, j])

nx.draw(G)
