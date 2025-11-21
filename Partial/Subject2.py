import networkx as nx
import numpy as np
from hmmlearn import hmm

states = ["W", "R", "S"]
n_states = len(states)

observations = ["L", "M", "H"]
n_observations = len(observations)

start_probability = np.array([0.4, 0.3, 0.3])

transition_probability = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.7, 0.1],
    [0.3, 0.2, 0.5],
])

emission_probability = np.array([
    [0.1, 0.7, 0.2],
    [0.05, 0.25, 0.7],
    [0.8, 0.15, 0.05],
])

model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

#b
observations_sequence = np.array([1, 2, 0]).reshape(-1, 1)
prob = model.score(observations_sequence)
print(np.exp(prob))
#c
log_probability, hidden_states = model.decode(observations_sequence,
                                              lengths = len(observations_sequence),
                                              algorithm ='viterbi')
print("Most likely hidden states:", hidden_states)
print("probability: ", np.exp(log_probability))