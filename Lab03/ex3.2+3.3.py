from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
# Defining the model structure. We can define the network by just passing a list of edges.
model = DiscreteBayesianNetwork([('C', 'P')])

# Defining individual CPDs.
cpd_d = TabularCPD(variable='C', variable_card=2, values=[[0.5], [0.5]]) # C = 0 player 0 chosen, A = 1 player 1 chosen

cpd_b = TabularCPD(variable='P', variable_card=2,
                   values=[[0.5, 0.43],
                           [0.5, 0.57]],
                   evidence=['C'],
                   evidence_card=[2])

# Associating the CPDs with the network
model.add_cpds(cpd_d, cpd_b)
# Verifying the model
assert model.check_model()

# Performing exact inference using Variable Elimination
infer = VariableElimination(model)
result = infer.query(variables=['P'])
print(result)

if result.values[1] > result.values[0]:
    print("P1 is more likely to have started the game!")
else:
    print("P0 is more likely to have started the game!")

