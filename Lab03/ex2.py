from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
# Defining the model structure. We can define the network by just passing a list of edges.
model = DiscreteBayesianNetwork([('A', 'B')])

# Defining individual CPDs.
cpd_d = TabularCPD(variable='A', variable_card=2, values=[[0.17], [0.83]]) # A = 0 add red ball, A = 1 add different colored ball
cpd_b = TabularCPD(variable='B', variable_card=2,
                   values=[[0.4, 0.3],
                           [0.6, 0.7]],
                   evidence=['A'],
                   evidence_card=[2])

# Associating the CPDs with the network
model.add_cpds(cpd_d, cpd_b)
# Verifying the model
assert model.check_model()

# Performing exact inference using Variable Elimination
infer = VariableElimination(model)
result = infer.query(variables=['B']).values[0]

print(result)

"Valoarea data de reteaua bayesiana este exact egala cu valoarea teoretica, spre deosebire de rezultatul dat de simularea de la ex1 lab2, care avea o oarecare eroare"

