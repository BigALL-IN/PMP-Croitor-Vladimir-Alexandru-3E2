from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx
# Defining the model structure. We can define the network by just passing a list of edges.
model = DiscreteBayesianNetwork([('S', 'O'), ('S', 'L'), ('S', 'M'), ('L', 'M')])

# Defining individual CPDs.
cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]]) # S=0 not spam, S=1 spam

cpd_o = TabularCPD(variable='O', variable_card=2,
                   values=[[0.9, 0.3],
                           [0.1, 0.7]],
                  evidence=['S'],
                  evidence_card=[2])

cpd_l = TabularCPD(variable='L', variable_card=2,
                   values=[[0.7, 0.2],
                           [0.3, 0.8]],
                   evidence=['S'],
                   evidence_card=[2])

cpd_m = TabularCPD(variable='M', variable_card=2,
                   values=[[0.8, 0.4, 0.5, 0.1],
                           [0.2, 0.6, 0.5, 0.9]],
                   evidence=['S', 'L'],
                   evidence_card=[2, 2])

# Associating the CPDs with the network
model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)
# Verifying the model
assert model.check_model()

print(model.local_independencies(['S', 'O', 'L', 'M']))
# Performing exact inference using Variable Elimination
infer = VariableElimination(model)
result = infer.query(variables=['S'], evidence={'O': 0, 'L' : 0, 'M' : 0})
result1 = infer.query(variables=['S'], evidence={'O': 0, 'L' : 0, 'M' : 1})
result2 = infer.query(variables=['S'], evidence={'O': 0, 'L' : 1, 'M' : 0})
result3 = infer.query(variables=['S'], evidence={'O': 0, 'L' : 1, 'M' : 1})
result4 = infer.query(variables=['S'], evidence={'O': 1, 'L' : 0, 'M' : 0})
result5 = infer.query(variables=['S'], evidence={'O': 1, 'L' : 0, 'M' : 1})
result6 = infer.query(variables=['S'], evidence={'O': 1, 'L' : 1, 'M' : 0})
result7 = infer.query(variables=['S'], evidence={'O': 1, 'L' : 1, 'M' : 1})

print(result)
print(result1)
print(result2)
print(result3)
print(result4)
print(result5)
print(result6)
print(result7)
