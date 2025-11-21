from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation

import matplotlib.pyplot as plt
import networkx as nx
model = DiscreteBayesianNetwork([('O', 'H'), ('O', 'W'), ('H', 'R'), ('W', 'R'), ('H', 'E'), ('R', 'C')])

cpd_o = TabularCPD(variable='O', variable_card=2, values=[[0.3], [0.7]]) # S=0 cold, S=1 mild
#Let Yes = 1, No = 0
cpd_h = TabularCPD(variable='H', variable_card=2,
                   values=[[0.1, 0.8],
                           [0.9, 0.2]],
                  evidence=['O'],
                  evidence_card=[2])

cpd_w = TabularCPD(variable='W', variable_card=2,
                   values=[[0.9, 0.4],
                           [0.1, 0.6]],
                   evidence=['O'],
                   evidence_card=[2])
#Let warm = 1, cool = 0
cpd_r = TabularCPD(variable='R', variable_card=2,
                   values=[[0.5, 0.2, 0.5, 0.7],
                           [0.5, 0.8, 0.5, 0.3]],
                   evidence=['H', 'W'],
                   evidence_card=[2, 2])

#Let high = 1, low = 0
cpd_e = TabularCPD(variable='E', variable_card=2,
                   values=[[0.8, 0.2],
                           [0.2, 0.8]],
                   evidence=['H'],
                   evidence_card=[2])
#let comfortable = 1, uncomfortable = 0
cpd_c = TabularCPD(variable='C', variable_card=2,
                   values=[[0.6, 0.15],
                           [0.4, 0.85]],
                   evidence=['R'],
                   evidence_card=[2])

# Associating the CPDs with the network
model.add_cpds(cpd_o, cpd_h, cpd_w, cpd_r, cpd_e, cpd_c)
# Verifying the model
assert model.check_model()

print(model.local_independencies(['O', 'H', 'W', 'R', 'E', 'C']))
# Performing exact inference using Variable Elimination
infer = VariableElimination(model)
result = infer.query(variables=['H'], evidence={'C': 1})
result1 = infer.query(variables=['E'], evidence={'C' : 1})

bp_infer = BeliefPropagation(model)
bp = bp_infer.map_query(variables=['H', 'W'], evidence={'C' : 1})

#a)
print(result)
print(result1)

#b)
print(bp)

#c)
'''
Da E  este independent de W dat H, deoarece observarea lui H  blocheaza drumul de la W la E
'''
