import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time
from ortools.linear_solver import pywraplp
from gurobi_rank import * #ort_solve, grb_solve

N = 20
group_ids =  torch.Tensor([[0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.]])
coeffs = np.random.rand(400)

### checked, these give the same result
P_grbX = grb_solve(N, coeffs, group_ids)
P_ortX = ort_solve(N, coeffs, group_ids)

gids = group_ids


env,m,x  = grb_setup(N, gids)
solver_grb   = grb_policyLP(env,m,x)
sol_grb  = solver_grb.solve(coeffs)



s,x  = ort_setup(N, gids)
solver_ort   = ort_policyLP(s,x)
sol_ort  = solver_ort.solve(coeffs)


P_grb = sol_grb
P_ort = sol_ort


print("P_grb.sum() = ")
print( P_grb.sum()    )

print("P_ort.sum() = ")
print( P_ort.sum()    )

print("P_grbX.sum() = ")
print( P_grbX.sum()    )

print("P_ortX.sum() = ")
print( P_ortX.sum()    )

this = torch.stack( [P_grb,P_ort] ).T
print(this)

print("diff = ")
print( P_grb - P_ort  )

print("diff.max() = ")
print( (P_grb - P_ort).max()  )

print("diff.min() = ")
print( (P_grb - P_ort).min()  )




print("diff = ")
print( P_grb - P_grbX  )

print("diff.max() = ")
print( (P_grb - P_grbX).max()  )

print("diff.min() = ")
print( (P_grb - P_grbX).min()  )
