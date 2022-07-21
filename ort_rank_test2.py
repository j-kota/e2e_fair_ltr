# copied from ort_rank_test 05/12/22

#import gurobipy as gp
#from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time
from ortools.linear_solver import pywraplp
#from gurobi_rank import * #ort_solve, grb_solve
from ort_rank import *
from networksJK import PolicyLP, PolicyLP_Plus, PolicyBlackboxWrapper, create_torch_LP

import sys
sys.path.insert(0,'./NeurIPSIntopt/Interior/')
sys.path.insert(0,'../..')
from ip_model_whole import *



position_bias_vector = 1. / torch.arange(1.,
                                         100.) ** 1.0


N = 20
group_ids =  torch.Tensor([0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.])
coeffs = np.random.rand(400)

### checked, these give the same result
#P_grbX = grb_solve(N, coeffs, group_ids)
#P_ortX = ort_solve(N, coeffs, group_ids)

gids = group_ids


#env,m,x  = grb_setup(N, gids)
#solver_grb   = grb_policyLP(env,m,x)
#sol_grb  = solver_grb.solve(coeffs)

s,x  = ort_setup_Neq(N, gids, 'disp0', 1.0, 1.0, 1e-5)
solver_ort   = ort_policyLP(s,x)
sol_ort  = solver_ort.solve(coeffs)

qpth_LP = PolicyLP_Plus(N=20, eps = 1e-7, position_bias_vector = position_bias_vector[:N])
print(torch.Tensor(coeffs).shape)
print(gids.shape)
sol_qpth = qpth_LP(torch.Tensor(coeffs).unsqueeze(0), gids.unsqueeze(0))


print( sol_ort )
print( sol_ort.shape )

print( sol_qpth )
print( sol_qpth.shape )

wrapped_bb = PolicyBlackboxWrapper(10.0, N, gids.unsqueeze(0), 'disp0', 1.0, 1.0, 1e-5).apply
sol_bb = wrapped_bb(torch.Tensor(coeffs).unsqueeze(0))

print( sol_bb )
print( sol_bb.shape )

G,h,A,b = create_torch_LP(N=N, position_bias_vector = position_bias_vector[:N], group_ids = gids.unsqueeze(0))
ip_solver = IPOfunc( torch.Tensor(),torch.Tensor(),A,b,
                       bounds= [(0.0, 1.)]   )

print("About to call IP solver")
sol_ip = ip_solver(-torch.Tensor(coeffs).unsqueeze(0))

print(sol_ip)
print(sol_ip.max())



"""
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
"""
