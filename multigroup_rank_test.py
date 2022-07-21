import numpy as np
import math
import random
import copy
import torch
import torch.nn as nn

import pickle
import pandas as pd
from gurobi_rank import *

N = 20
disp_type = 'disp0'
group0_merit = 1.0
group1_merit = 1.0
delta = 0.00000001

#scores = torch.Tensor(  [ 8.7845e-02,  3.0729e-01,  1.3841e-01,  2.2779e-01, -2.3924e-01,
#                          2.3625e-01, -1.7794e-01, -3.8440e-02,  1.9790e-01,  4.4483e-02,
#                          6.6272e-01,  4.5028e-01,  3.1430e-01,  6.6546e-01,  1.1579e-01,
#                          4.9145e-01,  2.8865e-01,  7.7143e-01,  8.4712e-02,  1.9706e-01]   ).unsqueeze(0).unsqueeze(0)

#dscts = ( 1.0 / torch.log2(torch.arange(20).float() + 2) ).unsqueeze(0).unsqueeze(0)
#coeffs = torch.bmm( scores, dscts.permute(0,2,1) ).detach().double().numpy()

torch.manual_seed(0)
coeffs = torch.rand(400)

print("coeffs = ")
print( coeffs )



#group_identities = torch.Tensor( [1., 2., 1., 3., 1., 0., 1., 0., 1., 4., 1., 2., 1., 0., 1., 0., 1., 0., 1., 0.] )
group_identities = torch.Tensor( [1., 0., 1., 2., 1., 2., 4., 4., 1., 2., 5., 0., 1., 5., 1., 3., 1., 0., 1., 3.] )

s,x = ort_setup_multi_Neq(N, group_identities, disp_type, group0_merit, group1_merit, delta)
#s,x = ort_setup_Neq(N, group_identities, disp_type, group0_merit, group1_merit, delta)

solver = ort_policyLP(s,x)

pmat = solver.solve(  coeffs )


print("pmat = ")
print( pmat )
