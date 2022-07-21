#!/usr/bin/env python3.7

# Copyright 2021, Gurobi Optimization, LLC

# This example formulates and solves the following simple MIP model
# using the matrix API:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time




N = 20

ROWlhs    = Variable( torch.zeros(N,N**2)  )
ROWrhs    = Variable(  ( torch.ones(N) )    )
COLlhs    = Variable( torch.zeros(N,N**2)  )
COLrhs    = Variable(  ( torch.ones(N) )    )
# All values are positive
POSlhs    = Variable(    -torch.eye(N**2,N**2)        )
POSrhs    = Variable(    -torch.zeros(N**2)        )
LEQ1lhs    = Variable(    torch.eye(N**2,N**2)        )
LEQ1rhs    = Variable(    torch.ones(N**2)        )



# Row sum constraints
for row in range(N):
    ROWlhs[row,row*N:(row+1)*N] = 1.0

# Column sum constraints
for col in range(N):
    COLlhs[col,col:-1:N] = 1.0
# fix the stupid issue of bottom left not filling
COLlhs[-1,-1] = 1.0


DSMl = torch.cat( (ROWlhs,COLlhs),0  )
DSMr = torch.cat( (ROWrhs,COLrhs),0  )
#Q =  eps*Variable(torch.eye(self.N**2))
BDlhs =  torch.cat( (POSlhs,LEQ1lhs),0  )
BDrhs =  torch.cat( (POSrhs,LEQ1rhs),0  )

DSMl = DSMl#.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq, self.N**2 )
DSMr = DSMr#.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq )



group_ids =  torch.Tensor([[0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.]])#,
                           #[0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
                           #[0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0.]])

position_bias_vector = 1. / torch.arange(1.,100.)
exposure = position_bias_vector[:N].float()

nBatch = group_ids.shape[0]

G = BDlhs.repeat(nBatch,1,1)
h = BDrhs.repeat(nBatch,1)

if( group_ids!=None ):
    print("Fairness indicated")
    #if x.shape[0] != group_ids.shape[0]:
    #    print("Error: Input scores and group ID's not not have the same batch size")
    #    input()

    # The fairness constraint should be:
    # f^T P v = 0
    # useful form here is
    # (v f^T) P*  = 0
    # where P* is P flattened (row-major)
    f = group_ids/group_ids.sum(1).reshape(-1,1) - (1 - group_ids)/(1 - group_ids).sum(1).reshape(-1,1)
    v = exposure.repeat(f.shape[0],1) # repeat to match dimensions of f (batch dim)

    # Set up v and f for outer product
    v_unsq = v.unsqueeze(1).permute(0,2,1)
    f_unsq = f.unsqueeze(1)

    # Outer product v f^T
    #   unroll to match P*
    #   unsqueeze to make each a 1-row matrix
    vXf = torch.bmm(v_unsq,f_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1) # this is still a batch
    fair_b = torch.zeros(nBatch,1)

    # JK Do we need to consider the computation graph wrt the group identity vectors?

    # Here we exploit x!=x for x==nan
    #vXf = torch.where(vXf==vXf, vXf, vXf.new_zeros(vXf.shape))

    A = torch.cat( (DSMl.repeat(nBatch,1,1),vXf),1 )
    #torch.cat((I.repeat(3,1,1),X.unsqueeze(1)),1)   # X is 2D, cat each row of X to a copy of I
                                                     # need this in case vXf is incorporated into ineq matrix
    b = torch.cat( (DSMr.repeat(nBatch,1),fair_b),1 )
else:
    print("No fairness indicated")
    A = DSMl.repeat(nBatch,1,1)
    b = DSMr.repeat(nBatch,1)


A = A[0]
b = b[0]
G = G[0]
h = h[0]
"""
print("A.size() = ")
print( A.size() )
print("b.size() = ")
print( b.size() )
print("G.size() = ")
print( G.size() )
print("h.size() = ")
print( h.size() )

print("A.to_sparse() = ")
print( A.to_sparse() )
print("G.to_sparse() = ")
print( G.to_sparse() )

print("A.to_sparse().indices() = ")
print( A.to_sparse().indices() )
print("A.to_sparse().values() = ")
print( A.to_sparse().values() )
print("G.to_sparse() = ")
print( G.to_sparse() )
"""
A_rows = np.array( A.to_sparse().indices()[0] )
A_cols = np.array( A.to_sparse().indices()[1] )
A_vals = np.array( A.to_sparse().values() )
A_rhs = np.array( b )

G_rows = np.array( G.to_sparse().indices()[0] )
G_cols = np.array( G.to_sparse().indices()[1] )
G_vals = np.array( G.to_sparse().values() )
G_rhs = np.array( h )



#quit()







#def convert_to_sp(tens):
#    for
#    return val, row, col

time1 = time.time()
for i in range(500):
    try:

        # Create a new model
        #with gp.Env(empty=True) as env, gp.Model("matrix1") as m:
        #    env.setParam('OutputFlag', 0)
        #    env.start()

        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)
        #env.setParam("Method",1)
        env.start()
        m = gp.Model("LP1", env=env)

        # Create variables
        #x = m.addMVar(shape=3, vtype=GRB.BINARY, name="x")
        x = m.addMVar(shape=400, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

        # Set objective
        #obj = np.array([1.0, 1.0, 2.0])
        coeffs = np.random.rand(400)

        #objExp = gp.QuadExpr()
        #for i in range(len(coeffs)):
        #    objExp.add( x[i]*x[i] + x[i], 3.0 )


        #obj = gp.QuadExpr(x[1])
        #m.setObjective(coeffs @ x  +  x @ x)


        #m.setObjective(coeffs @ x, GRB.MAXIMIZE)
        #m.setObjective(x[0]*x[0], GRB.MAXIMIZE)



        A = sp.csr_matrix((A_vals, (A_rows, A_cols)), shape=(A_rows.max()+1, A_cols.max()+1))
        rhs = A_rhs
        m.addConstr(A @ x == rhs, name="A")


        #G = sp.csr_matrix((G_vals, (G_rows, G_cols)), shape=(G_rows.max(), G_cols.max()))
        #rhs = G_rhs
        #m.addConstr(G @ x <= rhs, name="G")

        # Optimize model
        #for i in range(500):
        coeffs = np.random.rand(400)
        s_time1 = time.time()
        m.setObjective(coeffs @ x - x @ x, GRB.MAXIMIZE)
        m.optimize()
        s_time2 = time.time()
        print('hot start solve time: {}'.format(s_time2-s_time1))
        print("solved {}".format(i))



        #print(x.X)
        #print('Obj: %g' % m.objVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')


time2 = time.time()
print("total time:")
print(time2-time1)
