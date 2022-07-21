
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
from birkhoff import birkhoff_von_neumann_decomposition


# Create a new model
m = gp.Model("qp")



# Create variables
N = 6
dist = {}
for i in range(N):
    for j in range(N):
        dist[(i,j)] = random.random()

ukeys = {}
for i in range(N):
        ukeys[i] = random.random()


x = m.addVars(dist.keys(), lb=0,  obj=dist,  name='x')

u = m.addVars(ukeys.keys(), lb=0,  name='u')

for i in range(N)[1:]:
    for j in range(N)[1:]:
        if i != j:
            m.addConstr( u[i]-u[j] + N*x[i,j] <= N-1)


for k in range(N):
    m.addConstr(x.sum('*',k) == 1)
    m.addConstr(x.sum(k,'*') == 1)
    #m.addConstr(x[k,0]+x[k,2]+x[k,3] <= 1, "c9")
    #m.addConstr(x[0,k]+x[1,k]+x[2,k]+x[3,k] <= 1, "c10")

for i in range(N):
    for j in range(N)[i:]:
        m.addConstr(x[i,j]+x[j,i] <= 0.5 , "c10") # <=1 doesn't prevent self-edges

#for i in range(N):
#    for j in range(N)[i:]:
#        m.addConstr(x[i,j] == x[j,i]) # <=1 doesn't prevent self-edges


for i in range(N):
    m.addConstr(x[i,i] == 0)


m.optimize()

X = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        X[i,j] = x[i,j].x

print("X = ")
print( X )
print("X*X*X= ")
print( np.matmul(np.matmul(X,X), X) )

decomp = birkhoff_von_neumann_decomposition( np.matmul(np.matmul(X,X), X) )
convex_coeffs, permutations = zip(*decomp)
permutations = np.array(permutations)

print("permutations = ")
print( permutations )

#print('Obj: %g' % obj.getValue())
