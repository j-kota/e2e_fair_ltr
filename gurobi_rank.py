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
from parse_args import args


#args = N, group_ids, coeffs = np.random.rand(400),  group_ids (input as 1D, then unsqueeze to fit batch form for now)

def grb_solve(N, coeffs, group_ids):


    if( group_ids!=None ):
        group_ids = group_ids.unsqueeze(0).float()

    ROWlhs    = Variable( torch.zeros(N,N**2)  )
    ROWrhs    = Variable(  ( torch.ones(N) )  )
    COLlhs    = Variable( torch.zeros(N,N**2)  )
    COLrhs    = Variable(  ( torch.ones(N) )  )
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


    #group_ids =  torch.Tensor([[0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.]])

    position_bias_vector = 1. / torch.arange(1.,100.)
    exposure = position_bias_vector[:N].float()

    nBatch = 1 #group_ids.shape[0]

    G = BDlhs.repeat(nBatch,1,1)
    h = BDrhs.repeat(nBatch,1)

    if( group_ids!=None ):
        #print("Fairness indicated")
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
        #print("No fairness indicated")
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
    A_rhs  = np.array( b )

    G_rows = np.array( G.to_sparse().indices()[0] )
    G_cols = np.array( G.to_sparse().indices()[1] )
    G_vals = np.array( G.to_sparse().values() )
    G_rhs  = np.array( h )


    try:

        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)
        #env.setParam("Method",1)
        env.start()
        m = gp.Model("LP1", env=env)

        # Create variables
        x = m.addMVar(shape=400, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

        #coeffs = np.random.rand(400)


        A = sp.csr_matrix((A_vals, (A_rows, A_cols)), shape=(A_rows.max()+1, A_cols.max()+1))
        rhs = A_rhs
        m.addConstr(A @ x == rhs, name="A")

        #G = sp.csr_matrix((G_vals, (G_rows, G_cols)), shape=(G_rows.max(), G_cols.max()))
        #rhs = G_rhs
        #m.addConstr(G @ x <= rhs, name="G")

        # Optimize model

        coeffs = np.array(coeffs)
        m.setObjective(coeffs @ x , GRB.MAXIMIZE)
        m.optimize()
        #print("solved {}".format(i))



        #print(x.X)
        #print('Obj: %g' % m.objVal)

        return torch.Tensor(x.X)#.view(N,N)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')




# Initialized with environment, model and variables
# Needs objective coeffs to solve
class grb_policyLP():
    def __init__(self,env,model,x):
        self.env   = env
        self.model = model
        self.x = x

    def solve(self,coeffs):
        m = self.model
        x = self.x

        m.setObjective(coeffs @ x , GRB.MAXIMIZE)
        m.optimize()

        # delete
        """
        print("obj:")
        print( torch.dot(torch.Tensor(x.X),torch.Tensor(coeffs)) )
        input('k')
        """
        # delete


        return np.array(x.X)#, dtype=torch.float64)#.view(N,N)







# Initialized with environment, model and variables
# Needs objective coeffs to solve
class ort_policyLP():
    def __init__(self,solver,x):
        #self.env   = env
        self.solver = solver
        self.x = x

    def solve(self,coeffs):
        solver = self.solver
        x = self.x

        objective = solver.Objective()
        for i in range(len(coeffs)):
            objective.SetCoefficient(x[i], coeffs[i].item())

        objective.SetMaximization()
        solver.Solve()

        P = [ v.solution_value() for v in x]

        #print("obj = ")
        #print( np.dot( np.array(coeffs), np.array(P) )  )

        return np.array(P) #torch.Tensor(P)#, dtype=torch.float64)#.view(N,N)







# 0930

def grb_setup(N, group_ids):

    if( group_ids!=None ):
        group_ids = group_ids.unsqueeze(0).float()

    ROWlhs    = Variable( torch.zeros(N,N**2)  )
    ROWrhs    = Variable(  ( torch.ones(N) )  )
    COLlhs    = Variable( torch.zeros(N,N**2)  )
    COLrhs    = Variable(  ( torch.ones(N) )  )
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


    #group_ids =  torch.Tensor([[0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.]])

    position_bias_vector = 1. / torch.arange(1.,100.)
    exposure = position_bias_vector[:N].float()

    nBatch = 1 #group_ids.shape[0]

    G = BDlhs.repeat(nBatch,1,1)
    h = BDrhs.repeat(nBatch,1)

    if( group_ids!=None ):
        #print("Fairness indicated")
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
        #print("No fairness indicated")
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
    A_rhs  = np.array( b )

    G_rows = np.array( G.to_sparse().indices()[0] )
    G_cols = np.array( G.to_sparse().indices()[1] )
    G_vals = np.array( G.to_sparse().values() )
    G_rhs  = np.array( h )


    try:

        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)
        #env.setParam("Method",1)
        env.start()
        m = gp.Model("LP1", env=env)

        # Create variables
        x = m.addMVar(shape=400, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

        #coeffs = np.random.rand(400)


        A = sp.csr_matrix((A_vals, (A_rows, A_cols)), shape=(A_rows.max()+1, A_cols.max()+1))
        rhs = A_rhs
        m.addConstr(A @ x == rhs, name="A")

        #G = sp.csr_matrix((G_vals, (G_rows, G_cols)), shape=(G_rows.max(), G_cols.max()))
        #rhs = G_rhs
        #m.addConstr(G @ x <= rhs, name="G")

        # Optimize model

        #coeffs = np.array(coeffs)
        #m.setObjective(coeffs @ x - x @ x, GRB.MAXIMIZE)
        #m.optimize()
        #print("solved {}".format(i))

        #print(x.X)
        #print('Obj: %g' % m.objVal)

        return env,m,x      #torch.Tensor(x.X)#.view(N,N)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')









# Google OR solver section

def ort_solve(N, coeffs, group_ids):


    if( group_ids!=None ):
        group_ids = group_ids.unsqueeze(0).float()

    ROWlhs    = Variable( torch.zeros(N,N**2)  )
    ROWrhs    = Variable(  ( torch.ones(N) )  )
    COLlhs    = Variable( torch.zeros(N,N**2)  )
    COLrhs    = Variable(  ( torch.ones(N) )  )
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


    #group_ids =  torch.Tensor([[0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.]])

    position_bias_vector = 1. / torch.arange(1.,100.)
    exposure = position_bias_vector[:N].float()

    nBatch = 1 #group_ids.shape[0]

    G = BDlhs.repeat(nBatch,1,1)
    h = BDrhs.repeat(nBatch,1)

    if( group_ids!=None ):
        #print("Fairness indicated")
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
        v_unsq = v.unsqueeze(1)
        f_unsq = f.unsqueeze(1).permute(0,2,1)
        #v_unsq = v.unsqueeze(1).permute(0,2,1)
        #f_unsq = f.unsqueeze(1)

        # Outer product v f^T
        #   unroll to match P*
        #   unsqueeze to make each a 1-row matrix
        vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1) # this is still a batch
        #vXf = torch.bmm(v_unsq,f_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
        fair_b = torch.zeros(nBatch,1)

        # JK Do we need to consider the computation graph wrt the group identity vectors?

        # Here we exploit x!=x for x==nan
        #vXf = torch.where(vXf==vXf, vXf, vXf.new_zeros(vXf.shape))

        A = torch.cat( (DSMl.repeat(nBatch,1,1),vXf),1 )
        #torch.cat((I.repeat(3,1,1),X.unsqueeze(1)),1)   # X is 2D, cat each row of X to a copy of I
                                                         # need this in case vXf is incorporated into ineq matrix
        b = torch.cat( (DSMr.repeat(nBatch,1),fair_b),1 )
    else:
        #print("No fairness indicated")
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
    A_rhs  = np.array( b )

    G_rows = np.array( G.to_sparse().indices()[0] )
    G_cols = np.array( G.to_sparse().indices()[1] )
    G_vals = np.array( G.to_sparse().values() )
    G_rhs  = np.array( h )


    #try:

    solver = pywraplp.Solver('ScheduleFromRankings',
                     pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    x = [  solver.NumVar(0, 1,"x[{a}]".format(a=i)) for i in range(0,N**2)  ]

    #b = b.tolist()
    #A = A.tolist()

    # constraints of the matrix A
    A_constr = [  solver.Constraint( b[i].item(),b[i].item() ) for i in range(0,A.shape[0])  ]


    for i in range(A.shape[0]):     # each row of A is a constraint
        for j in range(A.shape[1]):
            if A[i][j].item() != 0.0:

                A_constr[i].SetCoefficient( x[j], A[i][j].item() )

    objective = solver.Objective()
    for i in range(len(coeffs)):
        objective.SetCoefficient(x[i], coeffs[i].item())

    objective.SetMaximization()
    solver.Solve()

    P = [ v.solution_value() for v in x]



    return torch.Tensor(P)#.view(N,N)



#s,x = ort_setup(args.list_len, gids, disp_type, group0_merit, group1_merit)

def ort_setup(N, group_ids, disp_type, group0_merit, group1_merit):

    if( group_ids!=None ):
        group_ids = group_ids.unsqueeze(0).float()

    ROWlhs    = Variable( torch.zeros(N,N**2)  )
    ROWrhs    = Variable(  ( torch.ones(N) )  )
    COLlhs    = Variable( torch.zeros(N,N**2)  )
    COLrhs    = Variable(  ( torch.ones(N) )  )
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


    #group_ids =  torch.Tensor([[0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.]])

    position_bias_vector = 1. / torch.arange(1.,100.)
    exposure = position_bias_vector[:N].float()

    nBatch = 1 #group_ids.shape[0]

    G = BDlhs.repeat(nBatch,1,1)
    h = BDrhs.repeat(nBatch,1)

    if( group_ids!=None ):
        #print("Fairness indicated")
        #if x.shape[0] != group_ids.shape[0]:
        #    print("Error: Input scores and group ID's not not have the same batch size")
        #    input()

        M0 = 1.0 if (disp_type == 'disp0') else group0_merit
        M1 = 1.0 if (disp_type == 'disp0') else group1_merit

        f =  M1*(1 - group_ids)/(1 - group_ids).sum(1).reshape(-1,1) -  M0*group_ids/group_ids.sum(1).reshape(-1,1)
        v = exposure.repeat(f.shape[0],1) # repeat to match dimensions of f (batch dim)

        # Set up v and f for outer product
        v_unsq = v.unsqueeze(1)
        f_unsq = f.unsqueeze(1).permute(0,2,1)
        #v_unsq = v.unsqueeze(1).permute(0,2,1)
        #f_unsq = f.unsqueeze(1)

        # Outer product v f^T
        #   unroll to match P*
        #   unsqueeze to make each a 1-row matrix
        vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1) # this is still a batch
        #vXf = torch.bmm(v_unsq,f_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
        fair_b = torch.zeros(nBatch,1)

        # JK Do we need to consider the computation graph wrt the group identity vectors?

        # Here we exploit x!=x for x==nan
        #vXf = torch.where(vXf==vXf, vXf, vXf.new_zeros(vXf.shape))

        A = torch.cat( (DSMl.repeat(nBatch,1,1),vXf),1 )
        #torch.cat((I.repeat(3,1,1),X.unsqueeze(1)),1)   # X is 2D, cat each row of X to a copy of I
                                                         # need this in case vXf is incorporated into ineq matrix
        b = torch.cat( (DSMr.repeat(nBatch,1),fair_b),1 )
    else:
        #print("No fairness indicated")
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
    A_rhs  = np.array( b )

    G_rows = np.array( G.to_sparse().indices()[0] )
    G_cols = np.array( G.to_sparse().indices()[1] )
    G_vals = np.array( G.to_sparse().values() )
    G_rhs  = np.array( h )


    #try:

    solver = pywraplp.Solver('ScheduleFromRankings',
                     pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    x = [  solver.NumVar(0, 1,"x[{a}]".format(a=i)) for i in range(0,N**2)  ]

    #b = b.tolist()
    #A = A.tolist()

    # constraints of the matrix A
    A_constr = [  solver.Constraint( b[i].item(),b[i].item() ) for i in range(0,A.shape[0])  ]


    for i in range(A.shape[0]):     # each row of A is a constraint
        for j in range(A.shape[1]):
            if A[i][j].item() != 0.0:

                A_constr[i].SetCoefficient( x[j], A[i][j].item() )

    """
    objective = solver.Objective()

    for i in range(len(coeffs)):
        objective.SetCoefficient(x[i], coeffs[i].item())

    objective.SetMaximization()
    solver.Solve()

    P = [ v.solution_value() for v in x]
    """

    return solver,x      #torch.Tensor(x.X)#.view(N,N)





def ort_setup_Neq(N, group_ids, disp_type, group0_merit, group1_merit, delta):

    if( group_ids!=None ):
        group_ids = group_ids.unsqueeze(0).float()

    ROWlhs    = Variable( torch.zeros(N,N**2)  )
    ROWrhs    = Variable(  ( torch.ones(N) )  )
    COLlhs    = Variable( torch.zeros(N,N**2)  )
    COLrhs    = Variable(  ( torch.ones(N) )  )
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


    #group_ids =  torch.Tensor([[0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.]])

    position_bias_vector = 1. / torch.arange(1.,100.)
    exposure = position_bias_vector[:N].float()

    nBatch = 1 #group_ids.shape[0]

    G = BDlhs.repeat(nBatch,1,1)
    h = BDrhs.repeat(nBatch,1)

    if( group_ids!=None ):
        #print("Fairness indicated")
        #if x.shape[0] != group_ids.shape[0]:
        #    print("Error: Input scores and group ID's not not have the same batch size")
        #    input()


        M0 = 1.0 if (disp_type == 'disp0') else group0_merit
        M1 = 1.0 if (disp_type == 'disp0') else group1_merit
        f =  M1*(1 - group_ids)/(1 - group_ids).sum(1).reshape(-1,1) -  M0*group_ids/group_ids.sum(1).reshape(-1,1)

        # The fairness constraint should be:
        # f^T P v = 0
        # useful form here is
        # (v f^T) P*  = 0
        # where P* is P flattened (row-major)
        #f = group_ids/group_ids.sum(1).reshape(-1,1) - (1 - group_ids)/(1 - group_ids).sum(1).reshape(-1,1)
        v = exposure.repeat(f.shape[0],1) # repeat to match dimensions of f (batch dim)

        # Set up v and f for outer product
        v_unsq = v.unsqueeze(1)
        f_unsq = f.unsqueeze(1).permute(0,2,1)
        #v_unsq = v.unsqueeze(1).permute(0,2,1)
        #f_unsq = f.unsqueeze(1)

        # Outer product v f^T
        #   unroll to match P*
        #   unsqueeze to make each a 1-row matrix
        vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1) # this is still a batch
        #vXf = torch.bmm(v_unsq,f_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
        fair_b = torch.zeros(nBatch,1)

        # JK Do we need to consider the computation graph wrt the group identity vectors?

        # Here we exploit x!=x for x==nan
        #vXf = torch.where(vXf==vXf, vXf, vXf.new_zeros(vXf.shape))

        #A = torch.cat( (DSMl.repeat(nBatch,1,1),vXf),1 )
        A = DSMl.repeat(nBatch,1,1)
        #torch.cat((I.repeat(3,1,1),X.unsqueeze(1)),1)   # X is 2D, cat each row of X to a copy of I
                                                         # need this in case vXf is incorporated into ineq matrix
        b = DSMr.repeat(nBatch,1)
    else:
        #print("No fairness indicated")
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
    A_rhs  = np.array( b )

    G_rows = np.array( G.to_sparse().indices()[0] )
    G_cols = np.array( G.to_sparse().indices()[1] )
    G_vals = np.array( G.to_sparse().values() )
    G_rhs  = np.array( h )


    NEQ      =  vXf.squeeze(0)#.unsqueeze(0)
    #NEQ     = torch.cat(  (vXf,-vXf), 1  )
    #NEQrhs  = torch.Tensor([delta,delta])


    solver = pywraplp.Solver('ScheduleFromRankings',
                     pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    x = [  solver.NumVar(0, 1,"x[{a}]".format(a=i)) for i in range(0,N**2)  ]

    #b = b.tolist()
    #A = A.tolist()

    # constraints of the matrix A
    A_constr = [  solver.Constraint( b[i].item(),b[i].item() ) for i in range(0,A.shape[0])  ]

    for i in range(A.shape[0]):     # each row of A is a constraint
        for j in range(A.shape[1]):
            if A[i][j].item() != 0.0:

                A_constr[i].SetCoefficient( x[j], A[i][j].item() )



    NEQ_constr = [  solver.Constraint( -delta,delta ) for i in range(0,NEQ.shape[0])  ]

    #print("NEQ = ")
    #print( NEQ )
    #print("NEQ.shape = ")
    #print( NEQ.shape )
    for i in range(NEQ.shape[0]):     # each row of A is a constraint
        for j in range(NEQ.shape[1]):
            if NEQ[i][j].item() != 0.0:

                NEQ_constr[i].SetCoefficient( x[j], NEQ[i][j].item() )



    """
    objective = solver.Objective()

    for i in range(len(coeffs)):
        objective.SetCoefficient(x[i], coeffs[i].item())

    objective.SetMaximization()
    solver.Solve()

    P = [ v.solution_value() for v in x]
    """

    return solver,x      #torch.Tensor(x.X)#.view(N,N)









# adapted from ort_setup_Neq
# for multi-group fairness
def ort_setup_multi_Neq(N, group_ids, disp_type, group0_merit, group1_merit, delta):

    if( group_ids!=None ):
        group_ids = group_ids.unsqueeze(0).float()

    ROWlhs    = Variable( torch.zeros(N,N**2)  )
    ROWrhs    = Variable(  ( torch.ones(N) )  )
    COLlhs    = Variable( torch.zeros(N,N**2)  )
    COLrhs    = Variable(  ( torch.ones(N) )  )
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


    #group_ids =  torch.Tensor([[0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.]])

    position_bias_vector = 1. / torch.arange(1.,100.)
    exposure = position_bias_vector[:N].float()

    nBatch = 1 #group_ids.shape[0]

    G = BDlhs.repeat(nBatch,1,1)
    h = BDrhs.repeat(nBatch,1)

    if( group_ids!=None ):
        #print("Fairness indicated")
        #if x.shape[0] != group_ids.shape[0]:
        #    print("Error: Input scores and group ID's not not have the same batch size")
        #    input()

        all_groups = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]

        n_groups = len(group_ids.unique())

        all_ones = torch.ones(group_ids.shape)

        factors  = {}
        factors[2] = {0.0:0.1, 1.0:0.7}
        factors[3] = {0.0:0.07, 1.0:0.15, 2.0:0.8}
        factors[4] = {0.0:0.05, 1.0:0.18, 2.0:0.3,  3.0:0.9}
        factors[5] = {0.0:0.04, 1.0:0.24, 2.0:0.38, 3.0:0.55, 4.0:1.2}
        factors[6] = {0.0:0.03, 1.0:0.30, 2.0:0.49, 3.0:0.68, 4.0:0.85, 5.0:1.6}
        factors[7] = {0.0:0.02, 1.0:0.28, 2.0:0.44, 3.0:0.6,  4.0:0.90, 5.0:1.125, 6.0:1.8}

        factors = factors[args.multi_groups]

        constraint_dict = {}
        for id in all_groups:
            if id in group_ids:
                this_group = (group_ids == id).float()
                print("id = {}".format(factors[id]))
                input()
                f = (1/factors[id])*this_group/(this_group).sum(1).reshape(-1,1)  -  all_ones/(all_ones).sum(1).reshape(-1,1)
                v = exposure.repeat(f.shape[0],1)
                v_unsq = v.unsqueeze(1)
                f_unsq = f.unsqueeze(1).permute(0,2,1)
                constraint_dict[id] = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1)


        #  Unweighted only for now
        #M0 = 1.0 if (disp_type == 'disp0') else group0_merit
        #M1 = 1.0 if (disp_type == 'disp0') else group1_merit
        #f =  M1*(1 - group_ids)/(1 - group_ids).sum(1).reshape(-1,1) -  M0*group_ids/group_ids.sum(1).reshape(-1,1)

        # The fairness constraint should be:
        # f^T P v = 0
        # useful form here is
        # (v f^T) P*  = 0
        # where P* is P flattened (row-major)
        #f = group_ids/group_ids.sum(1).reshape(-1,1) - (1 - group_ids)/(1 - group_ids).sum(1).reshape(-1,1)
        #v = exposure.repeat(f.shape[0],1) # repeat to match dimensions of f (batch dim)

        # Set up v and f for outer product
        #v_unsq = v.unsqueeze(1)
        #f_unsq = f.unsqueeze(1).permute(0,2,1)
        #v_unsq = v.unsqueeze(1).permute(0,2,1)
        #f_unsq = f.unsqueeze(1)

        # Outer product v f^T
        #   unroll to match P*
        #   unsqueeze to make each a 1-row matrix
        #vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1) # this is still a batch
        #vXf = torch.bmm(v_unsq,f_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
        #fair_b = torch.zeros(nBatch,1)

        # JK Do we need to consider the computation graph wrt the group identity vectors?

        # Here we exploit x!=x for x==nan
        #vXf = torch.where(vXf==vXf, vXf, vXf.new_zeros(vXf.shape))

        #A = torch.cat( (DSMl.repeat(nBatch,1,1),vXf),1 )
        A = DSMl.repeat(nBatch,1,1)
        #torch.cat((I.repeat(3,1,1),X.unsqueeze(1)),1)   # X is 2D, cat each row of X to a copy of I
                                                         # need this in case vXf is incorporated into ineq matrix
        b = DSMr.repeat(nBatch,1)
    else:
        #print("No fairness indicated")
        A = DSMl.repeat(nBatch,1,1)
        b = DSMr.repeat(nBatch,1)


    A = A[0]
    b = b[0]
    G = G[0]
    h = h[0]


    A_rows = np.array( A.to_sparse().indices()[0] )
    A_cols = np.array( A.to_sparse().indices()[1] )
    A_vals = np.array( A.to_sparse().values() )
    A_rhs  = np.array( b )

    G_rows = np.array( G.to_sparse().indices()[0] )
    G_cols = np.array( G.to_sparse().indices()[1] )
    G_vals = np.array( G.to_sparse().values() )
    G_rhs  = np.array( h )


    #NEQ      =  vXf.squeeze(0)#.unsqueeze(0)
    #NEQ     = torch.cat(  (vXf,-vXf), 1  )
    #NEQrhs  = torch.Tensor([delta,delta])


    solver = pywraplp.Solver('ScheduleFromRankings',
                     pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    x = [  solver.NumVar(0, 1,"x[{a}]".format(a=i)) for i in range(0,N**2)  ]

    #b = b.tolist()
    #A = A.tolist()

    # constraints of the matrix A
    A_constr = [  solver.Constraint( b[i].item(),b[i].item() ) for i in range(0,A.shape[0])  ]

    for i in range(A.shape[0]):     # each row of A is a constraint
        for j in range(A.shape[1]):
            if A[i][j].item() != 0.0:

                A_constr[i].SetCoefficient( x[j], A[i][j].item() )

    #expos = solver.NumVar(0, 100000,"expos".format(a=i))
    # JK TODO make expos equal to the average exposure - calculate with the vector of all 1's
    #exp_constr = solver.Constraint( 0,0 )



    NEQ_constr = [  solver.Constraint( -delta,delta ) for i in range(0,len(constraint_dict))  ]
    i=0
    for k,v in constraint_dict.items():     # each row of A is a constraint
        v = v.flatten()
        for j in range(len(v)):
            if v[j].item() != 0.0:
                NEQ_constr[i]
                x[j]
                v[j]
                NEQ_constr[i].SetCoefficient( x[j], v[j].item() )
        i = i+1

    return solver,x      #torch.Tensor(x.X)#.view(N,N)
