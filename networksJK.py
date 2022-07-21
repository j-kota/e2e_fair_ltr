# 1.1 NETWORK STRUCTURE
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from qpth.qp import QPFunction, SpQPFunction
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import math
from ort_rank import *

class MLPNet(nn.Module):
    """
    Implement a MLP with  single hidden layer. The choice of activation
    function can be passed as argument when init the network
    """
    def __init__(self, options = {'activation':'relu'}):
        super(MLPNet, self).__init__()
        #self.fc1 = nn.Linear(options['num_feats'],2)
        self.fc1 = nn.Linear(options['num_feats'],int(options['num_feats']/2))
        self.fc2 = nn.Linear(int(options['num_feats']/2),1)
        if options['act_func'] =='relu':
          self.act_func = nn.ReLU()
        if options['act_func'] == 'tanh':
          self.act_func = nn.Tanh()
        elif options['act_func']=='sigmoid':
          self.act_func = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        output = self.fc1(x)
        output = self.act_func(output)
        output = self.fc2(output)
        return output




class LRNet(nn.Module):
    """
    Logistic Regression network (NO hidden layer). So no activation function here.
    """
    def __init__(self, options):
        super(LRNet, self).__init__()
        self.fc1 = nn.Linear(options['num_feats'],1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
      return self.fc1(x)




# MLPNet augmented with class indicators M/F as input
# Reason - we want the previous layers to be able to predict a suitable
#   cost vector based on the group composition of each batch
class AugMLPNet(nn.Module):
    """
    Implement a MLP with  single hidden layer. The choice of activation
    function can be passed as argument when init the network
    """
    def __init__(self, options = {'activation':'relu'}):
        super(AugMLPNet, self).__init__()
        #self.fc1 = nn.Linear(options['num_feats'],2)
        self.fc1 = nn.Linear(options['num_feats']+1,int(options['num_feats']/2))
        self.fc2 = nn.Linear(int(options['num_feats']/2),1)
        if options['act_func'] =='relu':
          self.act_func = nn.ReLU()
        if options['act_func'] == 'tanh':
          self.act_func = nn.Tanh()
        elif options['act_func']=='sigmoid':
          self.act_func = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        output = self.fc1(x)
        output = self.act_func(output)
        output = self.fc2(output)
        return output







#### New from JK ####
#
# Adapted from MLPCapNet above
#
#

#class MLPCapNet(nn.Module):
#    """
#    Implement a MLP with  single hidden layer. The choice of activation
#    function can be passed as argument when init the network
#    """
#    def __init__(self, options = {'activation':'relu'}):
#        super(MLPNet, self).__init__()
#        self.fc1 = nn.Linear(options['num_feats'],2)
#        self.fc2 = nn.Linear(int(options['num_feats']/2),1)
#        if options['activation'] =='relu':
#          self.act_func = nn.ReLU()
#        if options['activation'] == 'tanh':
#          self.act_func = nn.Tanh()
#        elif options['activation']=='sigmoid':
#          self.act_func = nn.Sigmoid()
#        self.sigmoid = nn.Sigmoid()
#
#        #JK
#        # 'total_size' should be the total # samples in the entire training set
#        CapLayerLP( options['total_size'],  )
#
#    def forward(self, x):
#        output = self.fc1(x)
#        output = self.act_func(output)
#        output = self.fc2(output)
#
#        # the entire batch goes into the LP (becomes the objective coefficients)
#        output_unrolled = output.view(-1,output.numel())
#        # reshape into a 'batch' of length 1 (format needed for QP layer)
#        output_unrolled = output_unrolled.unsqueeze(0)
#
#        CapLayerLP
#
#        return lp_out





# LP layer with quadratic regularization
#
# This assumes that the input will be unit length in the batch dimension
# So the input should be 'unrolled' within the DNN before passing to this layer
#
# Fairness model
# max sum y_i b_i
# st
# sum b_i <= C          only one constraint
#
# Since the input is the objective function,
# input and output have the same dimention
#
#
# Working notes:
# Being a max problem, be wary of the sign of Q and p
#
#
#

class CapLayerLP(nn.Module):
    def __init__(self, nFeatures, nineq=1, neq=0, eps=1e-4, C = 10,  indices_male = torch.tensor([]), fair_mode = 1):
        # C is the capacity
        # N is the total size of the dataset
        # indices_male are the indices of one of two protected groups in the dataset
        super(CapLayerLP, self).__init__()

        N_male = indices_male.sum().item() # number of male / class A in the set

        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self.n = nFeatures
        self.indices_male = indices_male

        # Empty Tensor
        e = Variable(torch.Tensor())


        # Vector for Fairness constraint LHS and RHS
        fair_vector = torch.ones(1,self.n)
        fair_vector = indices_male.unsqueeze(0).float()   #[0][indices_male] = 1


        # Capacity constrant LHS and RHS
        Clhs    = Variable( torch.ones(1,self.n).to(self._device) )
        Crhs    = Variable(  ( torch.ones(1) * C ).to(self._device)   )

        # Right and left hand side of the bounding matrix (x >= 0, x <= 1)
        Blhs = Variable( torch.cat((-torch.eye(self.n,self.n), torch.eye(self.n,self.n)),0 ).to(self._device) )
        Brhs = Variable( torch.cat((torch.zeros(self.n),torch.ones(self.n)) ).to(self._device) )

        # Fairness Inequality and Equality constraints (optional)
        FIlhs1 = e
        FIrhs1 = e
        FIlhs2 = e
        FIrhs2 = e

        FElhs = e
        FErhs = e

        fair_mode = 1
        if fair_mode == 0:
            FElhs  =  Variable(  fair_vector.to(self._device)          )
            FErhs  =  Variable(     torch.Tensor([math.floor(C*N_male/self.n)])    )
        elif fair_mode == 1:
            # sum_male b_i <= 1 + C*N_male/self.n
            FIlhs1 =  Variable(  fair_vector.to(self._device)    )
            FIrhs1 =  Variable(     torch.Tensor([C*N_male/self.n]) + 1    )
            # sum_male b_i =>  C*N_male/self.n
            FIlhs2 = -Variable(  fair_vector.to(self._device)    )
            FIrhs2 = -Variable(     torch.Tensor([C*N_male/self.n])        )
        #else:
        #    FElhs = e
        #    FErhs = e

        #Blhs = Variable( (-torch.eye(self.n,self.n)).to(self._device) )
        #Brhs = Variable( (torch.zeros(self.n)) ).to(self._device) )


        # Total inequalities
        self.G = torch.cat( (Clhs,Blhs,FIlhs1,FIlhs2),0  )
        self.h = torch.cat( (Crhs,Brhs,FIrhs1,FIrhs2),0  )
        # Total equalities
        self.A = FElhs #torch.cat(  e  )
        self.b = FErhs #torch.cat(  e  )

        self.nineq = self.G.size(0)
        self.neq   = self.A.size(0)
        self.eps   = eps


    def forward(self, x):
        nBatch = x.size(0)

        if nBatch > 1:
            print("Warning: nonunit batch encountered in CapLayerLP forward pass" )


        # Quadratic regularization strength
        qreg_stren = self.eps
        e = Variable(torch.Tensor())
        Q = qreg_stren*Variable(torch.eye(self.n)).to(self._device)
        G = self.G.unsqueeze(0).expand( nBatch, self.nineq, self.n )
        h = self.h.unsqueeze(0).expand( nBatch, self.nineq )

        if self.neq > 0:
            A = self.A.unsqueeze(0).expand( nBatch, self.neq, self.n )
            b = self.b.unsqueeze(0).expand( nBatch, self.neq )
        else:
            A = e
            b = e


        inputs = x
        x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A.double(), b.double()   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), e, e   )
        x = x.float()


        return x










# Testing batch-level application of fairness
# No capacity constraints
# Recieves a batch of size > 1
# Applies fairness over each batch
# LP variable size == batch size

# Working notes:
#   - how are the males indicated now? Does indices_male hold up?
#   - can we allow scrambled batches now?
#   - All batch sizes must be equal (LP size must be constant) - how/where to enforce?
# Where to construct the constraint matrices now? In the forward pass, since they change per batch now?
#
# Remember - M/F indicators are appended to 'labels'

# Variables:
#   - Should we force a structure on the batch composition -
#                                                          - M appear before F
#                                                          - constant number M/F per batch
#   - Should we constraint # accepted M per batch based on fraction M in the batch, or fraction M total?
#
# Since the constraint layer is applied in the training loop, try to make this generalize to CapLayerLP
#        when the batch size is 1
# Sanity check - does the a_train coming for label[:1] in the training loop match with indices_male?
# A: They are the same, but only with shuffle turned False in Dataloader
#
# Try for comparison? Put all F at the end, to make constraint row 111...11100...00, then, permute the result to correct order after LP layer
#
# Q: Can we get a better approach by constraining both M and F, or M against F, rather than just M? Think about interaction with objective function, with tanh weights
#
# Q: Maybe a good test case for varying constraints will be a contrived dataset where all batches have the same M/F composition?


class FairLayerBatchLP(nn.Module):
    def __init__(self, nFeatures, data_size, eps=1e-4, C = 10,  fair_mode = 1):
        # C is the capacity
        # N is the total size of the dataset
        # indices_male are the indices of one of two protected groups in the dataset
        super(FairLayerBatchLP, self).__init__()

        self.data_size = data_size
        self.fair_mode = fair_mode
        self.eps   = eps
        self.n = nFeatures
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.C = C

        """
        N_male = indices_male.sum().item() # number of male / class A in the set
        use_cuda = torch.cuda.is_available()


        self.n = nFeatures
        self.indices_male = indices_male

        # Empty Tensor
        e = Variable(torch.Tensor())

        # Vector for Fairness constraint LHS and RHS
        fair_vector = torch.ones(1,self.n)
        fair_vector = indices_male.unsqueeze(0).float()   #[0][indices_male] = 1
        # Capacity constrant LHS and RHS
        Clhs    = Variable( torch.ones(1,self.n).to(self._device) )
        Crhs    = Variable(  ( torch.ones(1) * C ).to(self._device)   )
        # Right and left hand side of the bounding matrix (x >= 0, x <= 1)
        Blhs = Variable( torch.cat((-torch.eye(self.n,self.n), torch.eye(self.n,self.n)),0 ).to(self._device) )
        Brhs = Variable( torch.cat((torch.zeros(self.n),torch.ones(self.n)) ).to(self._device) )
        # Fairness Inequality and Equality constraints (optional)
        FIlhs1 = e
        FIrhs1 = e
        FIlhs2 = e
        FIrhs2 = e
        FElhs = e
        FErhs = e

        fair_mode = 1
        if fair_mode == 0:
            FElhs  =  Variable(  fair_vector.to(self._device)          )
            FErhs  =  Variable(     torch.Tensor([C*N_male/self.n])    )
        elif fair_mode == 1:
            # sum_male b_i <= 1 + C*N_male/self.n
            FIlhs1 =  Variable(  fair_vector.to(self._device)    )
            FIrhs1 =  Variable(     torch.Tensor([C*N_male/self.n]) + 1    )
            # sum_male b_i =>  C*N_male/self.n
            FIlhs2 = -Variable(  fair_vector.to(self._device)    )
            FIrhs2 = -Variable(     torch.Tensor([C*N_male/self.n])        )
        #else:
        #    FElhs = e
        #    FErhs = e

        #Blhs = Variable( (-torch.eye(self.n,self.n)).to(self._device) )
        #Brhs = Variable( (torch.zeros(self.n)) ).to(self._device) )

        print("Clhs.size() = ")
        print(Clhs.size())
        print("Blhs.size() = ")
        print(Blhs.size())
        print("FIlhs1.size() = ")
        print(FIlhs1.size())
        print("FIlhs2.size() = ")
        print(FIlhs2.size())

        # Total inequalities
        self.G = torch.cat( (Clhs,Blhs,FIlhs1,FIlhs2),0  )
        self.h = torch.cat( (Crhs,Brhs,FIrhs1,FIrhs2),0  )
        # Total equalities
        self.A = FElhs #torch.cat(  e  )
        self.b = FErhs #torch.cat(  e  )

        self.nineq = self.G.size(0)
        self.neq   = self.A.size(0)
        """



    #
    # Note - Remember that indices_male is converted at the moment it's passed as argument to constructor in original version
    def update_constraints(self, indices_male):
        N_male = indices_male.sum().item() # number of male / class A in the set

        use_cuda = torch.cuda.is_available()
        #####self._device = torch.device("cuda" if use_cuda else "cpu")

        #####self.n = nFeatures
        self.indices_male = indices_male

        # Empty Tensor
        e = Variable(torch.Tensor())


        # Vector for Fairness constraint LHS and RHS
        fair_vector = torch.ones(1,self.n)
        fair_vector = indices_male.unsqueeze(0).float()   #[0][indices_male] = 1


        # Capacity constrant LHS and RHS
        Clhs    = Variable( torch.ones(1,self.n).to(self._device) )
        Crhs    = Variable(  ( torch.ones(1) * self.C ).to(self._device)   )

        # Right and left hand side of the bounding matrix (x >= 0, x <= 1)
        Blhs = Variable( torch.cat((-torch.eye(self.n,self.n), torch.eye(self.n,self.n)),0 ).to(self._device) )
        Brhs = Variable( torch.cat((torch.zeros(self.n),torch.ones(self.n)) ).to(self._device) )

        # Fairness Inequality and Equality constraints (optional)
        FIlhs1 = e
        FIrhs1 = e
        FIlhs2 = e
        FIrhs2 = e

        FElhs = e
        FErhs = e

        fair_mode = self.fair_mode
        if fair_mode == 0:
            # sum_male b_i == C*N_male/self.n
            FElhs  =  Variable(  fair_vector.to(self._device)               )
            FErhs  =  Variable(     torch.Tensor([self.C*N_male/self.data_size])    )   #Need to replace self.n with the whole dataset size
        elif fair_mode == 1:
            # sum_male b_i <= 1 + C*N_male/self.n
            FIlhs1 =  Variable(  fair_vector.to(self._device)    )
            FIrhs1 =  Variable(     torch.Tensor([self.C*N_male/self.data_size]) + 1    )
            # sum_male b_i =>  C*N_male/self.n
            FIlhs2 = -Variable(  fair_vector.to(self._device)    )
            FIrhs2 = -Variable(     torch.Tensor([self.C*N_male/self.data_size])        )
        else:
            print("Error: Invalid fairness constraint")
            quit()
            FElhs = e
            FErhs = e

        #Blhs = Variable( (-torch.eye(self.n,self.n)).to(self._device) )
        #Brhs = Variable( (torch.zeros(self.n)) ).to(self._device) )


        # Total inequalities
        self.G = torch.cat( (Clhs,Blhs,FIlhs1,FIlhs2),0  )
        self.h = torch.cat( (Crhs,Brhs,FIrhs1,FIrhs2),0  )
        # Total equalities
        self.A = FElhs #torch.cat(  e  )
        self.b = FErhs #torch.cat(  e  )

        self.nineq = self.G.size(0)
        self.neq   = self.A.size(0)



    def forward(self, x):
        nBatch = x.size(0)

        #if nBatch > 1:
        #    print("Warning: nonunit batch encountered in CapLayerLP forward pass" )


        # Quadratic regularization strength
        qreg_stren = self.eps
        e = Variable(torch.Tensor())
        Q = qreg_stren*Variable(torch.eye(self.n)).to(self._device)
        G = self.G.unsqueeze(0).expand( nBatch, self.nineq, self.n )
        h = self.h.unsqueeze(0).expand( nBatch, self.nineq )

        if self.neq > 0:
            A = self.A.unsqueeze(0).expand( nBatch, self.neq, self.n )
            b = self.b.unsqueeze(0).expand( nBatch, self.neq )
        else:
            A = e
            b = e


        inputs = x
        x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A.double(), b.double()   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), e, e   )
        x = x.float()


        return x







class FairLogitsBatchLP(nn.Module):
    def __init__(self, nFeatures, data_size, C = 10):
        # C is the capacity
        # N is the total size of the dataset
        # indices_male are the indices of one of two protected groups in the dataset
        super(FairLogitsBatchLP, self).__init__()

        self.data_size = data_size
        self.n = nFeatures
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.C = C

        """
        N_male = indices_male.sum().item() # number of male / class A in the set
        use_cuda = torch.cuda.is_available()


        self.n = nFeatures
        self.indices_male = indices_male

        # Empty Tensor
        e = Variable(torch.Tensor())

        # Vector for Fairness constraint LHS and RHS
        fair_vector = torch.ones(1,self.n)
        fair_vector = indices_male.unsqueeze(0).float()   #[0][indices_male] = 1
        # Capacity constrant LHS and RHS
        Clhs    = Variable( torch.ones(1,self.n).to(self._device) )
        Crhs    = Variable(  ( torch.ones(1) * C ).to(self._device)   )
        # Right and left hand side of the bounding matrix (x >= 0, x <= 1)
        Blhs = Variable( torch.cat((-torch.eye(self.n,self.n), torch.eye(self.n,self.n)),0 ).to(self._device) )
        Brhs = Variable( torch.cat((torch.zeros(self.n),torch.ones(self.n)) ).to(self._device) )
        # Fairness Inequality and Equality constraints (optional)
        FIlhs1 = e
        FIrhs1 = e
        FIlhs2 = e
        FIrhs2 = e
        FElhs = e
        FErhs = e

        fair_mode = 1
        if fair_mode == 0:
            FElhs  =  Variable(  fair_vector.to(self._device)          )
            FErhs  =  Variable(     torch.Tensor([C*N_male/self.n])    )
        elif fair_mode == 1:
            # sum_male b_i <= 1 + C*N_male/self.n
            FIlhs1 =  Variable(  fair_vector.to(self._device)    )
            FIrhs1 =  Variable(     torch.Tensor([C*N_male/self.n]) + 1    )
            # sum_male b_i =>  C*N_male/self.n
            FIlhs2 = -Variable(  fair_vector.to(self._device)    )
            FIrhs2 = -Variable(     torch.Tensor([C*N_male/self.n])        )
        #else:
        #    FElhs = e
        #    FErhs = e

        #Blhs = Variable( (-torch.eye(self.n,self.n)).to(self._device) )
        #Brhs = Variable( (torch.zeros(self.n)) ).to(self._device) )

        print("Clhs.size() = ")
        print(Clhs.size())
        print("Blhs.size() = ")
        print(Blhs.size())
        print("FIlhs1.size() = ")
        print(FIlhs1.size())
        print("FIlhs2.size() = ")
        print(FIlhs2.size())

        # Total inequalities
        self.G = torch.cat( (Clhs,Blhs,FIlhs1,FIlhs2),0  )
        self.h = torch.cat( (Crhs,Brhs,FIrhs1,FIrhs2),0  )
        # Total equalities
        self.A = FElhs #torch.cat(  e  )
        self.b = FErhs #torch.cat(  e  )

        self.nineq = self.G.size(0)
        self.neq   = self.A.size(0)
        """



    #
    # Note - Remember that indices_male is converted at the moment it's passed as argument to constructor in original version
    def update_constraints(self, indices_male):
        N_male = indices_male.sum().item() # number of male / class A in the set
        N_fem  = indices_male.numel() - N_male

        use_cuda = torch.cuda.is_available()
        #####self._device = torch.device("cuda" if use_cuda else "cpu")

        #####self.n = nFeatures
        self.indices_male = indices_male

        # Empty Tensor
        e = Variable(torch.Tensor())


        # Vector for Fairness constraint LHS and RHS
        #fair_vector = torch.ones(1,self.n)
        M_vector = indices_male.unsqueeze(0).float()   #[0][indices_male] = 1
        F_vector = 1 - M_vector


        # Capacity constrant LHS and RHS
        #Clhs    = Variable( torch.ones(1,self.n).to(self._device) )
        #Crhs    = Variable(  ( torch.ones(1) * self.C ).to(self._device)   )

        # Right and left hand side of the bounding matrix (x >= 0, x <= 1)
        Blhs = Variable( torch.cat((-torch.eye(self.n,self.n), torch.eye(self.n,self.n)),0 ).to(self._device) )
        Brhs = Variable( torch.cat((torch.zeros(self.n),torch.ones(self.n)) ).to(self._device) )

        # Fairness Inequality and Equality constraints (optional)
        FIlhs1 = e
        FIrhs1 = e
        FIlhs2 = e
        FIrhs2 = e

        FElhs = e
        FErhs = e

        """  fair_mode isn't used in this layer
        fair_mode = self.fair_mode
        if fair_mode == 0:
            # sum_male b_i == C*N_male/self.n
            FElhs  =  Variable(  fair_vector.to(self._device)               )
            FErhs  =  Variable(     torch.Tensor([self.C*N_male/self.data_size])    )   #Need to replace self.n with the whole dataset size
        elif fair_mode == 1:
            # sum_male b_i <= 1 + C*N_male/self.n
            FIlhs1 =  Variable(  fair_vector.to(self._device)    )
            FIrhs1 =  Variable(     torch.Tensor([self.C*N_male/self.data_size]) + 1    )
            # sum_male b_i =>  C*N_male/self.n
            FIlhs2 = -Variable(  fair_vector.to(self._device)    )
            FIrhs2 = -Variable(     torch.Tensor([self.C*N_male/self.data_size])        )
        else:
            print("Error: Invalid fairness constraint")
            quit()
            FElhs = e
            FErhs = e
        """


        FElhs  =  Variable(  (  (1/N_male)*M_vector - (1/N_fem)*F_vector  ).to(self._device)               )
        FErhs  =  Variable(     torch.Tensor([0])    )


        # Total inequalities
        #self.G = torch.cat( (Clhs,Blhs,FIlhs1,FIlhs2),0  )
        #self.h = torch.cat( (Crhs,Brhs,FIrhs1,FIrhs2),0  )
        # JK comment - capacity constraints removed in this layer

        self.G = Blhs
        self.h = Brhs

        #self.G = torch.cat( (Blhs,FElhs),0  )   # trash
        #self.h = torch.cat( (Brhs,FErhs),0  )


        # Total equalities
        self.A = FElhs #torch.cat(  e  )
        self.b = FErhs #torch.cat(  e  )

        self.nineq = self.G.size(0)
        self.neq   = self.A.size(0)

        """
        print("self.G = ")
        print( self.G )
        print("self.h = ")
        print( self.h )
        print("self.A = ")
        print( self.A )
        print("self.b = ")
        print( self.b )
        input()
        """


    # Objective:
    #   | p - b | ^ 2   =   b^2 - 2pb     where b^2 cancels as a constant term in the objective
    #   here p is input and b is the decision variable
    # No longer assumed integer
    def forward(self, x):
        nBatch = x.size(0)

        #if nBatch > 1:
        #    print("Warning: nonunit batch encountered in CapLayerLP forward pass" )

        # Quadratic regularization strength
        e = Variable(torch.Tensor())
        Q = Variable(torch.eye(self.n)).to(self._device)  # No epsilon term
        G = self.G.unsqueeze(0).expand( nBatch, self.nineq, self.n )
        h = self.h.unsqueeze(0).expand( nBatch, self.nineq )

        if self.neq > 0:
            A = self.A.unsqueeze(0).expand( nBatch, self.neq, self.n )
            b = self.b.unsqueeze(0).expand( nBatch, self.neq )
        else:
            A = e
            b = e


        inputs = x   #correct line
        #inputs = 0.5*torch.ones(x.shape)
        x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A.double(), b.double()   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), e, e   )

        ###
        # Remember that
        ###

        x = x.float()

        """
        print("After QPFunction")
        print("inputs = ")
        print( inputs   )
        print("output = ")
        print( x    )
        print("self.indices_male = ")
        print( self.indices_male )
        input()
        """

        return x










class RankLP(nn.Module):
    # nNodes is the number of nodes on the left and right
    def __init__(self, N=1, eps=1e-4, indices_A=None):
        super(RankLP, self).__init__()

        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self.eps  = eps
        self.N    = N


        # Empty Tensor
        e = Variable(torch.Tensor())


        # Fairness Inequality and Equality constraints (optional)

        if indices_A != None :
            print("Fairness constraints active")

            self.indices_A = indices_A
            indices_B = 1 - indices_A
            N_A  = indices_A.sum().item() # number of male / class A in the set
            N_B  = indices_B.numel() - N_A
            fair_vector = torch.cat(tuple([ indices_A/N_A - indices_B/N_B ]*N))
            #FElhs  =  Variable(  (  (1/N_male)*M_vector - (1/N_fem)*F_vector  ).to(self._device)    )
            FElhs  =  Variable( fair_vector ).to(self._device)
            FErhs  =  Variable(     torch.Tensor([0])    )

        else:
            # TODO: Do something to end up with A = e here
            print("Fairness constraints NOT active")

            FElhs = e
            FErhs = e

        # want:
        # 1/|A|*indices_A - 1/|B|*indices_B   =  0




        # Total inequalities
        #self.G = torch.cat( (Clhs,Blhs,FIlhs1,FIlhs2),0  )
        #self.h = torch.cat( (Crhs,Brhs,FIrhs1,FIrhs2),0  )
        # JK comment - capacity constraints removed in this layer

        #self.G = Blhs
        #self.h = Brhs

        #self.G = torch.cat( (Blhs,FElhs),0  )   # trash
        #self.h = torch.cat( (Brhs,FErhs),0  )


        # Total equalities
        self.A = FElhs.unsqueeze(0) #torch.cat(  e  )
        self.b = FErhs #torch.cat(  e  )

        self.neq   = self.A.size(0)


        ###### ####### ####### ########## #######






        ROWlhs    = Variable( torch.zeros(N,N**2)  )
        ROWrhs    = Variable(  ( torch.ones(N) )    )
        COLlhs    = Variable( torch.zeros(N,N**2)  )
        COLrhs    = Variable(  ( torch.ones(N) )    )
        # All values are positive
        POSlhs    = Variable(    -torch.eye(N**2,N**2)        )
        POSrhs    = Variable(    -torch.zeros(N**2)        )



        # Row sum constraints
        for row in range(N):
            ROWlhs[row,row*N:(row+1)*N] = 1.0

        # Column sum constraints
        for col in range(N):
            COLlhs[col,col:-1:N] = 1.0
        # fix the stupid issue of bottom left not filling
        COLlhs[-1,-1] = 1.0



        # Total inequalities
        self.G = torch.cat( (ROWlhs,COLlhs, POSlhs),0  )
        self.h = torch.cat( (ROWrhs,COLrhs, POSrhs),0  )
        self.Q =  self.eps*Variable(torch.eye(self.N**2))

        # Difference from the Joaquims paper -
        #   inequality constraints used rather than equality



        self.nineq = self.G.size(0)

        # row  = position
        # col  = item / document


        # decision variable yij    1 <= i <= N1,  1 <= j <= N2
        # N1 nodes in G1, N2 nodes in G2
        #
        # max v^T y
        #
        # sum_i yij <= 1    all j in G1 vertices
        # sum_j yij <= 1    all i in G2 vertices
        #
        # yij >= 0  (is this necessary?)
        #
        #
        # ineq matrix w/ N rows, N^2 cols
        #
        # assume the variable y is an unrolled version of the matrix above
        # row-major order (row1, row2, ...)
        #
        # v[i*J+j] = v_ij is the weight matching node i on the left to j on the right





    def forward(self, x):
        nBatch = x.size(0)

        # Quadratic regularization strength
        qreg_stren = self.eps
        e = Variable(torch.Tensor())

        # Try these with and without the expand
        Q = self.Q.to(self._device)  #.unsqueeze(0).expand( nBatch, self.N**2,  self.N**2 )
        G = self.G.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq, self.N**2 )
        h = self.h.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq )

        A = self.A.to(self._device)
        b = self.b.to(self._device)
        #A = e.to(self._device)
        #b = e.to(self._device)

        """
        print("Inside matching layer forward:")
        print("Q.size() = ")
        print( Q.size() )
        print("G.size() = ")
        print( G.size() )
        print("h.size() = ")
        print( h.size() )
        print("A.size() = ")
        print( A.size() )
        print("b.size() = ")
        print( b.size() )
        print("x.size() = ")
        print( x.size() )
        """


        inputs = x
        x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A.double(), b.double()   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), e, e   )
        return x.view(-1,self.N,self.N).float()
        # shape returned from QPFunction should be nBatch,N**2




# Adapted from RankLP
# For fair ranking policy construction
# Trained with policy gradient method
class PolicyLP(nn.Module):
    # nNodes is the number of nodes on the left and right
    def __init__(self, N=1, eps=1e-4, indices_A=None):
        super(PolicyLP, self).__init__()

        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self.eps  = eps
        self.N    = N


        # Empty Tensor
        e = Variable(torch.Tensor())


        # Fairness Inequality and Equality constraints (optional)
        if indices_A != None :
            print("Fairness constraints active")

            self.indices_A = indices_A
            indices_B = 1 - indices_A
            N_A  = indices_A.sum().item() # number of male / class A in the set
            N_B  = indices_B.numel() - N_A
            fair_vector = torch.cat(tuple([ indices_A/N_A - indices_B/N_B ]*N))
            #FElhs  =  Variable(  (  (1/N_male)*M_vector - (1/N_fem)*F_vector  ).to(self._device)    )
            FElhs  =  Variable( fair_vector ).to(self._device)
            FErhs  =  Variable(     torch.Tensor([0])    )

        else:
            # TODO: Do something to end up with A = e here
            print("Fairness constraints NOT active")

            FElhs = e
            FErhs = e

        # want:
        # 1/|A|*indices_A - 1/|B|*indices_B   =  0


        # Total equalities
        self.A = FElhs#.unsqueeze(0) #torch.cat(  e  )
        self.b = FErhs #torch.cat(  e  )

        self.neq   = self.A.size(0)


        ###### ####### ####### ########## #######


        ROWlhs    = Variable( torch.zeros(N,N**2)  )
        ROWrhs    = Variable(  ( torch.ones(N) )    )
        COLlhs    = Variable( torch.zeros(N,N**2)  )
        COLrhs    = Variable(  ( torch.ones(N) )    )
        # All values are positive
        POSlhs    = Variable(    -torch.eye(N**2,N**2)        )
        POSrhs    = Variable(    -torch.zeros(N**2)        )



        # Row sum constraints
        for row in range(N):
            ROWlhs[row,row*N:(row+1)*N] = 1.0

        # Column sum constraints
        for col in range(N):
            COLlhs[col,col:-1:N] = 1.0
        # fix the stupid issue of bottom left not filling
        COLlhs[-1,-1] = 1.0



        # Total inequalities
        self.G = torch.cat( (ROWlhs,COLlhs, POSlhs),0  )
        self.h = torch.cat( (ROWrhs,COLrhs, POSrhs),0  )
        self.Q =  self.eps*Variable(torch.eye(self.N**2))

        # Difference from the Joaquims paper -
        #   inequality constraints used rather than equality



        self.nineq = self.G.size(0)

        # row  = position
        # col  = item / document


        # decision variable yij    1 <= i <= N1,  1 <= j <= N2
        # N1 nodes in G1, N2 nodes in G2
        #
        # max v^T y
        #
        # sum_i yij <= 1    all j in G1 vertices
        # sum_j yij <= 1    all i in G2 vertices
        #
        # yij >= 0  (is this necessary?)
        #
        #
        # ineq matrix w/ N rows, N^2 cols
        #
        # assume the variable y is an unrolled version of the matrix above
        # row-major order (row1, row2, ...)
        #
        # v[i*J+j] = v_ij is the weight matching node i on the left to j on the right





    def forward(self, x):
        nBatch = x.size(0)

        # Quadratic regularization strength
        qreg_stren = self.eps
        e = Variable(torch.Tensor())

        # Try these with and without the expand
        Q = self.Q.to(self._device)  #.unsqueeze(0).expand( nBatch, self.N**2,  self.N**2 )
        G = self.G.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq, self.N**2 )
        h = self.h.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq )

        A = self.A.to(self._device)
        b = self.b.to(self._device)
        #A = e.to(self._device)
        #b = e.to(self._device)


        print("Inside matching layer forward:")
        print("Q = ")
        print( Q )
        print("G = ")
        print( G )
        print("h = ")
        print( h )
        print("A = ")
        print( A )
        print("b = ")
        print( b )
        print("x = ")
        print( x )
        print("Q.size() = ")
        print( Q.size() )
        print("G.size() = ")
        print( G.size() )
        print("h.size() = ")
        print( h.size() )
        print("A.size() = ")
        print( A.size() )
        print("b.size() = ")
        print( b.size() )
        print("x.size() = ")
        print( x.size() )
        input()


        inputs = x
        x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A.double(), b.double()   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), e, e   )
        return x.view(-1,self.N,self.N).float()
        # shape returned from QPFunction should be nBatch,N**2







# Adapted from PolicyLP
class PolicyLP_Plus(nn.Module):
    # nNodes is the number of nodes on the left and right
    def __init__(self, N=1, eps=1e-1, position_bias_vector = torch.Tensor([])):
        super(PolicyLP_Plus, self).__init__()

        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self.eps  = eps
        self.N    = N
        #self.exposure = 1 / ( torch.arange(self.N)+2 ).float().log().to(self._device)  # this is what's used in FOEIR
        self.exposure = position_bias_vector[:N].float().to(self._device)
        # exposure v_j = 1 / log(1 + j)

        if self.exposure.numel() != self.N:
            print("Error: Exposure vector has unexpected dimensions")
            #print(self.exposure.numel())
            #print(self.N)

        # Empty Tensor
        e = Variable(torch.Tensor())


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



        # Total inequalities
        #self.G = torch.cat( (ROWlhs,COLlhs, POSlhs),0  )
        #self.h = torch.cat( (ROWrhs,COLrhs, POSrhs),0  )

        #self.DSMl = COLlhs
        #self.DSMr = COLrhs
        self.DSMl = torch.cat( (ROWlhs,COLlhs),0  ).to(self._device)
        self.DSMr = torch.cat( (ROWrhs,COLrhs),0  ).to(self._device)
        self.Q =  self.eps*Variable(torch.eye(self.N**2)).to(self._device)
        self.BDlhs =  torch.cat( (POSlhs,LEQ1lhs),0  ).to(self._device)
        self.BDrhs =  torch.cat( (POSrhs,LEQ1rhs),0  ).to(self._device)



        # Difference from the Joaquims paper -
        #   inequality constraints used rather than equality



        #self.nineq = self.G.size(0)

        # row  = position
        # col  = item / document

        # decision variable yij    1 <= i <= N1,  1 <= j <= N2
        # N1 nodes in G1, N2 nodes in G2
        #
        # max v^T y
        #
        # sum_i yij <= 1    all j in G1 vertices
        # sum_j yij <= 1    all i in G2 vertices
        #
        # yij >= 0  (is this necessary?)
        #
        #
        # ineq matrix w/ N rows, N^2 cols
        #nineq
        # assume the variable y is an unrolled version of the matrix above
        # row-major order (row1, row2, ...)
        #
        # v[i*J+j] = v_ij is the weight matching node i on the left to j on the right





    def forward(self, x, group_ids=None):
        nBatch = x.size(0)

        # Quadratic regularization strength
        qreg_stren = self.eps
        e = Variable(torch.Tensor())

        # Try these with and without the expand
        Q = self.Q  #.unsqueeze(0).expand( nBatch, self.N**2,  self.N**2 )
        DSMl = self.DSMl#.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq, self.N**2 )
        DSMr = self.DSMr#.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq )


        G = self.BDlhs.repeat(nBatch,1,1)
        h = self.BDrhs.repeat(nBatch,1)




        if( group_ids!=None ):

            if x.shape[0] != group_ids.shape[0]:
                print("Error: Input scores and group ID's not not have the same batch size")
                print(x.shape[0])
                print(group_ids.shape[0])
                input()

            # The fairness constraint should be:
            # f^T P v = 0
            # useful form here is
            # (v f^T) P*  = 0
            # where P* is P flattened (row-major)
            f = group_ids/group_ids.sum(1).reshape(-1,1) - (1 - group_ids)/(1 - group_ids).sum(1).reshape(-1,1)
            v = self.exposure.repeat(f.shape[0],1) # repeat to match dimensions of f (batch dim)

            # Set up v and f for outer product
            v_unsq = v.unsqueeze(1)
            f_unsq = f.unsqueeze(1).permute(0,2,1)
            #v_unsq = v.unsqueeze(1).permute(0,2,1)
            #f_unsq = f.unsqueeze(1)

            # Outer product v f^T
            #   unroll to match P*
            #   unsqueeze to make each a 1-row matrix
            vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
            #vXf = torch.bmm(v_unsq,f_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
            fair_b = torch.zeros(nBatch,1).to(self._device)

            # JK Do we need to consider the computation graph wrt the group identity vectors?

            # Here we exploit x!=x for x==nan
            #vXf = torch.where(vXf==vXf, vXf, vXf.new_zeros(vXf.shape))

            A = torch.cat( (DSMl.repeat(nBatch,1,1),vXf),1 )
            #torch.cat((I.repeat(3,1,1),X.unsqueeze(1)),1)   # X is 2D, cat each row of X to a copy of I
                                                             # need this in case vXf is incorporated into ineq matrix
            b = torch.cat( (DSMr.repeat(nBatch,1),fair_b),1 )
        else:
            A = DSMl.repeat(nBatch,1,1)
            b = DSMr.repeat(nBatch,1)



        inputs = x
        #x = QPFunction(verbose=1)(   Q.double(), -inputs.double(), G.double(), h.double(), e, e   )
        x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A.double(), b.double()   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), vXf.double(), b.double()   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), e, e   )
        return x.view(-1,self.N,self.N).float()
        # shape returned from QPFunction should be nBatch,N**2





def PolicyBlackboxWrapper(lambd, N, group_ids, disp_type, group0_merit, group1_merit, delta):

    class BlackboxWrap(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):

            y = []
            for i in range(len(x)):
                s,v = ort_setup_Neq(N, group_ids[i], disp_type, group0_merit, group1_merit, delta)
                solver = ort_policyLP(s,v)
                output = solver.solve(x[i])
                y.append(torch.Tensor(output))

            y = torch.stack(y)  #make sure this is doing the right thing over the batch
            ctx.save_for_backward( x,y )
            return y

        @staticmethod
        def backward(ctx, grad_output):

            x,y = ctx.saved_tensors
            x_p =  x +  grad_output * lambd

            y_lambd = []
            for i in range(len(x_p)):
                s,v = ort_setup_Neq(N, group_ids[i], disp_type, group0_merit, group1_merit, delta)
                solver = ort_policyLP(s,v)
                output = solver.solve(x_p[i])
                y_lambd.append(torch.Tensor(output))

            y_lambd = torch.stack(y_lambd)
            # multiply each gradient by the jacobian for the corresponding sample
            # then restack the results to preserve the batch gradients' format
            grad_input = - 1/lambd*(  y - y_lambd  )

            """
            print("y = ")
            print( y )
            print("y_lambd = ")
            print( y_lambd )
            print("y - y_lambd = ")
            print( y - y_lambd )
            print("grad_output = ")
            print( grad_output )
            print("grad_input = ")
            print( grad_input )
            input()
            """

            return grad_input

    return BlackboxWrap








def create_torch_LP(N=1, position_bias_vector = torch.Tensor([]), group_ids = None):

    exposure = position_bias_vector[:N].float()

    if exposure.numel() != N:
        print("Error: Exposure vector has unexpected dimensions")

    # Empty Tensor
    e = Variable(torch.Tensor())


    ROWlhs    = Variable( torch.zeros(N,N**2)  )
    ROWrhs    = Variable(  ( torch.ones(N) )    )
    COLlhs    = Variable( torch.zeros(N,N**2)  )
    COLrhs    = Variable(  ( torch.ones(N) )    )

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
    BDlhs =  torch.cat( (POSlhs,LEQ1lhs),0  )
    BDrhs =  torch.cat( (POSrhs,LEQ1rhs),0  )

    e = Variable(torch.Tensor())

    # Try these with and without the expand
    #Q = Q  #.unsqueeze(0).expand( nBatch, self.N**2,  self.N**2 )
    #DSMl = DSMl#.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq, self.N**2 )
    #DSMr = DSMr#.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq )


    G = BDlhs
    h = BDrhs


    f = group_ids/group_ids.sum(1).reshape(-1,1) - (1 - group_ids)/(1 - group_ids).sum(1).reshape(-1,1)
    v = exposure.repeat(f.shape[0],1) # repeat to match dimensions of f (batch dim)

    # Set up v and f for outer product
    v_unsq = v.unsqueeze(1)
    f_unsq = f.unsqueeze(1).permute(0,2,1)

    vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2)

    fair_b = torch.Tensor([0.0])




    A = torch.cat( (DSMl,vXf),0 )

    b = torch.cat( (DSMr,fair_b),0 )
    """
    print("G.shape = ")
    print( G.shape )
    print("h.shape = ")
    print( h.shape )
    print("A.shape = ")
    print( A.shape )
    print("b.shape = ")
    print( b.shape )
    """

    #inputs = x
    #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A.double(), b.double()   )

    return G,h,A,b




















# Adapted from PolicyLP
class PolicyLP_PlusSP(nn.Module):
    # nNodes is the number of nodes on the left and right
    def __init__(self, N=1, eps=1e-1, position_bias_vector = torch.Tensor([])):
        super(PolicyLP_PlusSP, self).__init__()

        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self.eps  = eps
        self.N    = N     # list length
        self.nx = N**2

        #self.exposure = 1 / ( torch.arange(self.N)+2 ).float().log().to(self._device)  # this is what's used in FOEIR
        self.exposure = position_bias_vector[:N].float().to(self._device)
        # exposure v_j = 1 / log(1 + j)

        if self.exposure.numel() != self.N:
            print("Error: Exposure vector has unexpected dimensions")

        # Empty Tensor
        e = Variable(torch.Tensor())


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



        # Total inequalities
        #self.G = torch.cat( (ROWlhs,COLlhs, POSlhs),0  )
        #self.h = torch.cat( (ROWrhs,COLrhs, POSrhs),0  )

        #self.DSMl = COLlhs
        #self.DSMr = COLrhs
        self.DSMl = torch.cat( (ROWlhs,COLlhs),0  ).to(self._device)
        self.DSMr = torch.cat( (ROWrhs,COLrhs),0  ).to(self._device)
        self.Q =  self.eps*Variable(torch.eye(self.N**2)).to(self._device)
        self.BDlhs =  torch.cat( (POSlhs,LEQ1lhs),0  ).to(self._device)
        self.BDrhs =  torch.cat( (POSrhs,LEQ1rhs),0  ).to(self._device)




    def forward(self, x, group_ids=None):
        nBatch = x.size(0)
        nx = self.nx





        # Quadratic regularization strength
        qreg_stren = self.eps
        e = Variable(torch.Tensor())

        # Try these with and without the expand
        Q = self.Q  #.unsqueeze(0).expand( nBatch, self.N**2,  self.N**2 )
        DSMl = self.DSMl#.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq, self.N**2 )
        DSMr = self.DSMr#.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq )


        G = self.BDlhs.repeat(nBatch,1,1)
        h = self.BDrhs.repeat(nBatch,1)




        if( group_ids!=None ):

            if x.shape[0] != group_ids.shape[0]:
                print("Error: Input scores and group ID's not not have the same batch size")
                input()

            # The fairness constraint should be:
            # f^T P v = 0
            # useful form here is
            # (v f^T) P*  = 0
            # where P* is P flattened (row-major)
            f = group_ids/group_ids.sum(1).reshape(-1,1) - (1 - group_ids)/(1 - group_ids).sum(1).reshape(-1,1)
            v = self.exposure.repeat(f.shape[0],1) # repeat to match dimensions of f (batch dim)

            # Set up v and f for outer product
            v_unsq = v.unsqueeze(1)
            f_unsq = f.unsqueeze(1).permute(0,2,1)
            #v_unsq = v.unsqueeze(1).permute(0,2,1)
            #f_unsq = f.unsqueeze(1)

            # Outer product v f^T
            #   unroll to match P*
            #   unsqueeze to make each a 1-row matrix
            vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
            #vXf = torch.bmm(v_unsq,f_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
            fair_b = torch.zeros(nBatch,1).to(self._device)

            # JK Do we need to consider the computation graph wrt the group identity vectors?

            # Here we exploit x!=x for x==nan
            #vXf = torch.where(vXf==vXf, vXf, vXf.new_zeros(vXf.shape))

            A = torch.cat( (DSMl.repeat(nBatch,1,1),vXf),1 )
            #torch.cat((I.repeat(3,1,1),X.unsqueeze(1)),1)   # X is 2D, cat each row of X to a copy of I
                                                             # need this in case vXf is incorporated into ineq matrix
            b = torch.cat( (DSMr.repeat(nBatch,1),fair_b),1 )
        else:
            A = DSMl.repeat(nBatch,1,1)
            b = DSMr.repeat(nBatch,1)





        ##### New sparse mode

        #spTensor = torch.cuda.sparse.DoubleTensor
        #iTensor = torch.cuda.LongTensor
        #dTensor = torch.cuda.DoubleTensor

        A = A[0].double()
        G = G[0].double()
        b = b[0].double()
        h = h[0].double()
        spTensor = torch.sparse.DoubleTensor
        iTensor  = torch.LongTensor
        dTensor  = torch.DoubleTensor

        Qi  = iTensor([range(nx), range(nx)])
        Qv  = Variable(dTensor(nx).fill_(self.eps))
        Qsz = torch.Size([nx, nx])


        t = A
        neq = t.shape[0]
        I = t != 0
        Av = Variable(dTensor(t[I]))
        Ai_np = torch.nonzero(t).T
        Ai = torch.stack((torch.LongTensor(Ai_np[0]),
                          torch.LongTensor(Ai_np[1])))#.cuda()
        Asz = torch.Size([neq, nx])


        t = G
        neq = t.shape[0]
        I = t != 0
        Gv = Variable(dTensor(t[I]))
        Gi_np = torch.nonzero(t).T
        Gi = torch.stack((torch.LongTensor(Gi_np[0]),
                          torch.LongTensor(Gi_np[1])))#.cuda()
        Gsz = torch.Size([neq, nx])


        """
        I = G != 0
        Gv = Variable(dTensor(G.double()[I]))
        Gi_np = np.nonzero(G)
        Gi = torch.stack((torch.LongTensor(Gi_np[0]),
                               torch.LongTensor(Gi_np[1])))#.cuda()
        Gsz = A.size() #torch.Size([neq, nx])
        """

        ############ END new

        """
        print("Qi = ")
        print( Qi    )
        print("Qv = ")
        print( Qv    )
        print("Qsz = ")
        print( Qsz    )
        print("Ai_np = ")
        print( Ai_np )
        print("Qi.size() = ")
        print( Qi.size()    )
        print("Qv.size() = ")
        print( Qv.size()    )
        print("Ai.size() = ")
        print( Ai.size()    )
        print("Av.size() = ")
        print( Av.size()    )
        print("Gi.size() = ")
        print( Gi.size()    )
        print("Gv.size() = ")
        print( Gv.size()    )
        """

        print("Qi.dtype = ")
        print( Qi.dtype )
        print("Qsz = ")
        print( Qsz )
        print("Gi.dtype = ")
        print( Gi.dtype )
        print("Gsz = ")
        print( Gsz )
        print("Ai.dtype = ")
        print( Ai.dtype )
        print("Asz = ")
        print( Asz )

        print("Qv.size() = ")
        print( Qv.size() )
        print("Qv.dtype = ")
        print( Qv.dtype )
        print("x.double().dtype = ")
        print( x.double().dtype )
        print("Gv.dtype = ")
        print( Gv.dtype )
        print("h.dtype = ")
        print( h.dtype )
        print("Av.dtype = ")
        print( Av.dtype )
        print("b.dtype = ")
        print( b.dtype )

        inputs = x.double()
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A.double(), b.double()   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A.double(), b.double()   )
        x = SpQPFunction(Qi, Qsz, Gi, Gsz, Ai, Asz, verbose=-1)(
                Qv.expand(nBatch, Qv.size(0)),
                inputs,
                Gv.expand(nBatch, Gv.size(0)),
                h.expand(nBatch, h.size(0)),
                Av.expand(nBatch, Av.size(0)),
                b.expand(nBatch, b.size(0))      )

        return x.view(-1,self.N,self.N).float()

        # JK note - to see hoe to assemble sparse matrices- sudoku models.py line 198




# JK - check - unfinished
# Haven't started work
# delta is the allowed fairness gap
class PolicyLP_PlusNeq(nn.Module):
    # nNodes is the number of nodes on the left and right
    def __init__(self, N=1, eps=1e-1, position_bias_vector = torch.Tensor([]), delta = 0.0):
        super(PolicyLP_PlusNeq, self).__init__()

        self.delta = delta

        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")

        self.eps  = eps
        self.N    = N
        self.exposure = position_bias_vector[:N].float().to(self._device)


        if self.exposure.numel() != self.N:
            print("Error: Exposure vector has unexpected dimensions")

        # Empty Tensor
        e = Variable(torch.Tensor())


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

        self.DSMl = torch.cat( (ROWlhs,COLlhs),0  ).to(self._device)
        self.DSMr = torch.cat( (ROWrhs,COLrhs),0  ).to(self._device)
        self.Q =  self.eps*Variable(torch.eye(self.N**2)).to(self._device)
        self.BDlhs =  torch.cat( (POSlhs,LEQ1lhs),0  ).to(self._device)
        self.BDrhs =  torch.cat( (POSrhs,LEQ1rhs),0  ).to(self._device)




    def forward(self, x, group_ids=None):
        nBatch = x.size(0)

        delta = self.delta

        # Quadratic regularization strength
        qreg_stren = self.eps
        e = Variable(torch.Tensor())

        # Try these with and without the expand
        Q = self.Q  #.unsqueeze(0).expand( nBatch, self.N**2,  self.N**2 )
        DSMl = self.DSMl#.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq, self.N**2 )
        DSMr = self.DSMr#.to(self._device)   #.unsqueeze(0).expand( nBatch, self.nineq )


        G = self.BDlhs.repeat(nBatch,1,1)
        h = self.BDrhs.repeat(nBatch,1)



        if( group_ids!=None ):

            if x.shape[0] != group_ids.shape[0]:
                print("Error: Input scores and group ID's not not have the same batch size")
                input()

            # The fairness constraint should be:
            # f^T P v = 0
            # useful form here is
            # (v f^T) P*  = 0
            # where P* is P flattened (row-major)
            f = group_ids/group_ids.sum(1).reshape(-1,1) - (1 - group_ids)/(1 - group_ids).sum(1).reshape(-1,1)
            v = self.exposure.repeat(f.shape[0],1) # repeat to match dimensions of f (batch dim)

            # Set up v and f for outer product
            v_unsq = v.unsqueeze(1)
            f_unsq = f.unsqueeze(1).permute(0,2,1)
            #v_unsq = v.unsqueeze(1).permute(0,2,1)
            #f_unsq = f.unsqueeze(1)

            # Outer product v f^T
            #   unroll to match P*
            #   unsqueeze to make each a 1-row matrix
            vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
            #vXf = torch.bmm(v_unsq,f_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
            fair_b = torch.zeros(nBatch,1).to(self._device)

            # JK Do we need to consider the computation graph wrt the group identity vectors?

            # Here we exploit x!=x for x==nan
            #vXf = torch.where(vXf==vXf, vXf, vXf.new_zeros(vXf.shape))

            A = torch.cat( (DSMl.repeat(nBatch,1,1),vXf),1 )
            #torch.cat((I.repeat(3,1,1),X.unsqueeze(1)),1)   # X is 2D, cat each row of X to a copy of I
                                                             # need this in case vXf is incorporated into ineq matrix
            b = torch.cat( (DSMr.repeat(nBatch,1),fair_b),1 )
        else:
            A = DSMl.repeat(nBatch,1,1)
            b = DSMr.repeat(nBatch,1)

        NEQ     = torch.cat(  (vXf,-vXf), 1  )
        NEQrhs  = torch.Tensor([delta,delta])


        # v x  <=  delta
        #-v x  <=  delta


        inputs = x
        #x = QPFunction(verbose=1)(   Q.double(), -inputs.double(), G.double(), h.double(), e, e   )
        x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A.double(), b.double()   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), vXf.double(), b.double()   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), e, e   )
        return x.view(-1,self.N,self.N).float()
        # shape returned from QPFunction should be nBatch,N**2








"""

def update_constraints(self, indices_male):
    N_male = indices_male.sum().item() # number of male / class A in the set
    N_fem  = indices_male.numel() - N_male

    use_cuda = torch.cuda.is_available()
    #####self._device = torch.device("cuda" if use_cuda else "cpu")

    #####self.n = nFeatures
    self.indices_male = indices_male

    # Empty Tensor
    e = Variable(torch.Tensor())


    # Vector for Fairness constraint LHS and RHS
    #fair_vector = torch.ones(1,self.n)
    M_vector = indices_male.unsqueeze(0).float()   #[0][indices_male] = 1
    F_vector = 1 - M_vector


    # Capacity constrant LHS and RHS
    #Clhs    = Variable( torch.ones(1,self.n).to(self._device) )
    #Crhs    = Variable(  ( torch.ones(1) * self.C ).to(self._device)   )

    # Right and left hand side of the bounding matrix (x >= 0, x <= 1)
    Blhs = Variable( torch.cat((-torch.eye(self.n,self.n), torch.eye(self.n,self.n)),0 ).to(self._device) )
    Brhs = Variable( torch.cat((torch.zeros(self.n),torch.ones(self.n)) ).to(self._device) )

    # Fairness Inequality and Equality constraints (optional)
    FIlhs1 = e
    FIrhs1 = e
    FIlhs2 = e
    FIrhs2 = e

    FElhs = e
    FErhs = e


    fair_mode = self.fair_mode
    if fair_mode == 0:
        # sum_male b_i == C*N_male/self.n
        FElhs  =  Variable(  fair_vector.to(self._device)               )
        FErhs  =  Variable(     torch.Tensor([self.C*N_male/self.data_size])    )   #Need to replace self.n with the whole dataset size
    elif fair_mode == 1:
        # sum_male b_i <= 1 + C*N_male/self.n
        FIlhs1 =  Variable(  fair_vector.to(self._device)    )
        FIrhs1 =  Variable(     torch.Tensor([self.C*N_male/self.data_size]) + 1    )
        # sum_male b_i =>  C*N_male/self.n
        FIlhs2 = -Variable(  fair_vector.to(self._device)    )
        FIrhs2 = -Variable(     torch.Tensor([self.C*N_male/self.data_size])        )
    else:
        print("Error: Invalid fairness constraint")
        quit()
        FElhs = e
        FErhs = e



    FElhs  =  Variable(  (  (1/N_male)*M_vector - (1/N_fem)*F_vector  ).to(self._device)               )
    FErhs  =  Variable(     torch.Tensor([0])    )


    # Total inequalities
    #self.G = torch.cat( (Clhs,Blhs,FIlhs1,FIlhs2),0  )
    #self.h = torch.cat( (Crhs,Brhs,FIrhs1,FIrhs2),0  )
    # JK comment - capacity constraints removed in this layer

    self.G = Blhs
    self.h = Brhs

    #self.G = torch.cat( (Blhs,FElhs),0  )   # trash
    #self.h = torch.cat( (Brhs,FErhs),0  )


    # Total equalities
    self.A = FElhs #torch.cat(  e  )
    self.b = FErhs #torch.cat(  e  )

    self.nineq = self.G.size(0)
    self.neq   = self.A.size(0)
"""







"""    Initial attempt - thought I had to rebuild constraints in the forward pass
class FairLayerBatchLP(nn.Module):
    def __init__(self, nFeatures, nineq=1, neq=0, eps=1e-4, C = 10,  indices_male = torch.tensor([]), fair_mode = 1, cap_mode = 1):
        # C is the capacity
        # N is the total size of the dataset
        # indices_male are the indices of one of two protected groups in the dataset
        super(FairLayerBatchLP, self).__init__()

        self.nFeatures = nFeatures
        self.C = C

        self.indices_male = indices_male


        use_cuda = torch.cuda.is_available()
        self._device = torch.device("cuda" if use_cuda else "cpu")



        self.eps   = eps


    def forward(self, x):
        nBatch = x.size(0)


        ################################### Previously in the constructor ##############################################


        self_n = self.nFeatures
        #self.indices_male = indices_male

        # Empty Tensor
        e = Variable(torch.Tensor())


        # Vector for Fairness constraint LHS and RHS
        fair_vector = torch.ones(1,self_n)
        fair_vector = self.indices_male.unsqueeze(0).float()   #[0][indices_male] = 1
        N_male = self.indices_male.sum().item() # number of male / class A in the set

        # Capacity constrant LHS and RHS
        Clhs    = Variable( torch.ones(1,self_n).to(self._device) )
        Crhs    = Variable(  ( torch.ones(1) * self.C ).to(self._device)   )

        # Right and left hand side of the bounding matrix (x >= 0, x <= 1)
        Blhs = Variable( torch.cat((-torch.eye(self_n,self_n), torch.eye(self_n,self_n)),0 ).to(self._device) )
        Brhs = Variable( torch.cat((torch.zeros(self_n),torch.ones(self_n)) ).to(self._device) )

        # Fairness Inequality and Equality constraints (optional)
        FIlhs1 = e
        FIrhs1 = e
        FIlhs2 = e
        FIrhs2 = e

        FElhs = e
        FErhs = e

        fair_mode = 1
        if fair_mode == 0:
            # sum_male b_i == 1 + C*N_male/self.n
            FElhs  =  Variable(  fair_vector.to(self._device)          )
            FErhs  =  Variable(     torch.Tensor([self.C*N_male/self_n])    )
        elif fair_mode == 1:
            # sum_male b_i <= 1 + C*N_male/self.n
            FIlhs1 =  Variable(  fair_vector.to(self._device)    )
            FIrhs1 =  Variable(     torch.Tensor([self.C*N_male/self_n]) + 1    )
            # sum_male b_i =>  C*N_male/self.n
            FIlhs2 = -Variable(  fair_vector.to(self._device)    )
            FIrhs2 = -Variable(     torch.Tensor([self.C*N_male/self_n])        )
        #else:
        #    FElhs = e
        #    FErhs = e

        #Blhs = Variable( (-torch.eye(self.n,self.n)).to(self._device) )
        #Brhs = Variable( (torch.zeros(self.n)) ).to(self._device) )


        # Total inequalities
        self_G = torch.cat( (Clhs,Blhs,FIlhs1,FIlhs2),0  )
        self_h = torch.cat( (Crhs,Brhs,FIrhs1,FIrhs2),0  )
        self_nineq = self_G.size(0)

        # Total equalities
        self_A = FElhs #torch.cat(  e  )
        self_b = FErhs #torch.cat(  e  )
        self_neq   = self_A.size(0)
        ####################################################################################






        # Quadratic regularization strength
        qreg_stren = self.eps
        e = Variable(torch.Tensor())
        Q = qreg_stren*Variable(torch.eye(self_n)).to(self._device)
        G = self_G.unsqueeze(0).expand( nBatch, self_nineq, self_n )
        h = self_h.unsqueeze(0).expand( nBatch, self_nineq )

        if self_neq > 0:
            A = self_A.unsqueeze(0).expand( nBatch, self_neq, self_n )
            b = self_b.unsqueeze(0).expand( nBatch, self_neq )
        else:
            A = e
            b = e


        inputs = x
        x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), A.double(), b.double()   )
        #x = QPFunction(verbose=-1)(   Q.double(), -inputs.double(), G.double(), h.double(), e, e   )
        x = x.float()


        return x
"""
