#JK 0803

import torch, numpy, os, sys
from torch import nn
from evaluation import sample_double_stoch, multiple_sample_and_log_probability
from networksJK import PolicyLP



scores =  torch.Tensor([[ 0.7161,  0.8573,  0.2468,  0.7465, -0.2263,  0.0081,  0.3507,  0.2075,
                          0.0739,  0.2545,  0.0328,  0.4629,  0.3628,  0.1519,  0.3576, -0.3872,
                          0.1297,  0.3564,  0.0587, -0.3208],
                        [ 0.4916,  0.1974,  0.5085,  0.7465, -0.2026,  0.1018, -0.2020,  0.0155,
                          0.2705,  0.2143,  0.4667, -0.4390,  0.4057, -0.2159,  0.2518, -0.0217,
                          0.1297, -0.2160,  0.2303, -0.2090],
                        [ 0.2468, -0.1159,  0.7465,  0.3507,  0.2683,  0.3899,  0.4826,  0.2075,
                          0.2177, -0.4446,  0.0154,  0.3252, -0.2590, -0.2012,  0.4468, -0.2096,
                         -0.2160,  0.3564, -0.0594,  0.6037],
                        [-0.1092,  0.2369,  0.3056, -0.0403,  0.0081,  0.7804,  0.0856,  0.2800,
                          0.0154, -0.0152, -0.1959, -0.4018, -0.4649, -0.4390, -0.2012,  0.4173,
                          0.4468, -0.1444, -0.0217, -0.0098]])


model = PolicyLP(N=scores.shape[1], eps=1e-4, indices_A=None)

print("scores = ")
print( scores )
print("scores.size() = ")
print( scores.size() )

print("scores.repeat(1,1,scores.shape[1]).squeeze() = "  )
print( scores.repeat(1,1,scores.shape[1]).squeeze()      )
print("scores.repeat(1,1,scores.shape[1]).squeeze().size() = "  )
print( scores.repeat(1,1,scores.shape[1]).squeeze().size()      )


p_mat = model(scores.repeat(1,1,scores.shape[1]).squeeze())

p_mat = torch.stack([torch.eye(20) for _ in range(4)])

print("p_mat = ")
print( p_mat )
print("p_mat.size() = ")
print( p_mat.size() )
input()

sample_size = 4
#rankings = multiple_sample_and_log_probability(scores, sample_size, return_prob=False, batch=True)
#rankings = sample_double_stoch(p_mat[0],sample_size, return_prob=True, batch=False)
rankings = sample_double_stoch(p_mat,sample_size, return_prob=True, batch=True)



print("Done!")
print("rankings = ")
print( rankings )
