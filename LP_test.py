#JK 0803
# Just a scrap notebook for testing individual components

import torch, numpy, os, sys
from torch import nn
#from evaluation import sample_double_stoch, multiple_sample_and_log_probability
from networksJK import PolicyLP, PolicyLP_Plus, PolicyLP_PlusSP
#from evaluation import compute_dcg_rankings
from birkhoff import birkhoff_von_neumann_decomposition
import numpy as np
from evaluation import compute_dcg_rankings, evaluate_model, multiple_sample_and_log_probability, sample_double_stoch, compute_dcg_max #JK
from fairness_loss import GroupFairnessLoss
from matplotlib import pyplot as plt
from gurobi_rank import * #grb_solve


# No bath input, just single 1D tensor
def to_permutation(ranks):
    ranks = ranks.long()
    P = torch.zeros(len(ranks),len(ranks))
    for k in range(len(ranks)):
        P[k][ranks[k]-1] = 1

    return P

# here pmat is a single sample, i.e. not a batch
# same holds for  group_identities, position_bias_vector
def test_fairness(pmat, group_identities, position_bias_vector):


    v = position_bias_vector[:pmat.shape[1]]
    f = ( group_identities / group_identities.sum() )   -   ( (1-group_identities) / (1-group_identities).sum() )
    ret = torch.mv(pmat,v)
    ret = torch.dot(f,ret)
    return ret.item()


def test_fairness_alt(pmat, group_identities, position_bias_vector):

    v = position_bias_vector[:pmat.shape[1]]
    f = ( group_identities / group_identities.sum() )   -   ( (1-group_identities) / (1-group_identities).sum() )
    fTv = torch.mm(f.unsqueeze(0).T,v.unsqueeze(0))
    ret = torch.dot( fTv.flatten(), pmat.flatten() )

    return ret.item()





"""
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
"""

"These scores appeared right before PolicyLP_Plus leading to NaN"
scores = torch.Tensor([[ 8.7845e-02,  3.0729e-01,  1.3841e-01,  2.2779e-01, -2.3924e-01,
                          2.3625e-01, -1.7794e-01, -3.8440e-02,  1.9790e-01,  4.4483e-02,
                          6.6272e-01,  4.5028e-01,  3.1430e-01,  6.6546e-01,  1.1579e-01,
                          4.9145e-01,  2.8865e-01,  7.7143e-01,  8.4712e-02,  1.9706e-01],
                        #[ 1.9792e-01, -2.3924e-01,  4.1352e-01,  1.5531e-01,  1.0950e-02,
                        #  7.0820e-02,  9.9321e-02,  3.9382e-01, -3.8440e-02,  2.8947e-01,
                        # -1.0751e-01,  1.9790e-01,  1.6047e-01, -4.0301e-01,  3.7110e-01,
                        #  7.1352e-01,  2.6979e-01,  8.2892e-01,  5.7665e-01,  7.7763e-01],
                        [ 9.4937e-02, -2.3924e-01,  4.5744e-01,  4.5957e-01, -8.7312e-02,
                          3.2316e-01, -4.1939e-02,  3.9382e-01,  7.9789e-01, -8.5231e-02,
                          1.9790e-01,  2.6188e-01,  6.5783e-02, -4.3274e-02,  2.8851e-01,
                          6.6546e-01,  1.1579e-01,  5.4227e-01,  8.4712e-02,  4.5897e-01],
                        [ 1.9792e-01, -5.0770e-01,  1.3841e-01,  4.5957e-01, -1.4202e-01,
                         -1.0275e-01,  4.9831e-01,  2.3625e-01,  1.0950e-02, -2.0029e-01,
                          2.7416e-01,  1.1547e-04,  1.2828e-01,  1.9790e-01, -4.3274e-02,
                          2.8865e-01,  7.7143e-01,  7.1352e-01,  7.7763e-01,  4.5897e-01]])


#scores.repeat(1,1,scores.shape[1])
#scores.unsqueeze(0).view(scores.shape[0],-1,1)
#scores.unsqueeze(0).view(scores.shape[0],1,-1)
#torch.ones(scores.shape[0],1,scores.shape[1])
#torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), torch.ones(scores.shape[0],1,scores.shape[1])  )
#torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1),
#scores = torch.nn.functional.relu(scores)
"""
group_identities = torch.Tensor([[0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                                 #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
"""
group_identities = torch.Tensor([[1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
                                 #[0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
                                 #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 #[0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 1.],
                                 #[0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0.]])
                                 #[1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0.],
                                 #[1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
                                 #[1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                 #[1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
                                 #[1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                 #[1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                 #[1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                 #[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                 #[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.],
                                 [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])


n_trials = 25
#group_identities = torch.tril( torch.ones(n_trials,n_trials) )[:-1]




#torch.manual_seed(5)
scores = torch.rand(group_identities.shape)
#torch.manual_seed(0)
#numpy.random.seed(0)


# Not related to the above; taken from some random output
test_rankings = torch.tensor(  [[[19,  2,  6, 15,  8, 14, 13, 12,  7, 18,  3, 11,  1,  9,  5, 17, 16, 0, 10,  4],
                                 [ 1, 10,  4, 17, 18,  9, 16,  2,  5, 19,  3, 12, 14,  0,  8,  6, 13, 7, 15, 11]],

                                [[13, 19, 14,  4,  5,  3,  7, 10,  1,  0, 15,  9, 12,  2, 16, 17,  6, 8, 11, 18],
                                 [ 6, 19,  7, 15,  5,  9,  2, 13,  1,  0, 18, 12,  3,  8, 16, 10,  4, 14, 17, 11]],

                                [[13, 18,  0, 17,  4, 15,  5,  9,  8, 11, 14,  6,  7, 10,  3,  1, 12, 16,  2, 19],
                                 [ 4, 15, 16,  1,  6, 14,  3, 13,  7,  9, 19,  0,  5,  8,  2, 12, 11, 17, 18, 10]],

                                [[11, 17,  8, 15, 16,  7, 13,  1,  2,  4, 12,  6,  5,  0, 19,  9, 18, 14,  3, 10],
                                 [17,  2,  6, 12,  0, 10, 14, 13,  5,  8, 19, 16, 11,  1,  7,  4,  9, 15, 18,  3]]]).long()


test_rels =       torch.Tensor([[0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                #[0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                #[0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

test_rels = torch.rand(group_identities.shape)

cutoff = 100
group_identities = group_identities[:,:cutoff]
test_rels = test_rels[:,:cutoff]
scores = scores[:,:cutoff]


#p_mat = 0.05*torch.ones(20,20).repeat(4,1,1)
batch = group_identities.shape[0]



"""
fXv = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
for k in range(batch):
    #fXv = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), torch.ones(scores.shape[0],1,scores.shape[1])  ).permute(0,2,1).reshape(scores.shape[0],-1)
    p_mat = policy_lp(  fXv[k].unsqueeze(0), None )
    #scores_rep = scores.repeat(1,1,scores.shape[1])
    #p_mat = policy_lp(  torch.repeat_interleave(scores,scores.shape[1],1)[k].unsqueeze(0), None ) #group_identities[k].unsqueeze(0))
    #p_mat = policy_lp( scores_rep.squeeze()[k].unsqueeze(0), None ) #group_identities[k].unsqueeze(0))
    #p_mat = policy_lp( torch.rand(scores_rep.shape).squeeze()[k].unsqueeze(0), None ) #group_identities[k].unsqueeze(0))
    print("Iteration {} complete".format(k))
    print("p_mat = ")
    print( p_mat )
    input("##############")
"""
 #Jk note: Not every instance is infeasible, every instance show nan in verbose output


rel = test_rels

position_bias_vector = 1. / torch.arange(1.,100.)
test_dscts = ( 1.0 / torch.log2(torch.arange(scores.shape[1]).float() + 2) ).repeat(batch,1,1)

policy_lp = PolicyLP_Plus(N=scores.shape[1], eps = 0.1, position_bias_vector = position_bias_vector)
score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)

score_cross = torch.rand(score_cross.shape)

#p_mat = policy_lp(  score_cross, group_identities )

#print("test_fairness(p_mat[0], group_identities[0], position_bias_vector) = ")
#print( test_fairness(p_mat[0], group_identities[0], position_bias_vector) )

#p_mat_ort = ort_solve(20, score_cross[0].detach().numpy(), group_identities[0])
s,x = ort_setup(20, group_identities[0])
#s,x = ort_setup_Neq(20, group_identities[0], 0.01)
solver = ort_policyLP(s,x)

p_mat_ort = []
for i in range(score_cross.shape[0]):
    p_mat_ort.append( solver.solve(score_cross[i].detach().numpy()).reshape(20,20) )

p_mat_ort = np.array(p_mat_ort)


print("test_fairness( torch.Tensor(p_mat_ort[0]), group_identities[0], position_bias_vector ) = ")
print( test_fairness( torch.Tensor(p_mat_ort[0]), group_identities[0], position_bias_vector ) )

# Objective vals:
#print( np.dot(p_mat_ort, score_cross.detach().numpy()) )
#print( np.dot(p_mat[0].flatten().detach().numpy(), score_cross[0].detach().numpy()) )

#print("test_fairness(p_mat[0], group_identities[0], position_bias_vector) = ")
#print( test_fairness(p_mat[0], group_identities[0], position_bias_vector) )

p_mat_ort = torch.Tensor(p_mat_ort)

#print("p_mat_ort = ")
#print( p_mat_ort    )

dcg_max = compute_dcg_max(rel)

loss_a = torch.bmm( p_mat_ort, test_dscts.view(scores.shape[0],-1,1) )
loss_b = torch.bmm( rel.view(scores.shape[0],1,-1), loss_a ).squeeze()
loss_norm = loss_b / dcg_max
loss = loss_norm

print("loss (NDCG) = ")
print( loss )


# Take one element from here
policy = p_mat_ort[1].cpu().detach().numpy()    #  precision is lost by np->torch->np

decomp = birkhoff_von_neumann_decomposition(policy)


convex_coeffs, permutations = zip(*decomp)
permutations = torch.Tensor(permutations)
sample_size = 100000
#rolls = torch.multinomial(torch.Tensor(convex_coeffs),sample_size,replacement=True).numpy()
rolls = np.random.multinomial(sample_size, np.array(convex_coeffs))  # sample the permutations based on convex_coeffs
print("len(convex_coeffs) = ")
print( len(convex_coeffs) )

print("np.sort(convex_coeffs) = ")
print( np.sort(convex_coeffs) )

#print("rolls/rolls.sum() = ")
#print( rolls/rolls.sum()    )
p_sample = permutations[rolls]       # access the permutations

print("rolls.mean() = ")
print( rolls.mean()    )

sum = 0
for i in range(len(p_sample)):
    sum += test_fairness(p_sample[i],group_identities[0], position_bias_vector)
sum /= len(p_sample)

print("sum = ")
print( sum  )

print("len(convex_coeffs) = ")
print( len(convex_coeffs) )

print("np.sort(convex_coeffs) = ")
print( np.sort(convex_coeffs) )



input("Press enter to quit")
quit()

















#### SPO test

spo_coeffs = score_cross[0]
spo_group_ids = group_identities[0]

P_spo = grb_solve(20, spo_coeffs, spo_group_ids)



policy = P_spo.view(20,20).cpu().detach().numpy()
#poicy = policy

print("policy = ")
print( policy )

decomp = birkhoff_von_neumann_decomposition(policy)


convex_coeffs, permutations = zip(*decomp)
permutations = torch.Tensor(permutations)
sample_size = 2000
rolls = torch.multinomial(torch.Tensor(convex_coeffs),sample_size,replacement=True).numpy()
#rolls = np.random.multinomial(sample_size, np.array(convex_coeffs))  # sample the permutations based on convex_coeffs
p_sample = permutations[rolls]       # access the permutations


sum = 0
for i in range(len(p_sample)):
    sum += test_fairness(p_sample[i],group_identities[0], position_bias_vector)
sum /= sample_size

print("sum = ")
print( sum  )

print("len(convex_coeffs) = ")
print( len(convex_coeffs) )

print("np.sort(convex_coeffs) = ")
print( np.sort(convex_coeffs) )

input("waiting")

#### END SPO test










sample_size = 1000
P = p_mat.cpu().detach().numpy()
R = []
for it, policy in enumerate(P):
    decomp = birkhoff_von_neumann_decomposition(policy)

    # Rebuild the decomp
    """
    new_decomp = []
    for c,p in decomp:
        if c < 1e-8:
            new_decomp.append( (c,p) )
    decomp = new_decomp
    """
    #####################

    convex_coeffs, permutations = zip(*decomp)
    permutations = np.array(permutations)
    rolls = torch.multinomial(torch.Tensor(convex_coeffs),sample_size,replacement=True).numpy()
    #rolls = np.random.multinomial(sample_size, np.array(convex_coeffs))  # sample the permutations based on convex_coeffs
    p_sample = permutations[rolls]       # access the permutations
    r_sample = p_sample.argmax(2)        # convert to rankings
    r_sample = torch.tensor( r_sample )  # convert to same datatype as FULTR implementation
    R.append(r_sample)

rankings = torch.stack(R)
print("rankings.size() = ")
print( rankings.size() )

test_dscts = ( 1.0 / torch.log2(torch.arange(scores.shape[1]).float() + 2) ).repeat(batch,1,1)



a = torch.bmm( p_mat, test_dscts.view(batch,-1,1) )
b = torch.bmm( test_rels.view(batch,1,-1), a )


ndcgs, dcgs = compute_dcg_rankings(rankings, rel)
dcg_max = compute_dcg_max(rel)

loss_a = torch.bmm( p_mat, test_dscts.view(scores.shape[0],-1,1) )
loss_b = torch.bmm( rel.view(scores.shape[0],1,-1), loss_a ).squeeze()
loss_norm = loss_b / dcg_max
loss = -loss_norm


utility_list = ndcgs

reward_variance_list = []
rewards_list = []
#entropy_list.append(entropy.item())   JK  no entropy
train_ndcg_list = []
train_dcg_list = []
weight_list = []

reward_variance_list.append(utility_list.var(dim=1).mean().item())
rewards_list.append(utility_list.mean().item())
#entropy_list.append(entropy.item())   JK  no entropy
train_ndcg_list.append(ndcgs.mean(dim=1).sum().item())
train_dcg_list.append(dcgs.mean(dim=1).sum().item())
weight_list.append(rel.sum().item())

weight_sum = np.sum(weight_list)


indicator_disparity = GroupFairnessLoss.compute_multiple_group_disparity(rankings, rel,
                                                                         group_identities,
                                                                         0,
                                                                         0,
                                                                         position_bias_vector,
                                                                         "disp0",
                                                                         noise=False,
                                                                         en=0.0).mean(dim=-1)


print("scores = ")
print(scores)

print("group_identities = ")
print(group_identities)

print("p_mat = ")
print( p_mat )

print("test_dscts = ")
print( test_dscts )

print("test_rels = ")
print( test_rels )

print("p_mat.size() = ")
print( p_mat.size() )


print("ndcgs = ")
print( ndcgs )

print("dcgs = ")
print( dcgs )


print("loss = ")
print( loss )

print("indicator_disparity = ")
print( indicator_disparity  )

print("group_identities.sum(1) = ")
print( group_identities.sum(1)   )

print("group_identities = ")
print( group_identities    )

print("len(p_mat) = ")
print( len(p_mat) )

print("\nFairness violations:")
for i in range(len(p_mat)):
    print( test_fairness(p_mat[i],group_identities[i], position_bias_vector) )


print("\nFairness violations (Alt):")
for i in range(len(p_mat)):
    print( test_fairness_alt(p_mat[i],group_identities[i], position_bias_vector) )


print(   to_permutation(torch.Tensor([1,2,3,4,5]))   )

#rankings.size() =
#torch.Size([11, 500, 20])

fair_viols = []

for i, samples in enumerate(rankings):
    sum = 0
    for ranking in samples:
        v = test_fairness(  to_permutation(ranking) , group_identities[i], position_bias_vector  )
        sum += v
    fair_viols.append(sum/len(samples))

print("\nFairness violations (Sampled):")
for i in range(len(fair_viols)):
    print( fair_viols[i] )


input("Waitn")


P = p_mat[0].cpu().detach().numpy()
print("P = ")
print( P )


decomp = birkhoff_von_neumann_decomposition(P)
convex_coeffs, permutations = zip(*decomp)



# Rebuild the decomp

new_decomp = []
for c,p in decomp:
    if c < 1e-8:
        new_decomp.append( (c,p) )
decomp = new_decomp

#####################



print("decomp = ")
print( decomp  )

print("decomp[0] = ")
print( decomp[0]  )

print("len(decomp) = ")
print( len(decomp)    )

print("convex_coeffs = ")
print( convex_coeffs )

recomp = 0
for i in range(len(decomp)):
    recomp += decomp[i][0]*decomp[i][1]

print("recomp = ")
print( recomp )

print("P - recomp = ")
print( P - recomp )

print("np.max(P - recomp) = ")
print( np.max(P - recomp) )

print("np.sort( (P - recomp).flatten() ) = ")
print( np.sort( (P - recomp).flatten() ) )
# test fairness of the recomp, and then of the sampling over the decomp

print("P Fairness violation:")
print( test_fairness(torch.Tensor(P),group_identities[0], position_bias_vector) )

print("recomp Fairness violation:")
print( test_fairness(torch.Tensor(recomp),group_identities[0], position_bias_vector) )

print("recomp combined Fairness violation:")
viol = 0
for i in range(len(decomp)):
    viol += decomp[i][0]* test_fairness(torch.Tensor(decomp[i][1]),group_identities[0], position_bias_vector)
print(viol)


convex_coeffs, permutations = zip(*decomp)
permutations = np.array(permutations)
sample_size = 5000
rolls = torch.multinomial(torch.Tensor(convex_coeffs),sample_size,replacement=True).numpy()
#rolls = np.random.multinomial(sample_size, np.array(convex_coeffs))  # sample the permutations based on convex_coeffs
p_sample = permutations[rolls]       # access the permutations
#r_sample = p_sample.argmax(2)        # convert to rankings
#r_sample = torch.tensor( r_sample )  # convert to same datatype as FULTR implementation
#R.append(r_sample)
psum = 0
for x in p_sample:
    psum += x

x = x/len(x)

print("x.sum() = ")
print( x.sum() )

print("recomp Fairness violation (sampled):")
print( test_fairness(torch.Tensor(x),group_identities[0], position_bias_vector) )



#torch.sum(exposures[inds_g0]) / inds_g0.sum() - torch.sum(exposures[inds_g1]) / inds_g1.sum()
#plt.plot( range(len(indicator_disparity)),indicator_disparity )
#plt.show()
