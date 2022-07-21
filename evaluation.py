import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
from models import convert_vars_to_gpu

from fairness_loss import get_group_identities, get_group_merits, \
    GroupFairnessLoss, BaselineAshudeepGroupFairnessLoss, test_fairness #JK

# JK
from networksJK import PolicyLP, PolicyLP_Plus, PolicyLP_PlusNeq
from birkhoff import birkhoff_von_neumann_decomposition
import time
from models import LinearModel # JK
import pickle
import pandas as pd
from ort_rank import *

def sample_multiple_ranking(probs, sample_size):
    candidate_set_size = probs.shape[0]
    ranking = torch.multinomial(
        probs.expand(
            sample_size,
            -1
        ),
        num_samples=candidate_set_size,
        replacement=False
    )
    return ranking


def sample_ranking(probs, output_propensities=True):
    propensity = 1.0
    candidate_set_size = probs.shape[0]
    ranking = torch.multinomial(
        probs,
        num_samples=candidate_set_size,
        replacement=False
    )
    if output_propensities:
        for i in range(candidate_set_size):
            propensity *= probs[ranking[i]]
            probs[ranking[i]] = 0.0
            probs = probs / probs.sum()
        return ranking, propensity
    else:
        return ranking


def multiple_sample_and_log_probability(
        scores, sample_size, return_prob=True, batch=False):
    #return_prob=True #JK remove
    if not batch:
        assert scores.dim() == 1
        subtracts = scores.new_zeros((sample_size, scores.size(0)))
        batch_index = torch.arange(sample_size, device=scores.device)
        if return_prob:
            log_probs = torch.zeros_like(subtracts, dtype=torch.float)
        rankings = []
        for j in range(scores.size(0)):
            probs = nn.functional.softmax(scores - subtracts, dim=1) + 1e-10
            posj = torch.multinomial(probs, 1).squeeze(-1)
            rankings.append(posj)
            if return_prob:
                log_probs[:, j] = probs[batch_index, posj].log()
            subtracts[batch_index, posj] = scores[posj] + 1e6
        rankings = torch.stack(rankings, dim=1)
        if return_prob:
            log_probs = log_probs.sum(dim=1)
            return rankings, log_probs
        else:
            return rankings
    else:
        assert scores.dim() == 2
        batch_size, candidiate_size = scores.size(0), scores.size(1)
        subtracts = scores.new_zeros((batch_size, sample_size, candidiate_size))
        batch_index = torch.arange(
            batch_size, device=scores.device).unsqueeze(1).expand(
            batch_size, sample_size)
        sample_index = torch.arange(
            sample_size, device=scores.device).expand(
            batch_size, sample_size)
        if return_prob:
            log_probs = torch.zeros_like(subtracts, dtype=torch.float)
        rankings = []
        for j in range(scores.size(1)):
            probs = nn.functional.softmax(
                scores.unsqueeze(1) - subtracts, dim=-1) + 1e-10
            posj = torch.multinomial(
                probs.reshape(
                    batch_size * sample_size,
                    -1
                ),
                1
            ).squeeze(-1).reshape(batch_size, sample_size)
            rankings.append(posj)
            if return_prob:
                log_probs[:, :, j] = probs[batch_index,
                                           sample_index, posj].log()
            subtracts[batch_index, sample_index,
                      posj] = scores[batch_index, posj] + 1e6
        rankings = torch.stack(rankings, dim=-1)
        if return_prob:
            log_probs = log_probs.sum(dim=-1)
            return rankings, log_probs
        else:
            return rankings



# JK variant of multiple_sample_and_log_probability but for doubly stochastic matrix
# assume that probs is a (batch of) doubly stochastic matrix
def sample_double_stoch(
        prob_mat, sample_size, return_prob=True, batch=False):
        #scores, sample_size, return_prob=True, batch=False):

    #return_prob=True #JK remove
    if not batch:
        assert prob_mat.dim() == 2
        #assert scores.dim() == 1
        masks = prob_mat.new_ones((sample_size, prob_mat.size(1)), device=prob_mat.device)
        #subtracts = scores.new_zeros((sample_size, scores.size(0)))
        batch_index = torch.arange(sample_size, device=prob_mat.device)
        #batch_index = torch.arange(sample_size, device=scores.device)
        if return_prob:
            log_probs = prob_mat.new_zeros((sample_size, prob_mat[0].size(0))).float()
            #log_probs = torch.zeros_like(subtracts, dtype=torch.float)
        rankings = []
        for j in range(prob_mat.size(0)):
        #for j in range(scores.size(0)):
            probs_sample_j = masks*prob_mat[j]  #this has sample_size rows, and prob_mat[j].size() columns
            probs = probs_sample_j /  probs_sample_j.sum(1).reshape(-1,1)
            #probs = nn.functional.softmax(scores - subtracts, dim=1) + 1e-10
            posj = torch.multinomial(probs, 1).squeeze(-1)
            rankings.append(posj)
            if return_prob:
                log_probs[:, j] = probs[batch_index, posj].clone().log()  # JK clone test
            masks[batch_index,posj] = 0
            masks = masks.clone()  # JK clone test
            #subtracts[batch_index, posj] = scores[posj] + 1e6
        rankings = torch.stack(rankings, dim=1)
        if return_prob:
            log_probs = log_probs.sum(dim=1)
            return rankings, log_probs
        else:
            return rankings
    else:
        assert prob_mat.dim() == 3
        #assert scores.dim() == 2
        batch_size, candidiate_size = prob_mat.size(0), prob_mat.size(1)
        #batch_size, candidiate_size = scores.size(0), scores.size(1)
        masks = prob_mat.new_ones((batch_size, sample_size, candidiate_size), device=prob_mat.device)
        #subtracts = scores.new_zeros((batch_size, sample_size, candidiate_size))
        batch_index = torch.arange(
            batch_size, device=prob_mat.device).unsqueeze(1).expand(
            #batch_size, device=scores.device).unsqueeze(1).expand(
            batch_size, sample_size)
        sample_index = torch.arange(
            sample_size, device=prob_mat.device).expand(
            #sample_size, device=scores.device).expand(
            batch_size, sample_size)
        if return_prob:
            log_probs = torch.zeros_like(masks, dtype=torch.float)
            #log_probs = torch.zeros_like(subtracts, dtype=torch.float)
        rankings = []
        for j in range(prob_mat.size(1)):
            ####probs_sample_j =  masks[:,None]*prob_mat[:,j,:]    # want to take prob_mat row j for all samples in batch
            probs = masks*prob_mat[:,j,:].unsqueeze(1) # JK equivalent to next line
            #probs = torch.stack( [masks[k]*prob_mat[:,j,:][k] for k in range(batch_size)] ).clone() # JK clone test
            #probs = nn.functional.softmax(
            #    scores.unsqueeze(1) - subtracts, dim=-1) + 1e-10
            #if probs.min().item() < 0.0:  # JK remove

            posj = torch.multinomial(
                    probs.reshape(batch_size*sample_size,-1) ,1
                                    ).squeeze(-1).reshape(batch_size, sample_size)
            #posj = torch.multinomial(
            #    probs.reshape(
            #        batch_size * sample_size,
            #        -1
            #    ),
            #    1
            #).squeeze(-1).reshape(batch_size, sample_size)
            rankings.append(posj)
            if return_prob:
                log_probs[:, :, j] = probs[batch_index,sample_index, posj].clone().log()  # JK clone test
            masks = masks.clone()   # JK clone test
            masks[batch_index, sample_index,posj] = 0

            #subtracts[batch_index, sample_index,
            #          posj] = scores[batch_index, posj] + 1e6
        rankings = torch.stack(rankings, dim=-1)
        if return_prob:
            log_probs = log_probs.sum(dim=-1)
            return rankings, log_probs
        else:
            return rankings





def compute_average_rank(rankings,
                         relevance_vector,
                         relevance_threshold=0):
    relevant = (relevance_vector > relevance_threshold).float()
    document_ranks = rankings.sort(dim=-1)[1].float()
    avg_rank = (document_ranks * (relevance_vector * relevant).unsqueeze(1)).sum(dim=-1)
    return avg_rank


def compute_dcg(ranking, relevance_vector, k=None):
    N = len(relevance_vector)
    if k is None:
        k = N
    ranking = ranking[:min((k, N))]
    len_ranking = float(len(ranking))
    sorted_relevances = -torch.sort(-relevance_vector)[0][:min((k, N))]
    len_sorted_relevances = float(len(sorted_relevances))

    dcgmax = torch.sum(sorted_relevances / torch.log2(
        torch.arange(len_sorted_relevances) + 2).to(relevance_vector.device))
    dcg = torch.sum(relevance_vector[ranking] / torch.log2(
        torch.arange(len_ranking) + 2).to(relevance_vector.device))
    if dcgmax == 0:
        dcgmax = 1.0
    return dcg / dcgmax, dcg


def compute_dcg_rankings(
        t_rankings, relevance_vector, binary_rel=False):
    """
    input t_rankings = [num_rankings X cand_set_size]
    returns dcg as a tensor of [num_rankings X 1] i.e. dcg for each ranking
    """
    if binary_rel:
        relevance_vector = (relevance_vector > 0).float()
    # t_rankings = t_rankings[:min((k, N)),:]
    len_rankings = float(t_rankings.shape[-1])
    sorted_relevances = torch.sort(
        relevance_vector,
        dim=-1,
        descending=True
    )[0]
    dcg = torch.zeros_like(t_rankings, dtype=torch.float)
    dcg.scatter_(-1, t_rankings,
                 1.0 / torch.log2(torch.arange(len_rankings, device=relevance_vector.device) + 2).expand_as(t_rankings))
    dcg *= relevance_vector.unsqueeze(-2)
    dcg = dcg.sum(dim=-1)
    dcgmax = torch.sum(sorted_relevances / torch.log2(torch.arange(len_rankings, device=relevance_vector.device) + 2),
                       dim=-1)   # JK  changed the * to /
    #dcgmax = torch.sum(sorted_relevances * torch.log2(torch.arange(len_rankings, device=relevance_vector.device) + 2),
    #                   dim=-1)
    nonzero = (dcgmax != 0.0)
    ndcg = dcg.clone()
    ndcg[nonzero] /= dcgmax[nonzero].unsqueeze(-1)
    return ndcg, dcg


# JK  need this to normalize the DCG
def compute_dcg_max(relevance_vector, binary_rel=False):
    """
    input t_rankings = [num_rankings X cand_set_size]
    returns dcg as a tensor of [num_rankings X 1] i.e. dcg for each ranking
    """
    if binary_rel:
        relevance_vector = (relevance_vector > 0).float()
    sorted_relevances = torch.sort(
        relevance_vector,
        dim=-1,
        descending=True
    )[0]
    rel_len = float(relevance_vector.shape[1])
    # JK beware of the / below, changed from *
    dcgmax = torch.sum(sorted_relevances / torch.log2(torch.arange(rel_len, device=relevance_vector.device) + 2),
                       dim=-1)
    #nonzero = (dcgmax != 0.0)
    #ndcg = dacg.clone()
    #ndcg[nonzero] /= dcgmax[nonzero].unsqueeze(-1)   #use these commands after return
    return dcgmax

def get_relative_gain(relevance):
    return (2.0 ** relevance - 1) / 16


def compute_err(ranking, relevance_vector):
    """
    Defined in Chapelle 11a (Section 5.1.1)
    """
    err = 0.0
    for i, doc in enumerate(ranking):
        not_found_probability = 1.0
        for j in range(i):
            not_found_probability *= 1 - get_relative_gain(
                relevance_vector[ranking[j]])
        err += get_relative_gain(
            relevance_vector[doc]) * not_found_probability / (1 + i)
    return err


def pairwise_mse(exposures, relevances):
    mse = 0.0
    e_by_r = exposures / relevances
    N = len(relevances)
    for i in range(N):
        for j in range(i, N):
            mse += (e_by_r[i] - e_by_r[j]) ** 2
    return mse / (N * N)


def scale_invariant_mse(exposures, relevances):
    """
    https://arxiv.org/pdf/1406.2283v1.pdf Equation 1, 2, 3
    """
    # sqrt(Eq. 3)
    assert (np.all(
        np.isfinite(exposures) & np.isfinite(relevances) & (exposures > 0) &
        (relevances > 0)))
    log_diff = np.log(exposures) - np.log(relevances)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(
            np.sum(np.square(log_diff)) / num_pixels -
            np.square(np.sum(log_diff)) / np.square(num_pixels))


def asymmetric_disparity(exposures, relevances):
    disparities = []
    for i in range(len(exposures)):
        for j in range(len(exposures)):
            if relevances[i] >= relevances[j]:
                if relevances[j] > 0.0:
                    disparities.append(
                        max([
                            0, exposures[i] / relevances[i] -
                            exposures[j] / relevances[j]
                        ]))
                else:
                    disparities.append(0)
    if np.isnan(np.mean(disparities)):
        print("NAN occured at", exposures, relevances, disparities)
    return np.mean(disparities)


def evaluate_model(model,
                   validation_data,
                   group0_merit = None,   # JK
                   group1_merit = None,   # JK
                   num_sample_per_query=10,
                   deterministic=False,
                   fairness_evaluation=False,
                   position_bias_vector=None,
                   group_fairness_evaluation=False,
                   track_other_disparities=False,
                   args=None,
                   normalize=False,
                   noise=None,
                   en=None):
    if noise is None:
        noise = args.noise
    if en is None:
        en = args.en
    ndcg_list = []
    dcg_list = []
    rank_list = []
    weight_list = []
    fair_viol_all_list = []  # JK this will hold all the fairness violations from the dataset
    abs_fair_viol_all_list = []
    if (fairness_evaluation
            or group_fairness_evaluation) and position_bias_vector is None:
        position_bias_vector = 1. / torch.arange(
            1., 100.) ** args.position_bias_power
        if args.gpu:
            position_bias_vector = position_bias_vector.cuda()

    print("Entering model evaluation")

    # compare the training and test dataset forms before going on

    val_feats, val_rel = validation_data

    all_exposures = []
    all_rels = []

    validation_dataset = torch.utils.data.TensorDataset(val_feats, val_rel)
    dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size)
    if args.progressbar:
        dataloader = tqdm(dataloader)

    if group_fairness_evaluation:
        if track_other_disparities:
            disparity_types = ['disp0','disp1', 'disp2', 'disp3', 'ashudeep', 'ashudeep_mod']   # JK add disp0
        else:
            disparity_types = [args.disparity_type]
        if 'disp2' in disparity_types or 'ashudeep_mod' in disparity_types:
            group0_merit, group1_merit = get_group_merits(
                val_feats,
                val_rel,
                args.group_feat_id,
                args.group_feat_threshold,
                mean=False
            )
            sign = 1.0 if group0_merit >= group1_merit else -1.0
        else:
            #group0_merit, group1_merit = None, None   JK why is this here
            sign = None
        group_disparities = {
            disparity_type: [] for disparity_type in disparity_types
        }
    model.eval()
    with torch.no_grad():
        for data in dataloader:  # for each query
            feats, rel = data

            scores = model(feats).squeeze(-1)
            scores = scores * args.eval_temperature
            if deterministic:
                num_sample_per_query = 1
                rankings = torch.sort(
                    scores,
                    descending=True, dim=-1)[1].unsqueeze(1)
            else:
                rankings = multiple_sample_and_log_probability(
                    scores,
                    num_sample_per_query,
                    return_prob=False,
                    batch=True
                )

            ndcgs, dcgs = compute_dcg_rankings(rankings, rel)
            rank = compute_average_rank(rankings, rel)
            dcg_list += dcgs.mean(dim=-1).tolist()
            ndcg_list += ndcgs.mean(dim=-1).tolist()
            rank_list += rank.mean(dim=-1).tolist()
            weight_list += rel.sum(dim=-1).tolist()

            if group_fairness_evaluation:
                group_identities = get_group_identities(
                    feats,
                    args.group_feat_id,
                    args.group_feat_threshold
                )
                inds_g0 = group_identities == 0
                inds_g1 = group_identities == 1

                if args.unweighted_fairness:
                    rel = (rel > 0.0).float()

                for disparity_type in disparity_types:
                    if disparity_type == 'ashudeep':
                        disparity = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector).mean(dim=-1)
                    elif disparity_type == 'ashudeep_mod':
                        disparity = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector, sign=sign).mean(
                            dim=-1)
                    else:

                        disparity = GroupFairnessLoss.compute_multiple_group_disparity(
                            rankings,
                            rel,
                            group_identities,
                            group0_merit,
                            group1_merit,
                            position_bias_vector,
                            disparity_type=disparity_type,
                            noise=noise,
                            en=en
                        )#.mean(dim=-1)   # this is 1D tensor of expected violations per policy (len is batch size)

                        #print("disparity = ")
                        #print( disparity    )
                        #print("disparity.shape = ")
                        #print( disparity.shape    )

                        #disparity = np.abs(disparity).mean(dim=-1)
                        abs_disparity = np.abs(  disparity.mean(dim=-1)  )
                        disparity     =          disparity.mean(dim=-1)
                        # JK inserted absolute value here - before taking any averages!  10/27

                        #print("disparity = ")
                        #print( disparity    )
                        #print("disparity.shape = ")
                        #print( disparity.shape    )


                        fair_viol_all_list     += disparity.tolist()
                        abs_fair_viol_all_list += abs_disparity.tolist()
                        #print("fair_viol_all_list = ")
                        #print( fair_viol_all_list    )
                        #print("len(fair_viol_all_list) = ")
                        #print( len(fair_viol_all_list)    )
                        #input("Waiting")

                    for i in range(len(rankings)):
                        if inds_g0[i].any() and inds_g1[i].any():
                            group_disparities[disparity_type].append(disparity[i].item())
                        #else: #JK remove
                        #    print("single-group sample found")

    model.train()
    avg_ndcg = np.mean(ndcg_list)
    if normalize:
        avg_dcg  = np.sum(dcg_list) / np.sum(weight_list)
        avg_rank = np.sum(rank_list) / np.sum(weight_list)
    else:
        avg_dcg  = np.mean(dcg_list)
        avg_rank = np.mean(rank_list)


    #fair_viol_all_list = np.abs( np.array(  fair_viol_all_list  ) )
    #print("fair_viol_all_list = ")
    #print( fair_viol_all_list    )
    fair_viols_quantiles = {}
    fair_viols_quantiles['1.00'] = np.quantile(abs_fair_viol_all_list,1.00)
    fair_viols_quantiles['0.95'] = np.quantile(abs_fair_viol_all_list,0.95)
    fair_viols_quantiles['0.90'] = np.quantile(abs_fair_viol_all_list,0.90)
    fair_viols_quantiles['0.85'] = np.quantile(abs_fair_viol_all_list,0.85)
    fair_viols_quantiles['0.80'] = np.quantile(abs_fair_viol_all_list,0.80)
    fair_viols_quantiles['0.75'] = np.quantile(abs_fair_viol_all_list,0.75)
    fair_viols_quantiles['0.70'] = np.quantile(abs_fair_viol_all_list,0.70)
    fair_viols_quantiles['0.65'] = np.quantile(abs_fair_viol_all_list,0.65)
    fair_viols_quantiles['0.60'] = np.quantile(abs_fair_viol_all_list,0.60)
    fair_viols_quantiles['0.55'] = np.quantile(abs_fair_viol_all_list,0.55)
    fair_viols_quantiles['0.50'] = np.quantile(abs_fair_viol_all_list,0.50)

    """
    print("fair_viol_all_list = ")
    print( fair_viol_all_list    )
    print("quantiles: ")
    print( fair_viols_quantiles['1.00'] )
    print( fair_viols_quantiles['0.95'] )
    print( fair_viols_quantiles['0.90'] )
    print( fair_viols_quantiles['0.85'] )
    print( fair_viols_quantiles['0.80'] )
    print( fair_viols_quantiles['0.75'] )
    print( fair_viols_quantiles['0.70'] )
    print( fair_viols_quantiles['0.65'] )
    print( fair_viols_quantiles['0.60'] )
    print( fair_viols_quantiles['0.55'] )
    print( fair_viols_quantiles['0.50'] )
    print("np.mean(fair_viol_all_list) = ")
    print( np.mean(fair_viol_all_list)    )
    """


    results = {
        "ndcg": avg_ndcg,
        "dcg": avg_dcg,
        "avg_rank": avg_rank,
        "fair_viol_all_list": abs_fair_viol_all_list,
        "fair_viols_quantiles": fair_viols_quantiles
    }
    if group_fairness_evaluation:
        # convert lists in dictionary to np arrays
        for disparity_type in group_disparities:
            group_disparities[disparity_type] = np.mean(
                group_disparities[disparity_type])

        other_disparities = {}
        for k, v in group_disparities.items():
            if k == 'ashudeep' or k == 'ashudeep_mod':
                disparity = v
                asym_disparity = v
            else:
                if args.indicator_type == "square":
                    disparity = v
                    asym_disparity = v ** 2
                elif args.indicator_type == "sign":
                    disparity = v
                    asym_disparity = abs(v)
                elif args.indicator_type == "none":
                    disparity = v
                    asym_disparity = v
                else:
                    raise NotImplementedError
            if k == args.disparity_type:
                avg_group_exposure_disparity = disparity
                avg_group_asym_disparity = asym_disparity
            other_disparities[k] = [asym_disparity, disparity]


        #print(   np.mean( abs_fair_viol_all_list )  )
        #print(   np.mean(     fair_viol_all_list )  )
        #print(   avg_group_exposure_disparity       )
        #input("waiting")


        results.update({
            "avg_abs_group_disparity": np.mean(abs_fair_viol_all_list),    # JK
            "avg_group_disparity": avg_group_exposure_disparity,
            "avg_group_asym_disparity": avg_group_asym_disparity
        })
        if track_other_disparities:
            results.update({"other_disparities": other_disparities})


        #print("avg_group_exposure_disparity = ")
        #print( avg_group_exposure_disparity    )
        #input("Waiting")


    return results


# JK
# Test-time evaluation for soft_policy_training
def evaluate_soft_model(model,
                        validation_data,
                        group0_merit = None,   # JK
                        group1_merit = None,   # JK
                        num_sample_per_query=10,
                        deterministic=False,
                        fairness_evaluation=False,
                        position_bias_vector=None,
                        group_fairness_evaluation=False,
                        track_other_disparities=False,
                        args=None,
                        normalize=False,
                        noise=None,
                        en=None):
    if noise is None:
        noise = args.noise
    if en is None:
        en = args.en
    ndcg_list = []
    dcg_list = []
    rank_list = []
    weight_list = []
    DSM_ndcg_list = []   #JK
    DSM_dcg_list = []
    mean_fair_viol_list = []
    max_fair_viol_list = []
    fair_viol_all_list = []   # JK this holds all the fairness violations encountered in the routine
    if (fairness_evaluation
            or group_fairness_evaluation) and position_bias_vector is None:
        position_bias_vector = 1. / torch.arange(
            1., 100.) ** args.position_bias_power
        if args.gpu:
            position_bias_vector = position_bias_vector.cuda()

    val_feats, val_rel = validation_data

    # JK limit the validation set for this
    #max_sample_eval = 1280#000
    #val_feats = val_feats[:max_sample_eval]
    #val_rel   = val_rel[:max_sample_eval]

    all_exposures = []
    all_rels = []

    relu = nn.ReLU()

    validation_dataset = torch.utils.data.TensorDataset(val_feats, val_rel)
    dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size)
    if args.progressbar:
        dataloader = tqdm(dataloader)

    if group_fairness_evaluation:
        if track_other_disparities:
            disparity_types = ['disp0','disp1', 'disp2', 'disp3', 'ashudeep', 'ashudeep_mod']   # JK add disp0
        else:
            disparity_types = [args.disparity_type]
        if 'disp2' in disparity_types or 'ashudeep_mod' in disparity_types:
            group0_merit, group1_merit = get_group_merits(
                val_feats,
                val_rel,
                args.group_feat_id,
                args.group_feat_threshold,
                mean=False
            )
            sign = 1.0 if group0_merit >= group1_merit else -1.0
        else:
            #group0_merit, group1_merit = None, None    # JK why is this here
            sign = None
        group_disparities = {
            disparity_type: [] for disparity_type in disparity_types
        }
    model.eval()
    with torch.no_grad():


        # Initialize solvers
        ##############
        ####### added in assuming we'll use SPO from now on
        solver_dict = {}
        for i in range(1,args.list_len):

            if args.allow_unfairness:
                # Delta Fairness
                # Google solver only
                gids = torch.zeros(args.list_len).long()
                gids[:i] = 1
                s,x = ort_setup_Neq(args.list_len, gids, args.disparity_type, group0_merit, group1_merit, args.fairness_gap)
                key = int(gids.sum().item())      # JK check this key - not used?
                solver_dict[i] = ort_policyLP(s,x)
            else:
                # Perfect Fairness
                gids = torch.zeros(args.list_len).long()
                gids[:i] = 1
                s,x = ort_setup(args.list_len, gids, args.disparity_type, group0_merit, group1_merit)
                key = int(gids.sum().item())      # JK check this key - not used?
                solver_dict[i] = ort_policyLP(s,x)

        for i,data in enumerate(dataloader):

            feats, rel = data
            batsize = feats.shape[0]
            group_identities = get_group_identities(feats, args.group_feat_id, args.group_feat_threshold)
            if group_identities.bool().all(1).any().item() or (1-group_identities).bool().all(1).any().item():
                continue
                # skip the iteration if only one group appears

            if args.embed_groups:
                scores, group_embed = model(feats, group_identities)
                scores= scores.squeeze(-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(batsize,-1,1), group_embed.unsqueeze(0).view(batsize,-1,1).permute(0,2,1)  ).reshape(batsize,-1)
            # Concatenate the document scores with group ID and predict N**2 independent QP coefficients using a MLP
            elif args.embed_quadscore:
                score_cross = model(feats, group_identities).squeeze(-1)
                #score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
            else:
                scores = model(feats).squeeze(-1)
                test_dscts = ( 1.0 / torch.log2(torch.arange(args.list_len).float() + 2) ).repeat(batsize,1,1)
                #score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), test_dscts.view(batsize,1,-1)  ).reshape(batsize,-1)

            test_dscts = ( 1.0 / torch.log2(torch.arange(args.list_len).float() + 2) ).repeat(batsize,1,1)
            true_costs = torch.bmm( rel.view(batsize,-1,1), test_dscts.view(batsize,1,-1)).view(batsize,1,-1)

            grad = []
            p_mat = []
            regrets = []
            with torch.no_grad():
                dcg_max = compute_dcg_max(rel)  # redundant, defined again below

                if not args.multi_groups:
                    for i in range(batsize):

                        spo_group_ids = group_identities[i].detach().numpy()
                        sorting_ind = np.argsort(spo_group_ids)[::-1]
                        reverse_ind = np.argsort(sorting_ind)

                        solver = solver_dict[ int(spo_group_ids.sum().item()) ]

                        V_true  = true_costs[i].squeeze().detach().double().numpy() #compute 'true' cost coefficients here
                        V_true1 = true_costs[i].squeeze().detach().double().numpy()                    #delete
                        V_true = (V_true.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()

                        sol_true = solver.solve(V_true)
                        sol_true = sol_true.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                        V_pred  = score_cross[i].squeeze().detach().double().numpy() #compute 'pred' cost coefficients here

                        V_pred = (V_pred.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                        sol_pred = solver.solve(V_pred)
                        sol_pred = sol_pred.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                        p_mat.append(torch.Tensor(sol_pred).view(args.list_len,args.list_len))

                        V_spo   = (2*V_pred - V_true)
                        V_spo   = (V_spo.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                        sol_spo  = solver.solve(V_spo)
                        sol_spo  = sol_spo.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                        #reg = torch.dot(V_true1,(sol_true - sol_pred))
                        reg = torch.Tensor(  [np.dot(V_true1,(sol_true - sol_pred))]  )
                        regrets.append(reg)
                        use_reg = True
                        if use_reg:
                            grad.append( torch.Tensor(sol_spo - sol_true)  /  dcg_max[i]  )
                        else:
                            grad.append( torch.Tensor(sol_spo - sol_true)  )

                    p_mat = torch.stack(p_mat)
                #######
                ################
                ################################
                else:
                    for i in range(batsize):
                        spo_group_ids = group_identities[i].detach().numpy()
                        sorting_ind = np.argsort(spo_group_ids)[::-1]
                        reverse_ind = np.argsort(sorting_ind)

                        input_group_ids = np.sort(spo_group_ids)[::-1]
                        #solver = solver_dict[ int(spo_group_ids.sum().item()) ]
                        if not str(input_group_ids) in solver_dict:
                            s,x = ort_setup_multi_Neq(args.list_len, torch.Tensor( input_group_ids.tolist() ), args.disparity_type, group0_merit, group1_merit, args.fairness_gap)
                            solver_dict[ str(input_group_ids) ] = ort_policyLP(s,x)

                        # infeasible now for the non-multigroups case
                        # fix before testing with multigroups

                        solver = solver_dict[ str(input_group_ids) ]

                        V_true  = true_costs[i].squeeze().detach().double().numpy() #compute 'true' cost coefficients here
                        V_true1 = true_costs[i].squeeze().detach().double().numpy()                    #delete
                        V_true  = (V_true.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()


                        sol_true = solver.solve(V_true)
                        sol_true = sol_true.reshape((args.list_len,args.list_len))[reverse_ind].flatten()


                        V_pred   = score_cross[i].squeeze().detach().double().numpy() #compute 'pred' cost coefficients here
                        V_pred   = (V_pred.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                        sol_pred = solver.solve(V_pred)
                        sol_pred = sol_pred.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                        p_mat.append(torch.Tensor(sol_pred).view(args.list_len,args.list_len))

                        V_spo    = (2*V_pred - V_true)
                        V_spo    = (V_spo.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                        sol_spo  = solver.solve(V_spo)
                        sol_spo  = sol_spo.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                        #reg = torch.dot(V_true1,(sol_true - sol_pred))
                        reg = torch.Tensor(  [np.dot(V_true1,(sol_true - sol_pred))]  )
                        regrets.append(reg)
                        use_reg = False
                        if use_reg:
                            grad.append( torch.Tensor(sol_spo - sol_true)  /  dcg_max[i]  )
                        else:
                            grad.append( torch.Tensor(sol_spo - sol_true)  )
                    p_mat = torch.stack(p_mat)


            if deterministic:
                num_sample_per_query = 1
                rankings = torch.sort(
                    scores,
                    descending=True, dim=-1)[1].unsqueeze(1)
            else:
                # JK replace old sampling method with this one
                with torch.no_grad():
                    P = p_mat.cpu().detach().numpy()
                    #max_instances_sample = 200 #min(200, P.shape[0]) # Take a max of 200 from each batch
                    #P = P[np.random.choice(P.shape[0],max_instances_sample,replace = True)]
                    R = []
                    for it, policy in enumerate(P):
                        decomp = birkhoff_von_neumann_decomposition(policy)
                        convex_coeffs, permutations = zip(*decomp)
                        permutations = np.array(permutations)
                        rolls = torch.multinomial(torch.Tensor(convex_coeffs),num_sample_per_query,replacement=True).numpy()
                        #rolls = np.random.multinomial(sample_size, np.array(convex_coeffs))  # sample the permutations based on convex_coeffs
                        p_sample = permutations[rolls]       # access the permutations
                        r_sample = p_sample.argmax(2)        # convert to rankings
                        r_sample = torch.tensor( r_sample )  # convert to same datatype as FULTR implementation
                        R.append(r_sample)
                        #print("Finished policy sampling iteration {}".format(it))
                    rankings = torch.stack(R)
                    if args.gpu:
                        rankings = rankings.cuda()   # JK testing

                ############
                # Soft evaluation metrics

                with torch.no_grad():

                    dcg_max = compute_dcg_max(rel)
                    test_dscts = ( 1.0 / torch.log2(torch.arange(args.list_len).float() + 2) ).repeat(batsize,1,1)
                    if args.gpu:
                        test_dscts = test_dscts.cuda()
                    #v_unsq = v.unsqueeze(1)
                    #f_unsq = f.unsqueeze(1).permute(0,2,1)
                    #vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
                    loss_a = torch.bmm( p_mat, test_dscts.view(batsize,-1,1) )
                    loss_b = torch.bmm( rel.view(batsize,1,-1), loss_a ).squeeze()
                    loss_norm = loss_b.squeeze() / dcg_max
                    loss = loss_norm.mean()


                    #DSM_ndcg_list.append(loss)     # 11/14 why is this here
                    #DSM_dcg_list.append(loss_b.squeeze().mean())

                    # Find average violation
                    #fair_viol_mean_batch = 0
                    #for kk in range(len(p_mat)):
                    #    fair_viol_mean_batch += test_fairness( p_mat[kk], group_identities[kk], position_bias_vector )
                    #fair_viol_mean_batch /= len(p_mat)

                    fair_viol_batch_list = []
                    for kk in range(len(p_mat)):
                        fair_viol_batch_list.append(    test_fairness( p_mat[kk], group_identities[kk], position_bias_vector, args.disparity_type, group0_merit, group1_merit )   )
                        # JK absolute value added 10/27


                    fair_viol_mean_batch = np.mean(fair_viol_batch_list).item()
                    fair_viol_max_batch  = np.max(fair_viol_batch_list).item()

                    fair_viol_all_list += fair_viol_batch_list

                    DSM_ndcg_list.append( loss.item() )
                    DSM_dcg_list.append( loss_b.squeeze().mean().item() )
                    mean_fair_viol_list.append( fair_viol_mean_batch )
                    max_fair_viol_list.append(fair_viol_max_batch)





                    #print("viol = ")
                    #print( fair_viol_mean_batch )
                    #input()

                # END Soft evaluation metrics
                ############



            ndcgs, dcgs = compute_dcg_rankings(rankings, rel)

            rank = compute_average_rank(rankings, rel)
            dcg_list  += dcgs.mean(dim=-1).tolist()
            ndcg_list += ndcgs.mean(dim=-1).tolist()
            rank_list += rank.mean(dim=-1).tolist()
            weight_list += rel.sum(dim=-1).tolist()

            if group_fairness_evaluation:
                group_identities = get_group_identities(
                    feats,
                    args.group_feat_id,
                    args.group_feat_threshold
                )
                inds_g0 = group_identities == 0
                inds_g1 = group_identities == 1

                if args.unweighted_fairness:
                    rel = (rel > 0.0).float()

                for disparity_type in disparity_types:
                    if disparity_type == 'ashudeep':
                        disparity = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector).mean(dim=-1)
                    elif disparity_type == 'ashudeep_mod':
                        disparity = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector, sign=sign).mean(
                            dim=-1)
                    else:
                        disparity = GroupFairnessLoss.compute_multiple_group_disparity(
                            rankings,
                            rel,
                            group_identities,
                            group0_merit,
                            group1_merit,
                            position_bias_vector,
                            disparity_type=disparity_type,
                            noise=noise,
                            en=en
                        )#.mean(dim=-1)
                        # JK absolute value on expected policy violation

                        disparity = np.abs(  disparity.mean(dim=-1)  )
                    for i in range(len(rankings)):
                        if inds_g0[i].any() and inds_g1[i].any():
                            group_disparities[disparity_type].append(disparity[i].item())

    model.train()
    avg_ndcg = np.mean(ndcg_list)
    if normalize:
        avg_dcg = np.sum(dcg_list) / np.sum(weight_list)
        avg_rank = np.sum(rank_list) / np.sum(weight_list)
    else:
        avg_dcg = np.mean(dcg_list)
        avg_rank = np.mean(rank_list)

    DSM_ndcg = np.mean(DSM_ndcg_list)                      # JK
    DSM_dcg = np.mean(DSM_dcg_list)
    DSM_mean_abs_viol = np.mean(  np.abs(mean_fair_viol_list)  )
    DSM_mean_viol = np.mean(  mean_fair_viol_list  )
    DSM_max_viol  = np.max(    max_fair_viol_list  )

    # Fairness Violation Quantiles
    #fair_viol_all_list = np.abs( np.array(  fair_viol_all_list  ) )
    #print("fair_viol_all_list = ")
    #print( fair_viol_all_list    )
    fair_viols_quantiles = {}
    fair_viols_quantiles['1.00'] = np.quantile( np.abs(fair_viol_all_list) ,1.00)
    fair_viols_quantiles['0.95'] = np.quantile( np.abs(fair_viol_all_list) ,0.95)
    fair_viols_quantiles['0.90'] = np.quantile( np.abs(fair_viol_all_list) ,0.90)
    fair_viols_quantiles['0.85'] = np.quantile( np.abs(fair_viol_all_list) ,0.85)
    fair_viols_quantiles['0.80'] = np.quantile( np.abs(fair_viol_all_list) ,0.80)
    fair_viols_quantiles['0.75'] = np.quantile( np.abs(fair_viol_all_list) ,0.75)
    fair_viols_quantiles['0.70'] = np.quantile( np.abs(fair_viol_all_list) ,0.70)
    fair_viols_quantiles['0.65'] = np.quantile( np.abs(fair_viol_all_list) ,0.65)
    fair_viols_quantiles['0.60'] = np.quantile( np.abs(fair_viol_all_list) ,0.60)
    fair_viols_quantiles['0.55'] = np.quantile( np.abs(fair_viol_all_list) ,0.55)
    fair_viols_quantiles['0.50'] = np.quantile( np.abs(fair_viol_all_list) ,0.50)


    results = {
        "DSM_ndcg": DSM_ndcg,
        "DSM_dcg": DSM_dcg,
        "DSM_mean_abs_viol": DSM_mean_abs_viol,
        "DSM_mean_viol": DSM_mean_viol,
        "DSM_max_viol": DSM_max_viol,
        "ndcg": avg_ndcg,
        "dcg": avg_dcg,
        "avg_rank": avg_rank,
        "fair_viols_quantiles":fair_viols_quantiles
    }
    if group_fairness_evaluation:
        # convert lists in dictionary to np arrays
        for disparity_type in group_disparities:
            group_disparities[disparity_type] = np.mean(
                group_disparities[disparity_type])

        other_disparities = {}
        for k, v in group_disparities.items():
            if k == 'ashudeep' or k == 'ashudeep_mod':
                disparity = v
                asym_disparity = v
            else:
                if args.indicator_type == "square":
                    disparity = v
                    asym_disparity = v ** 2
                elif args.indicator_type == "sign":
                    disparity = v
                    asym_disparity = abs(v)
                elif args.indicator_type == "none":
                    disparity = v
                    asym_disparity = v
                else:
                    raise NotImplementedError
            if k == args.disparity_type:
                avg_group_exposure_disparity = disparity
                avg_group_asym_disparity = asym_disparity
            other_disparities[k] = [asym_disparity, disparity]

        results.update({
            "avg_group_disparity": avg_group_exposure_disparity,
            "avg_group_asym_disparity": avg_group_asym_disparity
        })
        if track_other_disparities:
            results.update({"other_disparities": other_disparities})

    return results








# JK
# Test-time evaluation for soft_policy_training
def evaluate_soft_model_multi(model,
                   validation_data,
                   group0_merit = None,   # JK
                   group1_merit = None,   # JK
                   num_sample_per_query=10,
                   deterministic=False,
                   fairness_evaluation=False,
                   position_bias_vector=None,
                   group_fairness_evaluation=False,
                   track_other_disparities=False,
                   args=None,
                   normalize=False,
                   noise=None,
                   en=None):
    if noise is None:
        noise = args.noise
    if en is None:
        en = args.en
    ndcg_list = []
    dcg_list = []
    rank_list = []
    weight_list = []
    DSM_ndcg_list = []   #JK
    DSM_dcg_list = []
    mean_fair_viol_list = []
    max_fair_viol_list = []
    fair_viol_all_list = []   # JK this holds all the fairness violations encountered in the routine
    if (fairness_evaluation
            or group_fairness_evaluation) and position_bias_vector is None:
        position_bias_vector = 1. / torch.arange(
            1., 100.) ** args.position_bias_power
        if args.gpu:
            position_bias_vector = position_bias_vector.cuda()

    val_feats, val_rel = validation_data

    # JK limit the validation set for this
    #max_sample_eval = 1280#000
    #val_feats = val_feats[:max_sample_eval]
    #val_rel   = val_rel[:max_sample_eval]

    all_exposures = []
    all_rels = []

    relu = nn.ReLU()

    validation_dataset = torch.utils.data.TensorDataset(val_feats, val_rel)
    dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size)
    if args.progressbar:
        dataloader = tqdm(dataloader)

    if group_fairness_evaluation:
        if track_other_disparities:
            disparity_types = ['disp0','disp1', 'disp2', 'disp3', 'ashudeep', 'ashudeep_mod']   # JK add disp0
        else:
            disparity_types = [args.disparity_type]
        if 'disp2' in disparity_types or 'ashudeep_mod' in disparity_types:
            group0_merit, group1_merit = get_group_merits(
                val_feats,
                val_rel,
                args.group_feat_id,
                args.group_feat_threshold,
                mean=False
            )
            sign = 1.0 if group0_merit >= group1_merit else -1.0
        else:
            #group0_merit, group1_merit = None, None    # JK why is this here
            sign = None
        group_disparities = {
            disparity_type: [] for disparity_type in disparity_types
        }
    model.eval()
    with torch.no_grad():


        # Initialize solvers
        ##############
        ####### added in assuming we'll use SPO from now on
        solver_dict = {}
        for i in range(1,args.list_len):

            if args.allow_unfairness:
                # Delta Fairness
                # Google solver only
                gids = torch.zeros(args.list_len).long()
                gids[:i] = 1
                s,x = ort_setup_Neq(args.list_len, gids, args.disparity_type, group0_merit, group1_merit, args.fairness_gap)
                key = int(gids.sum().item())      # JK check this key - not used?
                solver_dict[i] = ort_policyLP(s,x)
            else:
                # Perfect Fairness
                gids = torch.zeros(args.list_len).long()
                gids[:i] = 1
                s,x = ort_setup(args.list_len, gids, args.disparity_type, group0_merit, group1_merit)
                key = int(gids.sum().item())      # JK check this key - not used?
                solver_dict[i] = ort_policyLP(s,x)

        for i,data in enumerate(dataloader):

            feats, rel = data
            batsize = feats.shape[0]
            group_identities = get_group_identities(feats, args.group_feat_id, args.group_feat_threshold)
            if group_identities.bool().all(1).any().item() or (1-group_identities).bool().all(1).any().item():
                continue
                # skip the iteration if only one group appears

            if args.embed_groups:
                scores, group_embed = model(feats, group_identities)
                scores= scores.squeeze(-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(batsize,-1,1), group_embed.unsqueeze(0).view(batsize,-1,1).permute(0,2,1)  ).reshape(batsize,-1)
            # Concatenate the document scores with group ID and predict N**2 independent QP coefficients using a MLP
            elif args.embed_quadscore:
                score_cross = model(feats, group_identities).squeeze(-1)
                #score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
            else:
                scores = model(feats).squeeze(-1)
                test_dscts = ( 1.0 / torch.log2(torch.arange(args.list_len).float() + 2) ).repeat(batsize,1,1)
                #score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), test_dscts.view(batsize,1,-1)  ).reshape(batsize,-1)

            test_dscts = ( 1.0 / torch.log2(torch.arange(args.list_len).float() + 2) ).repeat(batsize,1,1)
            true_costs = torch.bmm( rel.view(batsize,-1,1), test_dscts.view(batsize,1,-1)).view(batsize,1,-1)

            grad = []
            p_mat = []
            regrets = []
            with torch.no_grad():
                dcg_max = compute_dcg_max(rel)  # redundant, defined again below

                if not args.multi_groups:
                    for i in range(batsize):

                        spo_group_ids = group_identities[i].detach().numpy()
                        sorting_ind = np.argsort(spo_group_ids)[::-1]
                        reverse_ind = np.argsort(sorting_ind)

                        solver = solver_dict[ int(spo_group_ids.sum().item()) ]

                        V_true  = true_costs[i].squeeze().detach().double().numpy() #compute 'true' cost coefficients here
                        V_true1 = true_costs[i].squeeze().detach().double().numpy()                    #delete
                        V_true = (V_true.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()

                        sol_true = solver.solve(V_true)
                        sol_true = sol_true.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                        V_pred  = score_cross[i].squeeze().detach().double().numpy() #compute 'pred' cost coefficients here

                        V_pred = (V_pred.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                        sol_pred = solver.solve(V_pred)
                        sol_pred = sol_pred.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                        p_mat.append(torch.Tensor(sol_pred).view(args.list_len,args.list_len))

                        V_spo   = (2*V_pred - V_true)
                        V_spo   = (V_spo.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                        sol_spo  = solver.solve(V_spo)
                        sol_spo  = sol_spo.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                        #reg = torch.dot(V_true1,(sol_true - sol_pred))
                        reg = torch.Tensor(  [np.dot(V_true1,(sol_true - sol_pred))]  )
                        regrets.append(reg)
                        use_reg = True
                        if use_reg:
                            grad.append( torch.Tensor(sol_spo - sol_true)  /  dcg_max[i]  )
                        else:
                            grad.append( torch.Tensor(sol_spo - sol_true)  )

                    p_mat = torch.stack(p_mat)
                #######
                ################
                ################################
                else:
                    for i in range(batsize):
                        spo_group_ids = group_identities[i].detach().numpy()
                        sorting_ind = np.argsort(spo_group_ids)[::-1]
                        reverse_ind = np.argsort(sorting_ind)

                        input_group_ids = np.sort(spo_group_ids)[::-1]
                        #solver = solver_dict[ int(spo_group_ids.sum().item()) ]
                        if not str(input_group_ids) in solver_dict:
                            s,x = ort_setup_multi_Neq(args.list_len, torch.Tensor( input_group_ids.tolist() ), args.disparity_type, group0_merit, group1_merit, args.fairness_gap)
                            solver_dict[ str(input_group_ids) ] = ort_policyLP(s,x)

                        # infeasible now for the non-multigroups case
                        # fix before testing with multigroups

                        solver = solver_dict[ str(input_group_ids) ]

                        V_true  = true_costs[i].squeeze().detach().double().numpy() #compute 'true' cost coefficients here
                        V_true1 = true_costs[i].squeeze().detach().double().numpy()                    #delete
                        V_true  = (V_true.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()


                        sol_true = solver.solve(V_true)
                        sol_true = sol_true.reshape((args.list_len,args.list_len))[reverse_ind].flatten()


                        V_pred   = score_cross[i].squeeze().detach().double().numpy() #compute 'pred' cost coefficients here
                        V_pred   = (V_pred.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                        sol_pred = solver.solve(V_pred)
                        sol_pred = sol_pred.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                        p_mat.append(torch.Tensor(sol_pred).view(args.list_len,args.list_len))

                        V_spo    = (2*V_pred - V_true)
                        V_spo    = (V_spo.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                        sol_spo  = solver.solve(V_spo)
                        sol_spo  = sol_spo.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                        #reg = torch.dot(V_true1,(sol_true - sol_pred))
                        reg = torch.Tensor(  [np.dot(V_true1,(sol_true - sol_pred))]  )
                        regrets.append(reg)
                        use_reg = False
                        if use_reg:
                            grad.append( torch.Tensor(sol_spo - sol_true)  /  dcg_max[i]  )
                        else:
                            grad.append( torch.Tensor(sol_spo - sol_true)  )
                    p_mat = torch.stack(p_mat)


            if deterministic:
                num_sample_per_query = 1
                rankings = torch.sort(
                    scores,
                    descending=True, dim=-1)[1].unsqueeze(1)
            else:
                # JK replace old sampling method with this one
                with torch.no_grad():
                    P = p_mat.cpu().detach().numpy()
                    #max_instances_sample = 200 #min(200, P.shape[0]) # Take a max of 200 from each batch
                    #P = P[np.random.choice(P.shape[0],max_instances_sample,replace = True)]
                    R = []
                    for it, policy in enumerate(P):
                        decomp = birkhoff_von_neumann_decomposition(policy)
                        convex_coeffs, permutations = zip(*decomp)
                        permutations = np.array(permutations)
                        rolls = torch.multinomial(torch.Tensor(convex_coeffs),num_sample_per_query,replacement=True).numpy()
                        #rolls = np.random.multinomial(sample_size, np.array(convex_coeffs))  # sample the permutations based on convex_coeffs
                        p_sample = permutations[rolls]       # access the permutations
                        r_sample = p_sample.argmax(2)        # convert to rankings
                        r_sample = torch.tensor( r_sample )  # convert to same datatype as FULTR implementation
                        R.append(r_sample)
                        #print("Finished policy sampling iteration {}".format(it))
                    rankings = torch.stack(R)
                    if args.gpu:
                        rankings = rankings.cuda()   # JK testing

                ############
                # Soft evaluation metrics

                with torch.no_grad():

                    dcg_max = compute_dcg_max(rel)
                    test_dscts = ( 1.0 / torch.log2(torch.arange(args.list_len).float() + 2) ).repeat(batsize,1,1)
                    if args.gpu:
                        test_dscts = test_dscts.cuda()
                    #v_unsq = v.unsqueeze(1)
                    #f_unsq = f.unsqueeze(1).permute(0,2,1)
                    #vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
                    loss_a = torch.bmm( p_mat, test_dscts.view(batsize,-1,1) )
                    loss_b = torch.bmm( rel.view(batsize,1,-1), loss_a ).squeeze()
                    loss_norm = loss_b.squeeze() / dcg_max
                    loss = loss_norm.mean()


                    #DSM_ndcg_list.append(loss)     # 11/14 why is this here
                    #DSM_dcg_list.append(loss_b.squeeze().mean())

                    # Find average violation
                    #fair_viol_mean_batch = 0
                    #for kk in range(len(p_mat)):
                    #    fair_viol_mean_batch += test_fairness( p_mat[kk], group_identities[kk], position_bias_vector )
                    #fair_viol_mean_batch /= len(p_mat)

                    #fair_viol_batch_list = []
                    #for kk in range(len(p_mat)):
                    #    fair_viol_batch_list.append(    test_fairness( p_mat[kk], group_identities[kk], position_bias_vector, args.disparity_type, group0_merit, group1_merit )   )
                        # JK absolute value added 10/27


                    #fair_viol_mean_batch = np.mean(fair_viol_batch_list).item()
                    #fair_viol_max_batch  = np.max(fair_viol_batch_list).item()

                    #fair_viol_all_list += fair_viol_batch_list

                    DSM_ndcg_list.append( loss.item() )
                    DSM_dcg_list.append( loss_b.squeeze().mean().item() )
                    #mean_fair_viol_list.append( fair_viol_mean_batch )
                    #max_fair_viol_list.append(fair_viol_max_batch)





                    #print("viol = ")
                    #print( fair_viol_mean_batch )
                    #input()

                # END Soft evaluation metrics
                ############



            ndcgs, dcgs = compute_dcg_rankings(rankings, rel)

            rank = compute_average_rank(rankings, rel)
            dcg_list  += dcgs.mean(dim=-1).tolist()
            ndcg_list += ndcgs.mean(dim=-1).tolist()
            rank_list += rank.mean(dim=-1).tolist()
            weight_list += rel.sum(dim=-1).tolist()

            """
            if group_fairness_evaluation:
                group_identities = get_group_identities(
                    feats,
                    args.group_feat_id,
                    args.group_feat_threshold
                )
                inds_g0 = group_identities == 0
                inds_g1 = group_identities == 1

                if args.unweighted_fairness:
                    rel = (rel > 0.0).float()

                for disparity_type in disparity_types:
                    if disparity_type == 'ashudeep':
                        disparity = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector).mean(dim=-1)
                    elif disparity_type == 'ashudeep_mod':
                        disparity = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector, sign=sign).mean(
                            dim=-1)
                    else:
                        disparity = GroupFairnessLoss.compute_multiple_group_disparity(
                            rankings,
                            rel,
                            group_identities,
                            group0_merit,
                            group1_merit,
                            position_bias_vector,
                            disparity_type=disparity_type,
                            noise=noise,
                            en=en
                        )#.mean(dim=-1)
                        # JK absolute value on expected policy violation

                        disparity = np.abs(  disparity.mean(dim=-1)  )
                    for i in range(len(rankings)):
                        if inds_g0[i].any() and inds_g1[i].any():
                            group_disparities[disparity_type].append(disparity[i].item())
            """

    model.train()
    avg_ndcg = np.mean(ndcg_list)
    if normalize:
        avg_dcg = np.sum(dcg_list) / np.sum(weight_list)
        avg_rank = np.sum(rank_list) / np.sum(weight_list)
    else:
        avg_dcg = np.mean(dcg_list)
        avg_rank = np.mean(rank_list)

    DSM_ndcg = np.mean(DSM_ndcg_list)                      # JK
    DSM_dcg = np.mean(DSM_dcg_list)
    #DSM_mean_abs_viol = np.mean(  np.abs(mean_fair_viol_list)  )
    #DSM_mean_viol = np.mean(  mean_fair_viol_list  )
    #DSM_max_viol  = np.max(    max_fair_viol_list  )

    # Fairness Violation Quantiles
    #fair_viol_all_list = np.abs( np.array(  fair_viol_all_list  ) )
    #print("fair_viol_all_list = ")
    #print( fair_viol_all_list    )
    #fair_viols_quantiles = {}
    #fair_viols_quantiles['1.00'] = np.quantile( np.abs(fair_viol_all_list) ,1.00)
    #fair_viols_quantiles['0.95'] = np.quantile( np.abs(fair_viol_all_list) ,0.95)
    #fair_viols_quantiles['0.90'] = np.quantile( np.abs(fair_viol_all_list) ,0.90)
    #fair_viols_quantiles['0.85'] = np.quantile( np.abs(fair_viol_all_list) ,0.85)
    #fair_viols_quantiles['0.80'] = np.quantile( np.abs(fair_viol_all_list) ,0.80)
    #fair_viols_quantiles['0.75'] = np.quantile( np.abs(fair_viol_all_list) ,0.75)
    #fair_viols_quantiles['0.70'] = np.quantile( np.abs(fair_viol_all_list) ,0.70)
    #fair_viols_quantiles['0.65'] = np.quantile( np.abs(fair_viol_all_list) ,0.65)
    #fair_viols_quantiles['0.60'] = np.quantile( np.abs(fair_viol_all_list) ,0.60)
    #fair_viols_quantiles['0.55'] = np.quantile( np.abs(fair_viol_all_list) ,0.55)
    #fair_viols_quantiles['0.50'] = np.quantile( np.abs(fair_viol_all_list) ,0.50)


    results = {
        "DSM_ndcg": DSM_ndcg,
        "DSM_dcg": DSM_dcg,
        #"DSM_mean_abs_viol": DSM_mean_abs_viol,
        #"DSM_mean_viol": DSM_mean_viol,
        #"DSM_max_viol": DSM_max_viol,
        "ndcg": avg_ndcg,
        "dcg": avg_dcg,
        "avg_rank": avg_rank
        #"fair_viols_quantiles":fair_viols_quantiles
    }
    """
    if group_fairness_evaluation:
        # convert lists in dictionary to np arrays
        for disparity_type in group_disparities:
            group_disparities[disparity_type] = np.mean(
                group_disparities[disparity_type])

        other_disparities = {}
        for k, v in group_disparities.items():
            if k == 'ashudeep' or k == 'ashudeep_mod':
                disparity = v
                asym_disparity = v
            else:
                if args.indicator_type == "square":
                    disparity = v
                    asym_disparity = v ** 2
                elif args.indicator_type == "sign":
                    disparity = v
                    asym_disparity = abs(v)
                elif args.indicator_type == "none":
                    disparity = v
                    asym_disparity = v
                else:
                    raise NotImplementedError
            if k == args.disparity_type:
                avg_group_exposure_disparity = disparity
                avg_group_asym_disparity = asym_disparity
            other_disparities[k] = [asym_disparity, disparity]

        results.update({
            "avg_group_disparity": avg_group_exposure_disparity,
            "avg_group_asym_disparity": avg_group_asym_disparity
        })
        if track_other_disparities:
            results.update({"other_disparities": other_disparities})
    """
    return results








# JK
# Test-time evaluation for soft_policy_training
def evaluate_quantiles(model,
                       validation_data,
                       group0_merit = None,   # JK
                       group1_merit = None,   # JK
                       num_sample_per_query=10,
                       deterministic=False,
                       fairness_evaluation=False,
                       position_bias_vector=None,
                       group_fairness_evaluation=False,
                       track_other_disparities=False,
                       args=None,
                       normalize=False,
                       noise=None,
                       max_sample_eval = 1280,     # JK
                       en=None):
    if noise is None:
        noise = args.noise
    if en is None:
        en = args.en
    ndcg_list = []
    dcg_list = []
    rank_list = []
    weight_list = []
    DSM_ndcg_list = []   #JK
    mean_fair_viol_list = []
    max_fair_viol_list = []
    if (fairness_evaluation
            or group_fairness_evaluation) and position_bias_vector is None:
        position_bias_vector = 1. / torch.arange(
            1., 100.) ** args.position_bias_power
        if args.gpu:
            position_bias_vector = position_bias_vector.cuda()

    val_feats, val_rel = validation_data

    # JK limit the validation set for this
    val_feats = val_feats[:max_sample_eval]
    val_rel   = val_rel[:max_sample_eval]

    all_exposures = []
    all_rels = []

    relu = nn.ReLU()

    validation_dataset = torch.utils.data.TensorDataset(val_feats, val_rel)
    dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size)
    if args.progressbar:
        dataloader = tqdm(dataloader)

    if group_fairness_evaluation:
        if track_other_disparities:
            disparity_types = ['disp0','disp1', 'disp2', 'disp3', 'ashudeep', 'ashudeep_mod']   # JK add disp0
        else:
            disparity_types = [args.disparity_type]
        if 'disp2' in disparity_types or 'ashudeep_mod' in disparity_types:
            group0_merit, group1_merit = get_group_merits(
                val_feats,
                val_rel,
                args.group_feat_id,
                args.group_feat_threshold,
                mean=False
            )
            sign = 1.0 if group0_merit >= group1_merit else -1.0
        else:
            #group0_merit, group1_merit = None, None    # JK why is this here
            sign = None
        group_disparities = {
            disparity_type: [] for disparity_type in disparity_types
        }
    model.eval()
    with torch.no_grad():


        # Initialize solvers
        ##############
        ####### added in assuming we'll use SPO from now on
        solver_dict = {}
        for i in range(1,args.list_len):

            if args.allow_unfairness:
                # Delta Fairness
                # Google solver only
                gids = torch.zeros(args.list_len).long()
                gids[:i] = 1
                s,x = ort_setup_Neq(args.list_len, gids, args.disparity_type, group0_merit, group1_merit, args.fairness_gap)
                key = int(gids.sum().item())      # JK check this key - not used?
                solver_dict[i] = ort_policyLP(s,x)
            else:
                # Perfect Fairness
                gids = torch.zeros(args.list_len).long()
                gids[:i] = 1
                s,x = ort_setup(args.list_len, gids, args.disparity_type, group0_merit, group1_merit)
                key = int(gids.sum().item())      # JK check this key - not used?
                solver_dict[i] = ort_policyLP(s,x)

        #######
        ##############


        for i,data in enumerate(dataloader):
            # print('i = {}'.format(i))

            feats, rel = data
            batsize = feats.shape[0]

            group_identities = get_group_identities(feats, args.group_feat_id, args.group_feat_threshold)

            if group_identities.bool().all(1).any().item() or (1-group_identities).bool().all(1).any().item():
                continue
                # skip the iteration if only one group appears


            if args.embed_groups:
                scores, group_embed = model(feats, group_identities)
                scores= scores.squeeze(-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(batsize,-1,1), group_embed.unsqueeze(0).view(batsize,-1,1).permute(0,2,1)  ).reshape(batsize,-1)
            # Concatenate the document scores with group ID and predict N**2 independent QP coefficients using a MLP
            elif args.embed_quadscore:
                score_cross = model(feats, group_identities).squeeze(-1)
                #score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
            else:
                scores = model(feats).squeeze(-1)
                test_dscts = ( 1.0 / torch.log2(torch.arange(args.list_len).float() + 2) ).repeat(batsize,1,1)
                #score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
                score_cross = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), test_dscts.view(batsize,1,-1)  ).reshape(batsize,-1)


            test_dscts = ( 1.0 / torch.log2(torch.arange(args.list_len).float() + 2) ).repeat(batsize,1,1)
            true_costs = torch.bmm( rel.view(batsize,-1,1), test_dscts.view(batsize,1,-1)).view(batsize,1,-1)


            grad = []
            p_mat = []
            regrets = []
            with torch.no_grad():
                dcg_max = compute_dcg_max(rel)  # redundant, defined again below

                for i in range(batsize):

                    spo_group_ids = group_identities[i].detach().numpy()
                    sorting_ind = np.argsort(spo_group_ids)[::-1]
                    reverse_ind = np.argsort(sorting_ind)

                    solver = solver_dict[ int(spo_group_ids.sum().item()) ]

                    V_true  = true_costs[i].squeeze().detach().double().numpy() #compute 'true' cost coefficients here
                    V_true1 = true_costs[i].squeeze().detach().double().numpy()                    #delete
                    V_true = (V_true.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()

                    sol_true = solver.solve(V_true)
                    sol_true = sol_true.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                    V_pred  = score_cross[i].squeeze().detach().double().numpy() #compute 'pred' cost coefficients here

                    V_pred = (V_pred.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                    sol_pred = solver.solve(V_pred)
                    sol_pred = sol_pred.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                    p_mat.append(torch.Tensor(sol_pred).view(args.list_len,args.list_len))


                    V_spo   = (2*V_pred - V_true)
                    V_spo   = (V_spo.reshape((args.list_len,args.list_len)))[sorting_ind].flatten()
                    sol_spo  = solver.solve(V_spo)
                    sol_spo  = sol_spo.reshape((args.list_len,args.list_len))[reverse_ind].flatten()

                    #reg = torch.dot(V_true1,(sol_true - sol_pred))
                    reg = torch.Tensor(  [np.dot(V_true1,(sol_true - sol_pred))]  )
                    regrets.append(reg)
                    use_reg = True
                    if use_reg:
                        grad.append( torch.Tensor(sol_spo - sol_true)  /  dcg_max[i]  )
                    else:
                        grad.append( torch.Tensor(sol_spo - sol_true)  )


                p_mat = torch.stack(p_mat)

            #######
            ################
            ################################



            if deterministic:
                num_sample_per_query = 1
                rankings = torch.sort(
                    scores,
                    descending=True, dim=-1)[1].unsqueeze(1)
            else:
                # JK replace old sampling method with this one
                with torch.no_grad():
                    P = p_mat.cpu().detach().numpy()
                    #max_instances_sample = 200 #min(200, P.shape[0]) # Take a max of 200 from each batch
                    #P = P[np.random.choice(P.shape[0],max_instances_sample,replace = True)]
                    R = []
                    for it, policy in enumerate(P):
                        decomp = birkhoff_von_neumann_decomposition(policy)
                        convex_coeffs, permutations = zip(*decomp)
                        permutations = np.array(permutations)
                        rolls = torch.multinomial(torch.Tensor(convex_coeffs),num_sample_per_query,replacement=True).numpy()
                        #rolls = np.random.multinomial(sample_size, np.array(convex_coeffs))  # sample the permutations based on convex_coeffs
                        p_sample = permutations[rolls]       # access the permutations
                        r_sample = p_sample.argmax(2)        # convert to rankings
                        r_sample = torch.tensor( r_sample )  # convert to same datatype as FULTR implementation
                        R.append(r_sample)
                        #print("Finished policy sampling iteration {}".format(it))
                    rankings = torch.stack(R)
                    if args.gpu:
                        rankings = rankings.cuda()   # JK testing

                ############
                # Soft evaluation metrics

                with torch.no_grad():

                    print("evaluating ")

                    dcg_max = compute_dcg_max(rel)
                    test_dscts = ( 1.0 / torch.log2(torch.arange(args.list_len).float() + 2) ).repeat(batsize,1,1)
                    if args.gpu:
                        test_dscts = test_dscts.cuda()
                    #v_unsq = v.unsqueeze(1)
                    #f_unsq = f.unsqueeze(1).permute(0,2,1)
                    #vXf = torch.bmm(f_unsq,v_unsq).view(-1,group_ids.shape[1]**2).unsqueeze(1).to(self._device) # this is still a batch
                    loss_a = torch.bmm( p_mat, test_dscts.view(batsize,-1,1) )
                    loss_b = torch.bmm( rel.view(batsize,1,-1), loss_a ).squeeze()
                    loss_norm = loss_b.squeeze() / dcg_max
                    loss = loss_norm.mean()


                    DSM_ndcg_list.append(loss)

                    # Find average violation
                    #fair_viol_mean_batch = 0
                    #for kk in range(len(p_mat)):
                    #    fair_viol_mean_batch += test_fairness( p_mat[kk], group_identities[kk], position_bias_vector )
                    #fair_viol_mean_batch /= len(p_mat)

                    fair_viol_batch_list = []
                    for kk in range(len(p_mat)):
                        fair_viol_batch_list.append(   np.abs(  test_fairness( p_mat[kk], group_identities[kk], position_bias_vector, args.disparity_type, group0_merit, group1_merit )   )   )
                        # JK absolute value added 10/27
                    fair_viol_mean_batch = np.mean(fair_viol_batch_list).item()
                    fair_viol_max_batch  = np.max(fair_viol_batch_list).item()

                    DSM_ndcg_list.append( loss.item() )
                    mean_fair_viol_list.append( fair_viol_mean_batch )
                    max_fair_viol_list.append(fair_viol_max_batch)

                    #print("viol = ")
                    #print( fair_viol_mean_batch )
                    #input()

                # END Soft evaluation metrics
                ############



            ndcgs, dcgs = compute_dcg_rankings(rankings, rel)

            print(" ndcgs.mean() = ")
            print(  ndcgs.mean()    )

            rank = compute_average_rank(rankings, rel)
            dcg_list  += dcgs.mean(dim=-1).tolist()
            ndcg_list += ndcgs.mean(dim=-1).tolist()
            rank_list += rank.mean(dim=-1).tolist()
            weight_list += rel.sum(dim=-1).tolist()

            if group_fairness_evaluation:
                group_identities = get_group_identities(
                    feats,
                    args.group_feat_id,
                    args.group_feat_threshold
                )
                inds_g0 = group_identities == 0
                inds_g1 = group_identities == 1

                if args.unweighted_fairness:
                    rel = (rel > 0.0).float()

                for disparity_type in disparity_types:
                    if disparity_type == 'ashudeep':
                        disparity = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector).mean(dim=-1)
                    elif disparity_type == 'ashudeep_mod':
                        disparity = BaselineAshudeepGroupFairnessLoss.compute_group_fairness_coeffs_generic(
                            rankings, rel, group_identities, position_bias_vector, sign=sign).mean(
                            dim=-1)
                    else:
                        disparity = GroupFairnessLoss.compute_multiple_group_disparity(
                            rankings,
                            rel,
                            group_identities,
                            group0_merit,
                            group1_merit,
                            position_bias_vector,
                            disparity_type=disparity_type,
                            noise=noise,
                            en=en
                        ).mean(dim=-1)
                    for i in range(len(rankings)):
                        if inds_g0[i].any() and inds_g1[i].any():
                            group_disparities[disparity_type].append(disparity[i].item())

    model.train()
    avg_ndcg = np.mean(ndcg_list)
    if normalize:
        avg_dcg = np.sum(dcg_list) / np.sum(weight_list)
        avg_rank = np.sum(rank_list) / np.sum(weight_list)
    else:
        avg_dcg = np.mean(dcg_list)
        avg_rank = np.mean(rank_list)

    DSM_ndcg = np.mean(DSM_ndcg_list)                      # JK
    DSM_mean_viol = np.mean(  mean_fair_viol_list  )
    DSM_max_viol  = np.max(    max_fair_viol_list  )


    results = {
        "DSM_ndcg": DSM_ndcg,
        "DSM_mean_viol": DSM_mean_viol,
        "DSM_max_viol": DSM_max_viol,
        "ndcg": avg_ndcg,
        "dcg": avg_dcg,
        "avg_rank": avg_rank
    }
    if group_fairness_evaluation:
        # convert lists in dictionary to np arrays
        for disparity_type in group_disparities:
            group_disparities[disparity_type] = np.mean(
                group_disparities[disparity_type])

        other_disparities = {}
        for k, v in group_disparities.items():
            if k == 'ashudeep' or k == 'ashudeep_mod':
                disparity = v
                asym_disparity = v
            else:
                if args.indicator_type == "square":
                    disparity = v
                    asym_disparity = v ** 2
                elif args.indicator_type == "sign":
                    disparity = v
                    asym_disparity = abs(v)
                elif args.indicator_type == "none":
                    disparity = v
                    asym_disparity = v
                else:
                    raise NotImplementedError
            if k == args.disparity_type:
                avg_group_exposure_disparity = disparity
                avg_group_asym_disparity = asym_disparity
            other_disparities[k] = [asym_disparity, disparity]

        results.update({
            "avg_group_disparity": avg_group_exposure_disparity,
            "avg_group_asym_disparity": avg_group_asym_disparity
        })
        if track_other_disparities:
            results.update({"other_disparities": other_disparities})

    return results






def add_tiny_noise(one_hot_rel):
    """
    used to add tiny noise to avoid warnings in linregress
    """
    if one_hot_rel.min() == one_hot_rel.max():
        one_hot_rel = one_hot_rel + np.random.random(len(one_hot_rel)) * 1e-20
    return one_hot_rel


def optimal_exposure(num_relevant, num_docs, position_bias_function):
    """
    returns the optimal exposure that a randomized policy can give for
    the given number of relevant documents
    """
    top_k_exposure = np.mean(
        [position_bias_function(i) for i in range(num_relevant)])
    remaining_exposure = np.mean(
        [position_bias_function(i) for i in range(num_relevant, num_docs)])
    optimal_exposure = [top_k_exposure
                        ] * num_relevant + [remaining_exposure] * (
        num_docs - num_relevant)
    return optimal_exposure
