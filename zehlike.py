import torch
from utils import shuffle_combined
import numpy as np
from YahooDataReader import YahooDataReader
from models import NNModel, LinearModel        , MLP  # JK 11/11
from utils import parse_my_args_reinforce, torchify,      transform_dataset   #JK
from evaluation import evaluate_model   # JK argpase error (argument not found) happens here
from baselines import vvector
from progressbar import progressbar

# JK new
from parse_args import args
import ast
from datareader import reader_from_pickle
from fairness_loss import  get_group_identities
import copy
import pickle as pkl
import pandas as pd


def demographic_parity_train(model, dr, vdr, tdr, vvector, args, group0_merit, group1_merit):






    # JK 11/12
    feat, rel = dr    # dr.data    # JK 11/12
    #feat, rel = shuffle_combined(feat, rel)
    train_dataset = torch.utils.data.TensorDataset(feat, rel)
    len_train_set = len(feat) // args.batch_size + 1
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    flag_training = False
    JK_best_model = model
    patience = 15
    best_so_far = -8000
    #


    N = len(rel)
    from utils import get_optimizer
    optimizer = get_optimizer(
        model.parameters(),
        args.lr,    #[0],   # JK 11/11
        args.optimizer,
        weight_decay=0.0)   #      args.weight_decay[0])  # JK 11/11    0.0 is default

    for epoch in range(args.epochs):    #[0]):   # JK 11/11
        for batch_id, data in enumerate(train_dataloader):

            #print("Entering batch {}".format(batch_id))


            feats, rel = data


            #feat, rel = shuffle_combined(feat, rel)    # JK 11/12 moved outside


            optimizer.zero_grad()
            #curr_feats = feat[i]
            curr_feats = feats   # JK 11/12
            scores = model(curr_feats).squeeze()
            #scores = model(torchify(curr_feats)).squeeze()   # JK 11/11

            #probs = torch.nn.Softmax(dim=0)(scores)

            probs = torch.nn.Softmax(dim=1)(scores)  # JK 11/11

            #if rel[i].sum() == 0:       #np.sum(rel[i]) == 0:  # JK
            #    continue
            #normalized_rels = rel[i]  # / np.sum(rel[i])
            normalized_rels = torch.nn.Softmax(dim=1)(rel)   # JK 11/11


            # np.random.shuffle(normalized_rels)

            #ranking_loss = -torch.sum(
            #    torch.FloatTensor(normalized_rels) * torch.log(probs))
            ranking_loss = -torch.sum(
                normalized_rels * torch.log(probs))     # JK

            # print(scores, probs,
            #       torch.log(probs), normalized_rels,
            #       torch.log(probs) * torch.FloatTensor(normalized_rels),
            #       ranking_loss)

            exposures = vvector[0] * probs
            #groups = curr_feats[:, args.group_feat_id]

            groups = get_group_identities(
                curr_feats, args.group_feat_id, args.group_feat_threshold)

            """
            print("feats = ")
            print( feats )
            print("feats.shape = ")
            print( feats.shape )
            print("args.group_feat_id = ")
            print( args.group_feat_id    )
            print("args.group_feat_threshold = ")
            print( args.group_feat_threshold )
            print("groups = ")
            print( groups )
            print("probs = ")
            print( probs )
            print("exposures = ")
            print( exposures )
            print("scores = ")
            print( scores )
            print("probs = ")
            print( probs )
            """

            #if np.all(groups == 0) or np.all(groups == 1):   #JK 11/11
            if (groups == 0).all() or (groups == 1).all():
                fairness_loss = 0.0
            else:
                avg_exposure_0 = torch.sum(
                    torch.FloatTensor(1 - groups) * exposures) / torch.sum(
                        1 - torch.FloatTensor(groups))
                avg_exposure_1 = torch.sum(
                    torch.FloatTensor(groups) * exposures) / torch.sum(
                        torch.FloatTensor(groups))
                # print(avg_exposure_0, avg_exposure_1)
                fairness_loss = torch.pow(
                    torch.clamp(avg_exposure_1 - avg_exposure_0, min=0), 2)
            #loss = args.lambda_reward * ranking_loss + args.lambda_group_fairness * fairness_loss
            loss = 1.0 * ranking_loss + args.lambda_group_fairness * fairness_loss    # JK 11/11

            print("args.lambda_group_fairness = ")
            print( args.lambda_group_fairness )

            #print("loss = {}      {}".format(ranking_loss,args.lambda_group_fairness * fairness_loss ))

            loss.backward()
            optimizer.step()
            # break

        # end of epoch
        #if i % args.evaluate_interval == 0 and i != 0:  # JK 11/11
        results = evaluate_model(
            model,
            vdr,
            group0_merit = group0_merit,
            group1_merit = group1_merit,
            fairness_evaluation=False,
            group_fairness_evaluation=True,
            deterministic=True,
            args=args,
            num_sample_per_query=100)
        #print(results)
        print( results['ndcg'] )
        print( results['avg_group_disparity'] )


        valid_ndcg_final = results["ndcg"]      # JK evaluation.py line 504 for origin of these
        valid_dcg_final  = results["dcg"]
        valid_rank_final = results["avg_rank"]
        #if group_fairness_evaluation:
        valid_abs_group_expos_disp_final = results["avg_abs_group_disparity"]
        valid_group_expos_disp_final = results["avg_group_disparity"]
        valid_group_asym_disp_final = results["avg_group_asym_disparity"]
        fair_viols_quantiles_valid = results["fair_viols_quantiles"]
        # JK end test metric collection

        #valid_ndcg_list_plot.append( valid_ndcg_final )
        #valid_viol_list_plot.append( valid_group_asym_disp_final )

        stop_metric = 1.0 * valid_dcg_final
        if args.lambda_group_fairness > 0:
            stop_metric -= args.lambda_group_fairness * valid_group_asym_disp_final

        if  stop_metric > ( best_so_far + 1e-3):
            JK_best_model = copy.deepcopy(model)
            time_since_best = 0
            best_so_far = stop_metric
            results_valid_best = results.copy()
        else:
            time_since_best = time_since_best + 1

        print("time_since_best = {}".format(time_since_best))

        if time_since_best > patience:
            print("Early Stopping. Valid hasn't improved for {}".format(patience))
            flag_training = True

        if flag_training:
            break





    valid_ndcg_final = results_valid_best["ndcg"]
    valid_dcg_final  = results_valid_best["dcg"]
    valid_rank_final = results_valid_best["avg_rank"]
    #if group_fairness_evaluation:
    valid_abs_group_expos_disp_final = results_valid_best["avg_abs_group_disparity"]
    valid_group_expos_disp_final = results_valid_best["avg_group_disparity"]
    valid_group_asym_disp_final = results_valid_best["avg_group_asym_disparity"]
    fair_viols_quantiles_valid = results_valid_best["fair_viols_quantiles"]



    results = evaluate_model(
        JK_best_model,
        tdr,
        fairness_evaluation=False,
        group_fairness_evaluation=True,
        deterministic=True,
        args=args,
        num_sample_per_query=100,
        group0_merit = group0_merit,
        group1_merit = group1_merit
        )
    test_ndcg_final = results["ndcg"]      # JK evaluation.py line 504 for origin of these
    test_dcg_final  = results["dcg"]
    test_rank_final = results["avg_rank"]
    #if group_fairness_evaluation:
    test_abs_group_expos_disp_final = results["avg_abs_group_disparity"]
    test_group_expos_disp_final = results["avg_group_disparity"]
    test_group_asym_disp_final = results["avg_group_asym_disparity"]
    fair_viols_quantiles_test  = results["fair_viols_quantiles"]


    csv_outs = {}
    #csv_outs['entropy_final']  =  entropy_writelist_JK[-1]
    #csv_outs["rewards_final"]  =  rewards_writelist_JK[-1]
    #if args.lambda_group_fairness != 0.0:
    #    csv_outs["fairness_loss_final"] =  fairness_loss_writelist_JK[-1]
    #    csv_outs["max_fairness_loss_final"] =  max_fairness_loss_writelist_JK[-1]
    #csv_outs["reward_variance_final"] = reward_variance_writelist_JK[-1]
    #csv_outs["train_ndcg_final"] = train_ndcg_final
    #csv_outs["train_dcg_final"] = train_dcg_final
    #csv_outs["train_rank_final"] = train_rank_final
    #csv_outs["train_abs_group_expos_disp_final"] = train_abs_group_expos_disp_final
    #csv_outs["train_group_expos_disp_final"] = train_group_expos_disp_final
    #csv_outs["train_group_asym_disp_final"] = train_group_asym_disp_final
    csv_outs["test_ndcg_final"] = test_ndcg_final
    csv_outs["test_dcg_final"] = test_dcg_final
    csv_outs["test_rank_final"] = test_rank_final
    csv_outs["test_abs_group_expos_disp_final"] = test_abs_group_expos_disp_final
    csv_outs["test_group_expos_disp_final"] = test_group_expos_disp_final
    csv_outs["test_group_asym_disp_final"] = test_group_asym_disp_final
    csv_outs["valid_ndcg_final"] = valid_ndcg_final
    csv_outs["valid_dcg_final"] = valid_dcg_final
    csv_outs["valid_rank_final"] = valid_rank_final
    csv_outs["valid_abs_group_expos_disp_final"] = valid_abs_group_expos_disp_final
    csv_outs["valid_group_expos_disp_final"] = valid_group_expos_disp_final
    csv_outs["valid_group_asym_disp_final"] = valid_group_asym_disp_final
    #csv_outs["fair_viol_q_100"] = fair_viols_quantiles['1.00']
    #csv_outs["fair_viol_q_95"]  = fair_viols_quantiles['0.95']
    #csv_outs["fair_viol_q_90"]  = fair_viols_quantiles['0.90']
    #csv_outs["fair_viol_q_85"]  = fair_viols_quantiles['0.85']
    #csv_outs["fair_viol_q_80"]  = fair_viols_quantiles['0.80']
    #csv_outs["fair_viol_q_75"]  = fair_viols_quantiles['0.75']
    #csv_outs["fair_viol_q_70"]  = fair_viols_quantiles['0.70']
    #csv_outs["fair_viol_q_65"]  = fair_viols_quantiles['0.65']
    #csv_outs["fair_viol_q_60"]  = fair_viols_quantiles['0.60']
    #csv_outs["fair_viol_q_55"]  = fair_viols_quantiles['0.55']
    #csv_outs["fair_viol_q_50"]  = fair_viols_quantiles['0.50']
    csv_outs["fair_viol_q_100_test"] = fair_viols_quantiles_test['1.00']
    csv_outs["fair_viol_q_95_test"]  = fair_viols_quantiles_test['0.95']
    csv_outs["fair_viol_q_90_test"]  = fair_viols_quantiles_test['0.90']
    csv_outs["fair_viol_q_85_test"]  = fair_viols_quantiles_test['0.85']
    csv_outs["fair_viol_q_80_test"]  = fair_viols_quantiles_test['0.80']
    csv_outs["fair_viol_q_75_test"]  = fair_viols_quantiles_test['0.75']
    csv_outs["fair_viol_q_70_test"]  = fair_viols_quantiles_test['0.70']
    csv_outs["fair_viol_q_65_test"]  = fair_viols_quantiles_test['0.65']
    csv_outs["fair_viol_q_60_test"]  = fair_viols_quantiles_test['0.60']
    csv_outs["fair_viol_q_55_test"]  = fair_viols_quantiles_test['0.55']
    csv_outs["fair_viol_q_50_test"]  = fair_viols_quantiles_test['0.50']
    csv_outs["fair_viol_q_100_valid"] = fair_viols_quantiles_valid['1.00']
    csv_outs["fair_viol_q_95_valid"]  = fair_viols_quantiles_valid['0.95']
    csv_outs["fair_viol_q_90_valid"]  = fair_viols_quantiles_valid['0.90']
    csv_outs["fair_viol_q_85_valid"]  = fair_viols_quantiles_valid['0.85']
    csv_outs["fair_viol_q_80_valid"]  = fair_viols_quantiles_valid['0.80']
    csv_outs["fair_viol_q_75_valid"]  = fair_viols_quantiles_valid['0.75']
    csv_outs["fair_viol_q_70_valid"]  = fair_viols_quantiles_valid['0.70']
    csv_outs["fair_viol_q_65_valid"]  = fair_viols_quantiles_valid['0.65']
    csv_outs["fair_viol_q_60_valid"]  = fair_viols_quantiles_valid['0.60']
    csv_outs["fair_viol_q_55_valid"]  = fair_viols_quantiles_valid['0.55']
    csv_outs["fair_viol_q_50_valid"]  = fair_viols_quantiles_valid['0.50']
    csv_outs["stop_epoch"] = epoch




    csv_outs["index"] = args.index
    csv_outs["epochs"] = args.epochs
    csv_outs["lr"] = args.lr
    csv_outs["hidden_layer"] = args.hidden_layer
    csv_outs["optimizer"] = args.optimizer
    csv_outs["quad_reg"] = args.quad_reg
    csv_outs["partial_train_data"] = args.partial_train_data
    csv_outs["partial_val_data"] = args.partial_val_data
    csv_outs["full_test_data"] = args.full_test_data
    csv_outs["log_dir"] = args.log_dir
    csv_outs["sample_size"] = args.sample_size
    csv_outs["batch_size"] = args.batch_size
    csv_outs["soft_train"] = args.soft_train
    csv_outs["disparity_type"] = args.disparity_type
    csv_outs["lambda_group_fairness"] = args.lambda_group_fairness
    csv_outs["index"] = args.index
    csv_outs["dropout"] = args.dropout
    csv_outs["gme_new"] = args.gme_new

    csv_outs = {k:[v] for (k,v) in csv_outs.items()   }
    df_outs = pd.DataFrame.from_dict(csv_outs)
    outPathCsv = './csv/'+ "FULTR_" + args.output_tag + '_' + str(args.index)  + ".csv"

    df_outs.to_csv(outPathCsv)



    for (k,v) in csv_outs.items():
        print("{}:  {}".format(k,v))

    print("Outputs saved")



    quit()
    return JK_best_model


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":


    # JK 11/11
    if args.disparity_type == 'disp0':
        group0_merit = 1.0     # TODO write these in for the different datasets
        group1_merit = 1.0
    else:
        if 'mslr' in args.partial_train_data.lower():
            group0_merit = 1.91391408523019
            group1_merit = 2.5832905470933882
        else:
            group0_merit = 3.1677021123091107
            group1_merit = 1.415066736141729
    #


    train_data = reader_from_pickle(args.partial_train_data)
    train_data = train_data.data
    dr = transform_dataset(train_data, args.gpu, args.weighted)

    valid_data = reader_from_pickle(args.full_test_data)
    valid_data = valid_data.data
    vdr = transform_dataset(valid_data, args.gpu, True)  # JK new, previously done below (commented)

    test_data = reader_from_pickle(args.full_test_data)
    test_data = test_data.data
    tdr = transform_dataset(test_data, args.gpu, True)  # JK new, previously done below (commented)


    # JK 11/11   load lambda
    #a = ast.literal_eval(args.lambda_list)
    #lambdas_list = [float(c) for c in a]
    #args.lambda_group_fairness = lambdas_list[-1]
    #


    kwargs = {'clamp': args.clamp}
    if args.mask_group_feat:
        kwargs['masked_feat_id'] = args.group_feat_id
    model = MLP(args.input_dim, args.hidden_layer, args.dropout, **kwargs)

    model = demographic_parity_train(model, dr, vdr, tdr, vvector(200), args, group0_merit, group1_merit)
