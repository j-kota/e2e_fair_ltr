import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inputs",    type=str,    default="",  help="List of comma separated inputs")
args = parser.parse_args()

#23
params  = [ "lambda_group_fairness",
            "epochs",
            "lr",
            "hidden_layer",
            "seed",
            "dropout",
            "partial_train_data",
            "partial_val_data",
            "full_test_data",
            "quad_reg",
            "sample_size",
            "batch_size",
            "soft_train",
            "index",
            "allow_unfairness",
            "fairness_gap",
            "mode",
            "multi_groups",
            "entreg_decay",
            "evaluate_interval",
            "output_tag",
            "disparity_type"]
            ##--indicator_type square
            ##--reward_type dcg
            ##--log_dir runs/default_JK


"""
0.1,
1000,
1e-05,
5,
adam,
0.0,
/home/jkotary/fultr/transformed_datasets/mslr/Train/partial_train_36k.pkl,
/home/jkotary/fultr/transformed_datasets/mslr/Train/partial_valid_4k.pkl,
/home/jkotary/fultr/transformed_datasets/mslr/full/test.pkl,
0,
128,
64,
1,
1,
1,
1e-07,
0,
2,
0.1,
2000,
LP_tests_multi_disp12021-11-17_,
disp0
"""






#str  = "[0.1],100,0.001,1,adam,0.1,/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_5k.pkl,/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_5k.pkl,/home/jkotary/fultr/transformed_datasets/german/full/test.pkl,runs/default_JK,32,16,1,1,0,0.0,1,0,0.1,500,5ktests_altscoring,0"

tokens = args.inputs.split(',')
tokens[-2]

output = ""
if len(tokens) != len(params):
    print('Number of arguments and values not equal')
else:
    for i in range(len(tokens)):
        output += "--"+params[i] + " " +tokens[i] + " "

output += "--indicator_type square  --reward_type dcg  --log_dir runs/default_JK"

print(output)
