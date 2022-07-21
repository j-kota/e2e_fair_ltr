import argparse
from utils import unserialize, add_bool_arg


parser = argparse.ArgumentParser()

#parser.add_argument("lambda_list", type=str, default="[0.0]")
parser.add_argument("--args_file", type=str, default=None)
add_bool_arg(parser, "tuning", default=False)
parser.add_argument("--load_model", type=str, default=None)

parser.add_argument(
    "--partial_train_data", type=str,
    default="GermanCredit/prod/35:35:30_group14_double/split2/train-noise-0.1/partial_train_5k.pkl")
parser.add_argument(
    "--partial_val_data", type=str,
    default="GermanCredit/prod/35:35:30_group14_double/split2/train-noise-0.1/partial_valid_5k.pkl")
parser.add_argument(
    "--partial_test_data", type=str,
    default="GermanCredit/prod/35:35:30_group14_double/split2/test/partial_test_5k.pkl")

parser.add_argument(
    "--full_train_data", type=str,
    default="GermanCredit/prod/35:35:30_group14_double/split2/full/train.pkl")
parser.add_argument(
    "--full_val_data", type=str,
    default="GermanCredit/prod/35:35:30_group14_double/split2/full/valid.pkl")
parser.add_argument(
    "--full_test_data", type=str,
    default="GermanCredit/prod/35:35:30_group14_double/split2/full/test.pkl")

parser.add_argument("--fullinfo", type=str, default="partial")
parser.add_argument("--log_dir", type=str,
                    default="runs/new/try18_disp3_gp14_split0")
parser.add_argument("--hyperparam_folder", type=str,
                    default="disp3_gp14_split0")
parser.add_argument("--experiment_prefix", type=str,
                    default="tuning_disp3_gp14_split0")

add_bool_arg(parser, "gpu", False)
parser.add_argument("--gpu_id", type=int, default=0)
#parser.add_argument("--num_cores", type=int, default=-1)
parser.add_argument("--num_cores", type=int, default=6)   #JK
add_bool_arg(parser, "clamp", default=False)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--hidden_layer", type=int, default=None)
parser.add_argument("--input_dim", type=int, default=58)

add_bool_arg(parser, "mask_group_feat", default=False)
add_bool_arg(parser, "unweighted_fairness", default=False)
parser.add_argument("--group_feat_id", type=int, default=14)
parser.add_argument("--group_feat_threshold", type=float, default=None)
parser.add_argument("--group_disparity_indicator_batch_size",
                    type=int, default=250)
parser.add_argument("--position_bias_power", type=float, default=1.0)
parser.add_argument("--indicator_type", type=str, default="square", choices=['square', 'sign', 'none'])
parser.add_argument(
    "--disparity_type", type=str, default="disp0",
    choices=['disp0', 'disp1', 'disp2', 'disp3', 'ashudeep', 'ashudeep_mod']) # JK added choice disp0
add_bool_arg(parser, "track_other_disparities", False)

add_bool_arg(parser, "weighted", True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--sample_size", type=int, default=32)
parser.add_argument("--reward_type", type=str, default="dcg")
parser.add_argument("--baseline_type", type=str, default="value")
add_bool_arg(parser, "use_baseline", True)
parser.add_argument("--entropy_regularizer", type=float, default=0.0)
parser.add_argument("--entreg_decay", type=float, default=0.3)

parser.add_argument("--epochs", type=int, default=2000)
add_bool_arg(parser, "early_stopping", default=False)
parser.add_argument("--stop_patience", type=int, default=40)

parser.add_argument("--evalk", type=int, default=1000)
parser.add_argument("--eval_temperature", type=float, default=1.0)
parser.add_argument("--evaluate_interval", type=int, default=3000)
parser.add_argument("--eval_rank_limit", type=int, default=1000)
add_bool_arg(parser, "eval_other_train", default=False)
add_bool_arg(parser, "eval_weighted_val", default=False)
parser.add_argument("--eval_other_train_location", type=str,
                    default="GermanCredit/german_train_rank.pkl")
parser.add_argument(
    "--eval_weighted_val_location", type=str,
    default="GermanCredit/partial_german_test_rank_weightedclick_5k.pkl")
add_bool_arg(parser, "validation_deterministic", default=False)
add_bool_arg(parser, "evaluation_deterministic", default=False)

parser.add_argument("--lr", type=float, default=0.001)
add_bool_arg(parser, "lr_scheduler", True)
parser.add_argument("--lr_decay", type=float, default=0.2)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--weight_decay", type=float, default=0.0)

parser.add_argument("--pooling", type=str, default='concat_avg')
add_bool_arg(parser, "progressbar", False)
add_bool_arg(parser, "summary_writing", True)
add_bool_arg(parser, "save_checkpoints", True)
parser.add_argument("--write_losses_interval", type=int, default=1000)

# JK 0811
parser.add_argument("--DSM", type=int, default=0)
parser.add_argument("--DSMfair", type=int, default=0)
parser.add_argument("--soft_train", type=int, default=0)
parser.add_argument("--allow_unfairness", type=int, default=0)  # if fairness_gap=0, this turns equality constraints into equivalent inequalities
parser.add_argument("--fairness_gap", type=float, default=0.0)  # if allow_unfairness=0, this has no effect
parser.add_argument("--quad_reg", type=float, default=0.1)
parser.add_argument("--index", type=int, default = 9999)
parser.add_argument("--embed_groups", type=int, default = 0)
parser.add_argument("--embed_quadscore", type=int, default = 0)
parser.add_argument("--list_len", type=int, default = 20)  # length of lists to be ranked; needed only due to JK mods (group embedding)
parser.add_argument("--solver_software", type=str, default = 'google')
parser.add_argument("--output_tag", type=str, default = 'notag')
parser.add_argument("--seed", type=int, default = 9999)
parser.add_argument("--multi_groups", type=int, default=0)   # values of 1 and 2 are not appropriate; 0 indicates 2 groups (original)
parser.add_argument("--gme_new", type=int, default=1)   # 0 indicates original get_multiple_exposures function; 1 indicates the revision by JK
parser.add_argument("--lambda_group_fairness", type=float, default=0.000000000001)
parser.add_argument("--mode", type=str, default = 'spo')   # spo, bb, qp, int

add_bool_arg(parser, "noise", False)
parser.add_argument("--en", type=float, default=0.1)

args = parser.parse_args()


if 'mslr' in args.partial_train_data.lower():    # JK watch out for when partial_train_data isn't used
    args.group_feat_id = 132
    args.input_dim = 136
    args.dataset = 'mslr'
    args.group_feat_threshold = 0.03252032399177551   # JK taken from my_dispatcher.py
else:
    args.dataset = 'german'             # JK careful if you add more datasets

if args.args_file is not None:
    args_file = unserialize(args.args_file)
    for key, value in args_file.items():
        if key in args.__dict__:
            args.__dict__[key] = value
        else:
            print(key)
            raise NotImplementedError
