# e2e_fair_ltr
End-to-End Learning for Fair Ranking Systems

Learning to rank subject to constraints on group exposure, using differentiable optimization modules in predict-and-optimize fashion.


python run_hyperparams.py --lambda_group_fairness 0.1 --epochs 5 --lr 1e-05 --hidden_layer 5 --seed 0 --dropout 0.2 --partial_train_data ./transformed_datasets/german/Train/partial_train_5k.pkl --partial_val_data ./transformed_datasets/german/Train/partial_valid_5k.pkl --full_test_data ./transformed_datasets/german/full/train_test_valid.pkl --quad_reg 0.0 --sample_size 128 --batch_size 64 --soft_train 1 --index 1 --allow_unfairness 1 --fairness_gap 1e-05 --mode spo --multi_groups 0 --entreg_decay 0.1 --evaluate_interval 2000 --output_tag spo_tests_2022-06-02_ --disparity_type disp0 --indicator_type square  --reward_type dcg  --log_dir runs/default_JK
