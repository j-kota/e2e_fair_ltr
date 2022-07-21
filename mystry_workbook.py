# %% codecell
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# %% codecell
csv_file = 'csv_orig_fultr_benchmarks2.csv'
# %% codecell
frame = pd.read_csv(csv_file)
# %% codecell
metrics =[ 'entropy_final', #'sum_loss_final',
       'rewards_final', 'fairness_loss_final', 'reward_variance_final',
       'train_ndcg_final', 'train_dcg_final', 'weight_final',
       'test_ndcg_final', 'test_dcg_final', 'test_rank_final',
       'test_group_expos_disp_final', 'test_group_asym_disp_final' ]

# %% codecell
frame['fairness_loss_final']#.apply(lambda x: float(x.strip('][').split(', ')[-1]))
# %% codecell
frame.columns
# %% codecell
for col in metrics:
    frame[col] = frame[col].apply(lambda x: float(x.strip('][').split(', ')[-1]) if x!='[]' else 1000)
# %% codecell
frame['entreg_decay'] = frame['index'].apply(lambda x: int(x)%2==0)
# %% codecell
plt_metrics = ['entropy_final',   'rewards_final',   'fairness_loss_final',    'reward_variance_final',   'train_ndcg_final']
# %% codecell
ax=sns.boxplot(x='lr', y = 'train_ndcg_final',       #y='No-Overlap Violation as (%) of Average Proc. Time',   #'Makespan Error (%)',  #, #,  'constr_viol_avdegree_nv_V_avdur'  "makespan_recon_dist_V_makesp"


                   data = frame,

                   hue = 'decay',

                   #hue_order = [False,True],

                   #order=['FC','JM']

                   )


fig = ax.get_figure()
bottom, top   = ax.get_ylim()
#ax.set_ylim(0.0 , 30)
#ax.set_ylim(bottom , 0.0)
fig.savefig("train_ndcg_final.pdf")


# %% codecell
ax=sns.boxplot(x='lr', y = 'fairness_loss_final',       #y='No-Overlap Violation as (%) of Average Proc. Time',   #'Makespan Error (%)',  #, #,  'constr_viol_avdegree_nv_V_avdur'  "makespan_recon_dist_V_makesp"


                   data = frame,

                   hue = 'decay',

                   #hue_order = [False,True],

                   #order=['FC','JM']

                   )


fig = ax.get_figure()
bottom, top   = ax.get_ylim()
ax.set_ylim(0.0 , 0.001)
#ax.set_ylim(bottom , 0.0)
fig.savefig("fairness_loss_final.pdf")

# %% codecell
