# JK
# Testing scrap book
# Initially for broken backprop due to inplace options
import torch
import pickle


psave = pickle.load(open('fultr_dbg.p','rb'))
len(psave)




(masks,prob_mat, batch_size) = psave

j = 0

# goal
probs = torch.stack( [masks[k]*prob_mat[:,j,:][k] for k in range(batch_size)] ).clone()

prob_mat[:,j,:] # dist over items at position j, all batch elements (element 0 selected)
probs[:,j,:].size()


masks.size()

probs

masks*prob_mat[:,j,:].unsqueeze(1)
