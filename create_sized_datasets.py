import numpy as np
import pickle


#path = './transformed_datasets/german/Train/'
path = './transformed_datasets/mslr/Train/'


#a = pickle.load(open(path+'partial_train_250k.pkl','rb'))
a = pickle.load(open(path+'partial_train_359k.pkl','rb'))
len(a)
len(a[0])
len(a[1])
type(a[1])

y = [1,2,3,4,5]
z = ['a','b','c','d','e']
x = zip(z,y)



X = np.stack(a[0])

Y = np.stack(a[1])


X[0]
a[0][0]



N_ = np.array(range(len(X)))
inds_100k = np.random.choice(N_,100000)
inds_50k  = np.random.choice(inds_100k,50000)
inds_25k  = np.random.choice(inds_100k,25000)
inds_5k   = np.random.choice(inds_100k,5000)

Xout_100k = X[inds_100k]
Xout_50k = X[inds_50k]
Xout_25k  = X[inds_25k]
Xout_5k   = X[inds_5k]

Yout_100k = Y[inds_100k]
Yout_50k = Y[inds_50k]
Yout_25k  = Y[inds_25k]
Yout_5k   = Y[inds_5k]
print('Before 100k')
out_100k = (  [ x for x in Xout_100k ]  ,  [ y for y in Yout_100k ]   )
print('done with 100k')
out_50k  = (  [ x for x in Xout_50k  ]  ,  [ y for y in Yout_50k  ]   )
out_25k  = (  [ x for x in Xout_25k  ]  ,  [ y for y in Yout_25k  ]   )
out_5k   = (  [ x for x in Xout_5k   ]  ,  [ y for y in Yout_5k   ]   )



pickle.dump(out_100k,open(path + 'partial_train_JK_100k.pkl','wb'))
pickle.dump(out_50k, open(path + 'partial_train_JK_50k.pkl','wb'))
pickle.dump(out_25k, open(path + 'partial_train_JK_25k.pkl','wb'))
pickle.dump(out_5k,  open(path + 'partial_train_JK_5k.pkl','wb'))
print('Done')
