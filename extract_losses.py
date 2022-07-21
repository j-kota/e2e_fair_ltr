# Extract loss values from log file

import argparse
import string
import re
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--log_filename", type=str, default="none")
args = parser.parse_args()

file = open(args.log_filename,'rb')
lines = file.readlines()
losses = []

for line in lines[:-21]:
    line = str(line)
    if 'loss = ' in line:

        result = re.sub('[^\d]+[^0-9]','', line)
        line = result

        losses.append( float(line) )
        print( line )

num_steps = 150 #200 #cutoff // 100
cutoff = (len(losses) // num_steps) * num_steps
print("cutoff = ")
print( cutoff  )

rwd = np.array(losses)
rwd = rwd[:cutoff]
rwd = rwd.reshape(num_steps,-1)
rwd_ep = rwd.mean(1)


plt.plot(range(len(rwd_ep)),rwd_ep)
#plt.savefig('png/'+args.log_filename.split('.')[0].split('/')[1]+'_plt.png')
plt.show()
