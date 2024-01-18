
from cleanlab.pruning import get_noise_indices
import numpy as np
import pandas as pd

threshold = 0.9078
#This is a variable you need to control


def get_noise(label,psx):
    count=0
    sum=0
    for i in range(len(label)):
        if label[i]==0:
            count+=1
            sum+=psx[i][0]
    avg=sum/count
    print(avg)
    ordered=[]
    for i in range(len(label)):
        if psx[i][0] < threshold and label[i]==0:
            ordered.append(i)
    return ordered


def getPsx():
    df=pd.read_csv("probas.csv",header=None)
    return df.to_numpy()
if __name__ == '__main__':
    #label,urls = np.array(get_data())
    label=np.loadtxt('label.txt',dtype=int)

    psx = getPsx().reshape(-1, 2)
    print(len(label))
    ordered = get_noise(label=label,psx=psx)
    print(len(ordered))
    print(ordered)
    np.savetxt('index.txt',ordered,fmt="%d")