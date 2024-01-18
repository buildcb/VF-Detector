import numpy as np
import pandas as pd
import random
dataset_name = 'ase_dataset_sept_19_2021.csv'

urls = np.loadtxt('url.txt', dtype=str)
label=np.loadtxt('label.txt',dtype=int)
er_index=[]
def calcError():
    for i in range(len(label)):
        if label[i] == 0:
            er_index.append(i)
    np.savetxt('er_index.txt', er_index, fmt="%d")
def getError():
    return  np.loadtxt('er_index.txt',dtype=int)
calcError()
er_index=getError()
percentage=0.80
num=int(len(er_index) * percentage)
er_index = np.array(random.sample(list(er_index), num))

urls=np.take(urls,er_index)



df=pd.read_csv(dataset_name)
header=df.columns
items=df.to_numpy().tolist()
print(len(items))
re_item=[]
sum=0
for item in items:
    commit_id = item[0]
    repo = item[1]
    url = repo + '/commit/' + commit_id
    label=item[3]
    if url in urls:
        sum+=1
    else :
        re_item.append(item)
print(len(re_item))

print(sum)
df=pd.DataFrame(re_item)
assert False
df.to_csv('ase_dataset_sept_19_2021_2.csv',header=header,index=False)