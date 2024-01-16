import numpy as np
import pandas as pd
dataset_name = 'F:\\多粒度漏洞检测器数据集\\ase_dataset_sept_19_2021_4.csv'
index=np.loadtxt('index.txt',dtype=int)
urls = np.loadtxt('url.txt', dtype=str)
print(len(urls))
urls=np.take(urls,index)
print(len(urls))

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
df.to_csv('ase_dataset_sept_19_2021_4.csv',header=header,index=False)
