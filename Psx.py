import ast
import numpy as np
import pandas as pd

# 读取CSV文件
df = pd.read_csv('probas.csv',header=None)

proba_list=[]
for  data in np.array(df):
    data=data.astype(str)
    #print(data)
    data=data[0][8:-2].replace(",\n        ", "*")
    data=data.replace(","," ")
    #print(data)
    array_strings=data.split("*")
    #print(array_strings)
    for array_str in array_strings:
        elements = array_str.strip("[]").split()
        row = [float(element) for element in elements]
        proba_list.append(row)
   
print(len(proba_list))
df=pd.DataFrame(proba_list)
# 将结果保存回CSV文件
df.to_csv('probas.csv',header=None, index=False)