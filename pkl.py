import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_pickle("../../LSWMD.pkl")

df = df.drop(['waferIndex'], axis = 1)

def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1
df['waferMapDim']=df.waferMap.apply(find_dim)

df['failureNum']=df.failureType
df['trainTestNum']=df.trianTestLabel
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
mapping_traintest={'Training':0,'Test':1}
df=df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})


df_withlabel = df[(df['failureNum']>=0) & (df['failureNum']<=8)]
df_withlabel =df_withlabel.reset_index()
df_withpattern = df[(df['failureNum']>=0) & (df['failureNum']<=7)]
df_withpattern = df_withpattern.reset_index()
df_nonpattern = df[(df['failureNum']==8)]
df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]

df_withpattern.to_pickle("WM-811K.pkl")

df = pd.read_pickle("WM-811K.pkl")

df["oldFailureNum"] = df["failureNum"]
for index, row in df.iterrows():
    map = df["waferMap"][index]
    wafer_area = 0
    defect_area = 0
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i][j] == 1:
                wafer_area += 1
            elif map[i][j] == 2:
                wafer_area += 1
                defect_area += 1
    
    # print(f"{index}")
    if defect_area/wafer_area < 0.05:
        df.loc[index, "failureNum"] = df.loc[index, "oldFailureNum"] + 8
    # else:
    #     df.loc[index, "failureNum"] = df.loc[index, "oldFailureNum"] + 8

df.to_pickle("data.pkl")