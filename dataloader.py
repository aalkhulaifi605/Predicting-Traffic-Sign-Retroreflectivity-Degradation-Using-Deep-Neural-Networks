import numpy as np
import pandas as pd
def get_whole_data():
    data = pd.read_csv('./data/Retreoreflectvity_data.csv',sep=',',header=0)
    data_np = data.to_numpy()
    return data

def get_whole_data_avg():
    data = pd.read_csv('./data/RF_averaged.csv',sep=',',header=0)
    columns_titles = ['Brand','Sheeting Type','Color','Orientation (Degrees)','Observation Angle (Degrees)',
    'Installation Year','RA-values(Cd/lx/m2)','Age (years)']
    data=data.reindex(columns=columns_titles)
    data_np = data.to_numpy()
    return data