import numpy as np
import csv

def load_yeast_data(filename):
    dict_class ={ 'CYT':1, 'NUC':2, 'MIT':3, 'ME3':4, 'ME2':5, 'ME1':6, 'EXC':7, 'VAC':8, 'POX':9, 'ERL':10 }
    raw_data = open(filename)
    reader = csv.reader(raw_data, delimiter=' ', quoting=csv.QUOTE_NONE)
    all_data = list(reader)
    S_arr = np.array(all_data)
    row = len(S_arr)
    data = []
    target = []
    for i in xrange(row) :
        tmp = filter(None,S_arr[i])
        data.append(map(float,tmp[1:-1]))
    for i in xrange(row) :
        target.append(dict_class[S_arr[i][-1]])
    data = np.array(data)
    target  = np.array(target)
    return [data,target,dict_class]

load_yeast_data('yeast.data')