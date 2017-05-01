import numpy as np
import csv


def load_vowel_data(filename):
    dict_class ={ 'hid':0, 'hId':1, 'hEd':2, 'hAd':3, 'hYd':4, 'had':5, 'hOd':6, 'hod':7, 'hUd':8, 'hud':9, 'hed':10 }
    raw_data = open(filename)
    reader = csv.reader(raw_data, delimiter=' ', quoting=csv.QUOTE_NONE)
    all_data = list(reader)
    S_arr = np.array(all_data)
    row = len(S_arr)
    data = []
    target = []
    for i in xrange(row) :
        tmp = filter(None,S_arr[i])
        data.append(map(float,tmp[3:-1]))
    for i in xrange(row) :
        target.append(int(S_arr[i][-1]))
    data = np.array(data)
    target  = np.array(target)
    return [data,target,dict_class]


def load_zoo_data(filename):
    raw_data = open(filename)
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    all_data = list(reader)
    S_arr = np.array(all_data)
    length = len(S_arr[0])
    data = S_arr[:, 1:length - 1]
    data = data.astype(int)
    target = S_arr[:, length - 1]
    target = target.astype(int)

    classes={ i:i+1 for i in xrange(len(np.unique(target))) }

    for i in xrange(len(target)):
        target[i] = target[i]-1
        
    return [data,target,classes]

def load_yeast_data(filename):
    dict_class ={ 'ERL':0 ,'CYT':1, 'NUC':2, 'MIT':3, 'ME3':4, 'ME2':5, 'ME1':6, 'EXC':7, 'VAC':8, 'POX':9}
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
#load_zoo_data('zoo.data')

def load_glass_data(filename):
    raw_data = open(filename)
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    all_data = list(reader)
    S_arr = np.array(all_data)
    length = len(S_arr[0])
    data = S_arr[:, 1:length - 1]
    data = data.astype(float)
    target = S_arr[:, length - 1]
    target = target.astype(int)

    classes={ i:i+1 for i in xrange(len(np.unique(target))) }

    for i in xrange(len(target)):
        target[i] = target[i]-1
        
    return [data,target,classes]

def load_glass_data_backup(filename):
    dict_class ={ '7':0, '1':1, '2':2, '3':3, '5':4, '6':5 }
    raw_data = open(filename)
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    all_data = list(reader)
    S_arr = np.array(all_data)
    row = len(S_arr)
    data = []
    target = []
    for i in xrange(row) :
        tmp = filter(None,S_arr[i])
        data.append(map(float,tmp[1:-1]))
    for i in xrange(row) :
        target.append(int(S_arr[i][-1])-1)
    data = np.array(data)
    target  = np.array(target)
    return [data,target,dict_class]

def load_vehicle_data(filename):
    dict_class ={ 'opel':0, 'saab':1, 'bus':2, 'van':3 }
    raw_data = open(filename)
    reader = csv.reader(raw_data, delimiter=' ', quoting=csv.QUOTE_NONE)
    all_data = list(reader)
    S_arr = np.array(all_data)
    row = len(S_arr)
    data = []
    target = []
    for i in xrange(row) :
        tmp = filter(None,S_arr[i])
        data.append(map(float,tmp[:-1]))
    #print len(data[0])
    for i in xrange(row) :
        if S_arr[i][-1] == '' :
            x = -2
        else :
            x = -1
        target.append(dict_class[S_arr[i][x]])
    data = np.array(data)
    target  = np.array(target)
    return [data,target,dict_class]