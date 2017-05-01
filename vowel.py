import numpy as np
import csv

def load_vowel_data(filename):
    dict_class ={ 'hid':0.0, 'hId':1.0, 'hEd':2.0, 'hAd':3.0, 'hYd':4.0, 'had':5.0, 'hOd':6.0, 'hod':7.0, 'hUd':8.0, 'hud':9.0, 'hed':10.0 }
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
        target.append(float(S_arr[i][-1]))
    data = np.array(data)
    target  = np.array(target)
    return [data,target,dict_class]

x,y,dict1 = load_vowel_data('vowel-context.data')
print x
print y
print dict1