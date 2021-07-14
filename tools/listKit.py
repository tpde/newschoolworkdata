#! -*- coding:utf-8 -*-

def find_nearest(list_heights, value):
    list_distance = [abs(height-value) for height in list_heights]
    return list_heights[ list_distance.index(min(list_distance)) ]

def str2list(strs, is_int):
    """split string:"[ a, b, c ]" into list:[a,b,c]
    
    # Arguments
        strs: string     exp: "[ a, b, c ]"

    # Returns
        list_data: [('a',3), ('b',1), ('c',2)]
    
    """
    strs = strs.strip('[')
    strs = strs.strip(']')
    list_src = strs.split(',')
    list_dst = [s.strip(' ') for s in list_src]
    if list_dst == [''] or strs == 'None':
        list_dst = None
    else:
        if is_int:
            list_dst = [int(s) for s in list_dst]
    return list_dst

def str2dl(strlist):
    dl = strlist.replace('\n','').replace('[','').replace(']','').split(' ')
    dl = [float(v) for v in dl if v]
    assert(len(dl)==11)
    return dl

def unique_list(seq, excludes=[]):   
    """   
    返回包含原列表中所有元素的新列表，将重复元素去掉，并保持元素原有次序  
    excludes: 不希望出现在新列表中的元素们  
    """  
    seen = set(excludes)  # seen是曾经出现的元素集合   
    return [x for x in seq if x not in seen and not seen.add(x)]


def extractIdxElement(list_data, list_idx):
    list_dst = list()
    for i in range(len(list_idx)):
        list_dst.append( list() )
    for e in range(len(list_data)):
        data = list_data[e]
        for idx in range(len(list_idx)):
            list_dst[idx].append(data[list_idx[idx]])
    return list_dst

def combine2list( list_x, list_y ):
    # list_x = [[0.1,120], ...]
    # list_y = [[0.1,120], ...]
    # list_dst = [[0.1,120,0.1,120], ...]
    assert(len(list_x)==len(list_y))
    list_dst = [list() for data in list_x]
    for idx in range(len(list_x)):
        list_dst[idx].extend(list_x[idx]) if type(list_x[idx])==type(list()) else list_dst[idx].extend([list_x[idx]])
        list_dst[idx].extend(list_y[idx]) if type(list_y[idx])==type(list()) else list_dst[idx].extend([list_y[idx]])
    return list_dst

def split2list(list_dst):
    list_x = list()
    list_y = list()
    for idx in range(len(list_dst)):
        data = list_dst[idx]
        list_x.append( data[0] )
        list_y.append( data[1] )
    return list_x, list_y


def sortlist(list_src, sortidx, reverse=False):
    list_dst = sorted(list_src, key=lambda src:src[sortidx], reverse=reverse)
    return list_dst

