#! -*- coding:utf-8 -*-
import os
from ..basic import mprint

def mprintdict(dict_data):
    for key,data in dict_data.items():
        mprint(key,data)

def dict_keyOffset(dict_src, offset):
    dict_dst = dict()
    for key, value in dict_src.items():
        dict_dst[key+offset] = value
    return dict_dst

def dict_keystandard(dict_src, keylen, prefix, tail, offset=0):
    rightkey = lambda keylen,index: '%02d'%(index) if keylen==2     \
                            else '%03d'%(index) if keylen==3        \
                            else '%04d'%(index) if keylen==4        \
                            else '%05d'
    dict_dst = dict()
    for key,value in dict_src.items():
        newkey = prefix + rightkey(keylen,int(key+offset)) + tail
        dict_dst[newkey] = value
    return dict_dst


def dict2list( dict_data, startbit, orderlen):
    """because when you transvert img from dict to list, the information contained in key will loss
        so we must Sort them according to the existing order feature,
        that's why we use parameter of starbit and orderlen
    """
    list_data = sorted(dict_data.iteritems(), key=lambda item:int(item[0][startbit:startbit+orderlen]), reverse=False)
    for i in range(len(list_data)):
        list_data[i] = list_data[i][1]
    return list_data

def delkeytail(dict_src):
    # >>>os.path.splitext('a/b/c/d.e.png')
    # ('a/b/c/d.e', '.png')
    dict_dst = dict()
    for key,value in dict_src.items():
        key_ = os.path.splitext(key)[0]
        dict_dst[key_] = value
    return dict_dst
        

def completeMisskey(dict_dict_data):
    dict_listkey = {key:list(dict_data.keys()) for key,dict_data in dict_dict_data.items()}
    uniquekey = set([])
    for listkey in dict_listkey.values():
        [uniquekey.add(key) for key in listkey]

    for key,dict_data in dict_dict_data.items():
        listkey = dict_listkey[key]
        misskey = [k for k in uniquekey if k not in listkey]
        for k in misskey:
            dict_data[k] = 0
        dict_dict_data[key] = dict_data
    return dict_dict_data