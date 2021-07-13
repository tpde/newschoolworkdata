#! -*- coding:utf-8 -*-
import os
from ..basic import mkdir

def readlist(filepath):
    list_lines = []
    f = open(filepath)
    line = f.readline()
    while line:
        list_lines.append(line.rstrip('\n'))
        line = f.readline()
    f.close()
    return list_lines

def savelist(list_data, savepath, mode='w'):
    f = open(savepath, mode)
    strx = ''
    for idx in range(len(list_data)):
        strx += str(list_data[idx])+'\n'
    strx = strx.rstrip('\n')
    f.write(strx)
    f.close()


def readdict(filepath):
    list_lines = readlist(filepath)
    list_data = [line.split('\t') for line in list_lines]
    dict_data = dict()
    for data in list_data:
        dict_data[data[0]] = data[1]
        
        
def savedict(dict_data, savepath):
    f = open(savepath, 'w')
    strx = ''
    for key,value in dict_data.items():
        strx += str(key)+'\t'+str(value)+'\n'
    strx = strx.rstrip('\n')
    f.write(strx)
    f.close()



def readlistlist(filepath, datatype, splitchar='\t'):
    list_lines = readlist(filepath)
    list_datas = [line.split(splitchar) for line in list_lines]
    list_list = [list() for line in list_lines]
    for row in range(len(list_datas)):
        data = list_datas[row]
        for col in range(len(data)):
            if datatype == 'float':
                list_list[row].append( float(data[col]) )
            else:
                list_list[row].append( data[col] )
    return list_list

def savelistlist(list_list_data, savepath):
    f = open(savepath, 'w')
    strx = ''
    for idx in range(len(list_list_data)):
        list_data = list_list_data[idx]
        for i in range(len(list_data)):
            strx += str(list_data[i])+'\t'
        strx = strx.rstrip('\t')
        strx += '\n'
    f.write(strx)
    f.close()


def readlistdict(filepath, saveorder, dtype='float'):
    # saveorder=['thresh','TP','TN','FP','FN']
    list_lines = readlist(filepath)
    list_datas = [line.split('\t') for line in list_lines]
    list_dict = [dict() for line in list_lines]
    for row in range(len(list_datas)):
        data = list_datas[row]
        for col in range(len(data)):
            if dtype == 'float':
                list_dict[row][saveorder[col]] = float(data[col])
            else:
                list_dict[row][saveorder[col]] = data[col]
    return list_dict


def savelistdict(list_dict_data, savepath, saveorder=None):
    # saveorder=['thresh','TP','TN','FP','FN']
    if not os.path.exists(os.path.split(savepath)[0]):
        mkdir(os.path.split(savepath)[0])
    f = open(savepath, 'w')
    strx = ''
    for idx in range(len(list_dict_data)):
        dict_data = list_dict_data[idx]
        if saveorder == None:
            for key in dict_data.keys():
                strx += str(key) +':'+str(dict_data[key])+'\t'
        else:
            for i in range(len(saveorder)):
                strx += str(dict_data[saveorder[i]])+'\t'
            
        strx = strx.rstrip('\t')
        strx += '\n'
    f.write(strx)
    f.close()



    