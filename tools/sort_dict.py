#! -*- coding:utf-8 -*-

def sort_useorder(dict_src, startbit, orderlen, offset=0, order_char=None, is_getkey=False):
    """sort dict_data use the order feature of:
                int( key[startbit:startbit+orderlen] )

    # Arguments
        startbit: key order start bit
        orderlen: key order length

    # Returns
        list_data: [('a',3), ('b',1), ('c',2)]
    """
    if order_char==None:
        list_data = sorted(dict_src.items(), key=lambda item:int(item[0][startbit:startbit+orderlen]), reverse=False)
    else:
        list_data = sorted(dict_src.items(), key=lambda item:int(item[0][:item[0].find(order_char)]), reverse=False)
    dict_dst = dict()
    dict_key = dict()
    for i in range(len(list_data)):
        dict_dst[i+offset] = list_data[i][1]
        dict_key[i+offset] = list_data[i][0]
    if is_getkey:
        return dict_dst, dict_key
    else:
        return dict_dst

def sortdict_usechar(dict_src, char_start, char_end, offset):
    list_data = sorted(dict_src.items(), key=lambda item:int(item[0][item[0].find(char_start)+1:item[0].find(char_end)]), reverse=False)
    dict_dst = dict()
    for i in range(len(list_data)):
        dict_dst[i+offset] = list_data[i][1]
    return dict_dst

def sortdict_usetail(dict_src, startbit, orderlen, offset, order_tails, order_char=None):
    if order_char==None:
        list_data = sorted(dict_src.items(), key=lambda item:int(item[0][startbit:startbit+orderlen]), reverse=False)
    else:
        list_data = sorted(dict_src.items(), key=lambda item:int(item[0][:item[0].find(order_char)]), reverse=False)
    dict_data = dict()
    for i in range(len(list_data)):
        file = list_data[i][0]
        if order_char==None:
            index = int(file[startbit:startbit+orderlen])*len(order_tails)+order_tails.index( file[startbit+orderlen:file.find('.')] )
        else:
            print(file)
            index = int(file[:file.find(order_char)])*len(order_tails)+order_tails.index(file[file.find(order_char):file.find('.')])
        dict_data[index] = list_data[i][1]
    list_data = sorted(dict_data.iteritems(), key=lambda item:item[0], reverse=False)
    dict_dst = dict()
    for i in range(len(list_data)):
        dict_dst[i+offset] = list_data[i][1]
    return dict_dst


    
def sort_disorder( list_src, offset=0):
    """
    [dict_src1, dict_src2, ...]
    """
    list_dst = list()
    for idx in range(len(list_src)):
        list_dst.append(dict())

    for key in list_src[0]:
        for idx in range(len(list_src)):
            if key in list_src[idx]:
                list_dst[idx][offset] = list_src[idx][key]
            else:
                print('sort_Simultaneously:: dict_src2:%s missing'%key)
                raise(IOError)
        offset += 1
    return list_dst

def sortlist(list_data, idx):
    # keys = ['two', 'three', 'one']
    # values = [2, 3, 1]
    # data = ['b', 'c', 'a']
    # c = zip(keys, values, data)
    # d = sorted(c, key=lambda x: x[1])
    # order = zip(*d)
    # print(list(order))
    c = zip(*list_data)
    d = sorted(c, key=lambda x: x[idx])
    order = zip(*d)
    return list(order)
