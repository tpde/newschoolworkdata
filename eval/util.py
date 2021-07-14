#! -*- coding:utf-8 -*-
import numpy as np
from math import sqrt
from ..io import readlistdict

def AnalysisTFPN(filepath, order, best='dice'):
    list_dict = readlistdict(filepath, order)
    list_paras = []
    list_roc = []
    for i in range(len(list_dict)):
        dict_TFPN = list_dict[i]
        threshold,TP,TN,FP,FN = dict_TFPN['thresh'],dict_TFPN['TP'],dict_TFPN['TN'],dict_TFPN['FP'],dict_TFPN['FN']
        accuracy = (TP+TN+1E-30) / (TP+FP+FN+TN+1E-20)
        specificity = (TN+1E-30) / (TN+FP+1E-20)
        sensitivity = (TP+1E-30) / (TP+FN+1E-20)
        precision = (TP+1E-30) / (TP+FP+1E-20)
        dice = (2.*TP+1E-30) / (2.*TP+FP+FN+1E-20)
        score = (TP+1E-30) / (TP+FP+FN+1E-20)
        TPR = (TP+1E-30)/(TP+FN+1E-20)
        FPR = (FP+1E-30)/(FP+TN+1E-20)
#         if threshold==0:
#             TPR = 0
#             precision = 0
        list_paras.append( {'thresh':threshold, 'sens':sensitivity, 'prec':precision,
                            'acc':accuracy, 'spec':specificity, 'dice':dice, 'score':score} )
        list_roc.append( [threshold, FPR, TPR] )

    list_best = [para[best] for para in list_paras]
    best_idx = list_best.index( max(list_best) )
    return list_paras, list_roc, list_paras[best_idx]
     

def TFPN2param(TFPN):
    """
    TFPN 是下列情况的任一种都可以计算
    {'TP':0, 'FP':0, 'FN':0, 'TN':0}
    {'001.png':{'TP':0, 'FP':0, 'FN':0, 'TN':0}, ...}
    """
    def calparam(dict_TFPN):
        dict_param = dict()
        dict_param['accu'] = (dict_TFPN['TP']+dict_TFPN['TN']+1E-30) / (dict_TFPN['TP']+dict_TFPN['FP']+dict_TFPN['FN']+dict_TFPN['TN']+1E-30)
        dict_param['spec'] = (dict_TFPN['TN']+1E-30) / (dict_TFPN['TN']+dict_TFPN['FP']+1E-30)
        dict_param['sens'] = (dict_TFPN['TP']+1E-30) / (dict_TFPN['TP']+dict_TFPN['FN']+1E-30)
        dict_param['prec'] = (dict_TFPN['TP']+1E-30) / (dict_TFPN['TP']+dict_TFPN['FP']+1E-30)
        dict_param['dice'] = (2*dict_TFPN['TP']+1E-30) / (2*dict_TFPN['TP']+dict_TFPN['FP']+dict_TFPN['FN']+1E-30)
        return dict_param
    
    list_key = list(TFPN.keys())
    if type(TFPN[list_key[0]])==type(dict()):
        param = dict()
        for key in list_key:
            param[key] = calparam(TFPN[key])
        return param
    else:
        return calparam(TFPN)


def metrix_synthesis(gros_trunk, gros_treetops, arr_pres):
    # 末梢上正确预测的点 / 非主干上预测为正样本的点
    correct = np.sum(arr_pres*gros_treetops)
    pNoTrunk = (np.sum(arr_pres*(1-gros_trunk)) + 1E-30)
    return correct, pNoTrunk, correct/pNoTrunk
    
def metrix_compare(gros_trunk, gros_treetops, arr_pre1, arr_pre2):
    # pre2 在末梢上比 pre1 多预测的点
    justify = np.sum(arr_pre2 * gros_treetops) - np.sum(arr_pre1 * gros_treetops)
    # pre1 在正样本上没有预测到的点：FP
    FP = np.sum(  (gros_trunk+gros_treetops)*(1-arr_pre1) )
    return justify / (FP + 1E-30)


def get_MCC(TP, FP, FN, TN):
    div = sqrt(TP+FP)*sqrt(TP+FN)*sqrt(TN+FP)*sqrt(TN+FN)
    MCC = ( TP*TN - FP*FN )/div
    return MCC