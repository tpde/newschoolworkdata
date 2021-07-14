from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from sklearn.metrics.classification import confusion_matrix
import numpy as np
from math import sqrt
from .util import get_MCC

# from skimage import filters
# def thresh_otsu(pred, mask, flatten=True):
#     for i in range(pred.shape[0]):
#         pred
#     threshold=filters.threshold_otsu(pred_vessels[mask==1])
#     pred_vessels_bin=np.zeros(pred_vessels.shape)
#     pred_vessels_bin[pred_vessels>=threshold]=1
#     
#     if flatten:
#         return pred_vessels_bin[masks==1].flatten()
#     else:
#         return pred_vessels_bin



def get_curve(arr_gros, arr_pres, arr_mask=None):
    if isinstance(arr_mask, np.ndarray):
        fpr, tpr, _ = roc_curve(arr_gros[arr_mask==1].flatten(), arr_pres[arr_mask==1].flatten())
        list_prec, list_recall, _ = precision_recall_curve(arr_gros[arr_mask==1].flatten(), arr_pres[arr_mask==1].flatten())
    else:
        fpr, tpr, _ = roc_curve(arr_gros.flatten(), arr_pres.flatten())
        list_prec, list_recall, _ = precision_recall_curve(arr_gros.flatten(), arr_pres.flatten())
    
    return fpr, tpr, list_prec, list_recall
    
 
def thresh_dice(arr_gros, arr_pres, arr_mask=None):
    if isinstance(arr_mask, np.ndarray):
        precision, recall, thresholds = precision_recall_curve(arr_gros[arr_mask==1].flatten(), arr_pres[arr_mask==1].flatten(),  pos_label=1)
    else:
        precision, recall, thresholds = precision_recall_curve(arr_gros.flatten(), arr_pres.flatten(),  pos_label=1)
    best_f1=-1
    for idx in range(len(precision)):
        curr_f1=2.*precision[idx]*recall[idx]/(precision[idx]+recall[idx])
        if best_f1<curr_f1:
            best_f1=curr_f1
            best_thresh=thresholds[idx]

    pres_bw = np.zeros(arr_pres.shape)
    pres_bw[arr_pres>=best_thresh] = 1
    
    if isinstance(arr_mask, np.ndarray):
        cm=confusion_matrix(arr_gros[arr_mask==1].flatten(), pres_bw[arr_mask==1].flatten())
    else:
        cm=confusion_matrix(arr_gros.flatten(), pres_bw.flatten())
    # TP:cm[1,1]    TN:cm[0,0]    FP:cm[0,1]    FN:cm[1,0]
    TP, TN = cm[1,1], cm[0,0]
    FP, FN = cm[0,1], cm[1,0]
    acc=1.*(TN+TP)/np.sum(cm)
    spec=1.*TN/(TN+FP)
    sens=1.*TP/(FN+TP)
    prec=1.*TP/(TP+FP)
    dice=2.*TP/(2.*TP+FP+FN)
    MCC = get_MCC(TP, FP, FN, TN)
    return acc, spec, sens, prec, dice, MCC, best_f1, best_thresh, pres_bw

def matthews(arr_gros, pres_bw):
    assert(len(np.unique(pres_bw))==2)
    cm=confusion_matrix(arr_gros.flatten(), pres_bw.flatten())
    TP, TN = cm[1,1], cm[0,0]
    FP, FN = cm[0,1], cm[1,0]
    div = sqrt(TP+FP)*sqrt(TP+FN)*sqrt(TN+FP)*sqrt(TN+FN)
    MCC = ( TP*TN - FP*FN )/div
    return MCC

def auc_area(points_x, points_y):
#     s_area = 0.0
#     for i in range( len(points_x)-1 ):
#         s_area = s_area + (points_x[i+1]-points_x[i])*(points_y[i]+points_y[i+1])/2.0
#     return abs(s_area)
    points_xy = zip(points_x,points_y)
    points_xy = sorted(points_xy, key=lambda item:item[0])
    points_x, points_y = zip(*points_xy)
    return auc(points_x, points_y)
    