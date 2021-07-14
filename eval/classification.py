from keras.utils.np_utils import to_categorical
import numpy as np
from .util import get_MCC

def eval_cls( arr_pred, arr_label, n_class, list_thresh):
    if len(arr_label.shape)==1:
        arr_label = to_categorical(arr_label, n_class)
    assert arr_pred.shape[0] == arr_label.shape[0]
    assert len(arr_pred.shape)>1
    
    list_TFPN = list()
    list_parameter = list()
    for threshold in list_thresh:
        arr_pred_posi = (arr_pred[:,0]>threshold).astype(np.bool)
        arr_pred_nege = (arr_pred[:,0]<=threshold).astype(np.bool)
        
        arr_TP = (arr_pred_posi*arr_label[:,0]).astype(np.int)
        arr_TN = ( arr_pred_nege*(1-arr_label[:,0]) ).astype(np.int)
        arr_FP = (arr_pred_posi*(1-arr_label[:,0])).astype(np.int)
        arr_FN = (arr_pred_nege*arr_label[:,0]).astype(np.int)
        
        assert np.unique( (arr_TP+arr_TN+arr_FP+arr_FN).astype(np.int) ) == [1]
        
        TP = np.sum(arr_TP)
        TN = np.sum(arr_TN)
        FP = np.sum(arr_FP)
        FN = np.sum(arr_FN)
        list_TFPN.append( [threshold,TP,TN,FP,FN] )
        
        # eval_result[i] = segmentation_area.eval_area_extend( pre, gro, 350, 0)
        accuracy = (TP+TN+ 1E-30) / (TP+FP+FN+TN + 1E-30)
        specificity = (TN+ 1E-30) / (TN+FP + 1E-30)
        sensitivity = (TP+ 1E-30) / (TP+FN + 1E-30)
        precision = (TP+ 1E-30) / (TP+FP + 1E-30)
        dice = (2*TP +1E-30) / ( 2*TP+FP+FN +1E-30)
        list_parameter.append( [threshold,accuracy,specificity,sensitivity,precision,dice] )
    return list_TFPN, list_parameter



def eval_arteriovenous(arr_groV, arr_groA, arr_preV, arr_preA):
    S1 = np.sum(  arr_groA*arr_groV*(1-arr_preA)*arr_preV       )
    S2 = np.sum(  arr_groA*(1-arr_groV)*(1-arr_preA)*arr_preV   )
    S3 = np.sum(  arr_groA*arr_groV*arr_preA*(1-arr_preV)       )
    S4 = np.sum(  arr_groA*arr_groV*arr_preA*arr_preV           )
    S5 = np.sum(  arr_groA*(1-arr_groV)*arr_preA*arr_preV       )
    S6 = np.sum(  arr_groA*(1-arr_groV)*arr_preA*(1-arr_preV)   )
    S7 = np.sum(  (1-arr_groA)*arr_groV*arr_preA*(1-arr_preV)   )
    S8 = np.sum(  (1-arr_groA)*arr_groV*arr_preA*arr_preV       )
    S9 = np.sum(  (1-arr_groA)*arr_groV*(1-arr_preA)*arr_preV   )
    S = S1+S2+S3+S4+S5+S6+S7+S8+S9
    cross = ((arr_groA>0)|(arr_groV>0)) & ((arr_preA>0)|(arr_preV>0))
    assert( np.sum(cross.astype(np.int))==S )
    
    TPv, FPv, FNv, TNv = S1+S4+S8+S9, S2, S7, S6+S5+S3
    TPa, FPa, FNa, TNa = S3+S4+S5+S6, S7, S2, S9+S1+S8 
    
    MISCv = FPa/(TPv + FPa)
    MISCa = FPv/(TPa + FPv)
    ACC = 1 - (S2+S7)/(S1+S2+S3+S4+S5+S6+S7+S8+S9)
         
    MCCv = get_MCC(TPv, FPv, FNv, TNv)
    MCCa = get_MCC(TPa, FPa, FNa, TNa)
    
    MISCv0 = S7/(S9 + S7)
    MISCa0 = S2/(S6 + S2)
    ACC0 = 1 - (S2+S7)/(S2+S7+S6+S9)

    TPv, FPv, FNv, TNv = S9, S2, S7, S6   
    MCCv0 = get_MCC(TPv, FPv, FNv, TNv)
    
    TPa, FPa, FNa, TNa = S6, S7, S2, S9
    MCCa0 = get_MCC(TPa, FPa, FNa, TNa)
    return {'MISCv':MISCv, 'MISCa':MISCa, 
            'ACC':ACC, 'MCCv':MCCv, 'MCCa':MCCa,
            'MISCv0':MISCv0, 'MISCa0':MISCa0, 
            'ACC0':ACC0, 'MCCv0':MCCv0, 'MCCa0':MCCa0}


