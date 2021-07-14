#! -*- coding:utf-8 -*-
import os
import numpy as np
from ..basic import mkdir
from ..io import readexcel, save2excel, savelist
from ..imglib import write_images
from ..nparray import arrs2dict, stack_array, compare
from ..eval import AnalysisTFPN, thresh_dice, get_curve, auc_area, roc_auc_score
from ..tools import completeMisskey, extractIdxElement, combine2list, sortlist, split2list
from .graphic import plot_curves

def plot_evalresult(dict_TFPNfile, TFPN_order, savedir, best='dice'):
    # dict_file {'exp1':'exp1/TFPN.txt', 'exp2':'exp2/TFPN.txt'}
    # TFPN_order ['thresh','TP','TN','FP','FN']
    mkdir(savedir)
    def adjustcurve(list_xy):
        for i in range(len(list_xy)):
            xy = list_xy[i]
            if xy[0]<0.0000000000000009:
                xy[1] = 1
            list_xy[i] = xy
        return list_xy

    dict_pr = dict()
    dict_roc = dict()
    dict_xy = dict()
    dict_best = dict()
    for exp, filepath in dict_TFPNfile.items():
        dict_best[exp] = dict()
        
        list_paras, list_roc, best_para = AnalysisTFPN( filepath, TFPN_order, best)
        dict_best[exp] = best_para
        list_xy = [[paras['sens'],paras['prec']] for paras in list_paras]
        list_xy = sortlist(list_xy, 0)
        list_xy = adjustcurve(list_xy)
        list_x, list_y = split2list(list_xy)
        savelist(list_x, os.path.join(savedir,exp+'_prx.txt'))
        savelist(list_y, os.path.join(savedir,exp+'_pry.txt'))
        dict_pr[exp+' auc(%.4f)'%auc_area(list_x, list_y)] = {'x':list_x, 'y':list_y}
        dict_xy[exp+'_prx'] = list_x
        dict_xy[exp+'_pry'] = list_y
        dict_best[exp]['prauc'] = auc_area(list_x, list_y)
        
        points_FPR, poinst_TPR = extractIdxElement(list_roc, [1,2])
        list_xy = combine2list( points_FPR, poinst_TPR )
        list_xy = sortlist(list_xy, 0)
        list_x, list_y = split2list(list_xy)
        savelist(list_x, os.path.join(savedir,exp+'_rocx.txt'))
        savelist(list_y, os.path.join(savedir,exp+'_rocy.txt'))
        dict_roc[exp+' auc(%.4f)'%auc_area(list_x, list_y)] = {'x':list_x, 'y':list_y}
        dict_xy[exp+'_rocx'] = list_x
        dict_xy[exp+'_rocy'] = list_y
        dict_best[exp]['rocauc'] = auc_area(list_x, list_y)

    plot_curves( dict_pr, 'PR-curve', 'recall', 'preci', os.path.join(savedir,'curve_pr.png'), is_scatter=False)
    plot_curves( dict_roc, 'roc-curve', 'FPR', 'TPR', os.path.join(savedir,'curve_roc.png'), is_scatter=False)
    save2excel(dict_best, os.path.join(savedir,'eval.xls'))
    save2excel(dict_xy, os.path.join(savedir,'plot.xls'))

def plot_evalpixel(dict_gro, dict_pre, savedir, is_savebw=True, dict_msk=None):
    if dict_msk is None:
        list_imgarr, list_key = stack_array([dict_gro,dict_pre])
        acc, spec, sens, prec, dice, MCC, best_f1, best_thresh, pres_bw = thresh_dice(list_imgarr[0], list_imgarr[1], None)
    else:
        list_imgarr, list_key = stack_array([dict_gro,dict_pre, dict_msk])
        acc, spec, sens, prec, dice, MCC, best_f1, best_thresh, pres_bw = thresh_dice(list_imgarr[0], list_imgarr[1], list_imgarr[2])
    
    if is_savebw:
        write_images(arrs2dict(pres_bw, list_key), os.path.join(savedir,'binary'))
    dict_compare = compare( list_imgarr[0], pres_bw, list_key, dict_msk)
    write_images(dict_compare, os.path.join(savedir,'compare') )

    fpr, tpr, list_prec, list_recall = get_curve(list_imgarr[0], list_imgarr[1])
    s_roc = auc_area(fpr, tpr)
    s_pr = auc_area(list_prec, list_recall)
    
    AUC_ROC=roc_auc_score(list_imgarr[0].flatten(), list_imgarr[1].flatten())
    
    dict_result = {'acc':acc, 'spec':spec, 'sens':sens, 'prec':prec, 'dice':dice, 'MCC':MCC, 
                   'f1':best_f1, 's_roc':s_roc, 's_pr':s_pr, 'AUC_ROC':AUC_ROC, 'thresh':best_thresh}
    
    save2excel( dict_result, os.path.join(savedir,'evalpixel.xls'))
    
    plot_curves( {'PR auc(%.4f)'%auc_area(list_recall,list_prec) : {'x':list_recall,'y':list_prec}},
                 'PR-curve', 'recall', 'preci', os.path.join(savedir,'curve_pr.png'), is_scatter=False)
    
    plot_curves( {'roc auc(%.4f)'%auc_area(fpr,tpr) : {'x':fpr, 'y':tpr}},
                 'roc-curve', 'fpr', 'tpr', os.path.join(savedir,'curve_roc.png'), is_scatter=False)
    
    import random
    idx_pr = sorted(random.sample(range(list_recall.shape[0]), min(1000,list_recall.shape[0])))
    idx_roc = sorted(random.sample(range(fpr.shape[0]), min(1000,fpr.shape[0])))
    curve_xy = {'pr_x':list(list_recall[idx_pr]), 'pr_y':list(list_prec[idx_pr]),
                'roc_x':list(fpr[idx_roc]), 'roc_y':list(tpr[idx_roc])}
    save2excel(curve_xy, os.path.join(savedir,'curve_xy.xls'))

def eval_perimg(dict_gro, dict_pre, savedir):
    dict_perimg = dict()
    for key in dict_gro.keys():
        arr_gros = dict_gro[key][np.newaxis,...]
        arr_pres = dict_pre[key][np.newaxis,...]
        fpr, tpr, list_prec, list_recall = get_curve(arr_gros, arr_pres, None)
        acc, spec, sens, prec, f1_max, _, best_thresh = thresh_dice(arr_gros, arr_pres)
        roc_auc = roc_auc_score(arr_gros.flatten(), arr_pres.flatten())
        dict_perimg[key] = {'f1':f1_max, 'sens':sens, 'prec':prec, 'acc':acc, 'spec':spec, 'thresh':best_thresh,
                            'roc_auc':roc_auc,'pr_auc':auc_area(list_recall, list_prec)}
    save2excel(dict_perimg, os.path.join(savedir,'eval_perimg.xls'))



# def update_files(dict_imgdir, updatedir):
#     """
#     Args:
#         dict_dict_vessel = {'artery':read_images('newArtery'), 
#                             'vein':read_images('newVein') }
#         updatedir = 'version2'
#     """
#     for splitpart in os.listdir(updatedir):
#         for dbset in ['test', 'train', 'valid']:
#             for vessel in ['artery','vein']:
#                 list_file = os.listdir( os.path.join('version2',splitpart,dbset,vessel) )
#                 dict_vessel = {file:dict_dict_vessel[vessel][file] for file in list_file}
#                 write_images(dict_vessel, os.path.join('version2',splitpart,dbset,vessel))

def standlize_curves(xlsdir, xkey, ykey, savepath, matchstr='curve_xy'):
    """
    xlsdir = '/home/tpde/project/ArteryVeinSO2/WorkSpace2/AllCurves/SO2net/artery'
    xkey, ykey = 'pr_x', 'pr_y'
    savepath = '/home/tpde/project/ArteryVeinSO2/WorkSpace2/AllCurves/SO2net_artery_Rstd.xls'
    matchstr='curve_xy'
    """
    dict_stdcurves = {'stdx':[x/500 for x in range(501)]}
    list_xlspath = [os.path.join(xlsdir, file) for file in os.listdir(xlsdir) if matchstr in file]
    for idx, xlspath in enumerate(list_xlspath):
        dict_curve = readexcel(xlspath, 'dictlist', 'h', 'float')
        list_xy = [xy for xy in zip(dict_curve[xkey], dict_curve[ykey])]
        list_xy.sort(key=lambda xy:xy[0], reverse=False)
        list_x = [xy[0] for xy in list_xy]
        list_y = [xy[1] for xy in list_xy]
        dict_stdcurves['stdy%d'%idx] = []
        for std_x in dict_stdcurves['stdx']:
            min_x = [x for x in list_x if x<std_x]
            if(len(min_x)==0):
                dict_stdcurves['stdy%d'%idx].append(list_y[0])
            elif(len(min_x)==len(list_x)):
                dict_stdcurves['stdy%d'%idx].append(list_y[-1])
            else:
                a = std_x - min_x[-1]
                b = list_x[len(min_x)] - std_x
                std_y = (b*list_y[len(min_x)-1]+a*list_y[len(min_x)])/(a+b)
                dict_stdcurves['stdy%d'%idx].append( std_y )

    arr_stdy = np.array([dict_stdcurves['stdy%d'%idx] for idx in range(len(list_xlspath))])
    dict_stdcurves['ymean'] = np.mean(arr_stdy, axis=0).tolist()
    dict_stdcurves['yvar'] = np.std(arr_stdy, axis=0).tolist()
    save2excel(dict_stdcurves, savepath)
    return dict_stdcurves

def sortexcel_autosearch(dict_expdir, savefilepath, evalfile, matchstr):
    """
    Args:
        dict_expdir = { 'mm_cascade_dp0.1':'/home/tpde/project/Ailib3/workSpace/arteriovenous/myThirdDraw/mm_cascade_dp0.1',
                        'mm_cascade_dp0.1_':'/home/tpde/project/Ailib3/workSpace/arteriovenous/myThirdDraw/mm_cascade_dp0.1_',
                        'rgb_cascade':'/home/tpde/project/Ailib3/workSpace/arteriovenous/myThirdDraw/rgb_cascade'}
        savefilepath = 'myThirdDraw.xls'
        evalfile='evalpixel.xls'
        matchstr = 'split'
    """
    dict_all = dict()
    for expname in dict_expdir:
        expdir = dict_expdir[expname]
        if any(['eval_' in sfile for sfile in os.listdir(expdir)]):
            for evaldir in os.listdir(expdir):
                if os.path.isdir(os.path.join(expdir,evaldir)) and 'eval_' in evaldir:
                    if evalfile in os.listdir(os.path.join(expdir,evaldir)):
                        filepath = os.path.join(expdir,evaldir,evalfile)
                        key = expname+'_'+evaldir[evaldir.index(matchstr)+len(matchstr):]
                        dict_all[key] = readexcel(filepath, 'dict', 'vertical', 'float')
                    else:
                        for task in os.listdir( os.path.join(expdir,evaldir) ):
                            if os.path.isdir(os.path.join(expdir,evaldir,task)):
                                filepath = os.path.join(expdir,evaldir,task,evalfile)
                                key = expname+'_'+task+evaldir[evaldir.index(matchstr)+len(matchstr):]
                                dict_all[key] = readexcel(filepath, 'dict', 'vertical', dtype='float')
                    
        else:
            for task in os.listdir(expdir):
                list_evaldir = [sfile for sfile in os.listdir(os.path.join(expdir,task)) if 'eval_' in sfile]
                # assert(any(list_evaldir))
                for evaldir in list_evaldir:
                    filepath = os.path.join(expdir,task,evaldir,evalfile)
                    key = expname+'_'+task+evaldir[evaldir.index(matchstr)+len(matchstr):]
                    dict_all[key] = readexcel(filepath, 'dict', 'vertical', 'float')
    dict_all = completeMisskey(dict_all)
    save2excel(dict_all, savefilepath)

def autosearch_curvexls(list_keys, dict_dir, save_dir, list_xlsfile, matchstr):
    """
    Args:        
        list_keys = ['SM_Unet_Vein5_best', 'SM_Unet_Vein4_800']
        dict_dir = { 'SM_Unet':'IEEE/SM_Unet',
                    'SM_RUnet':'IEEE/SM_RUnet',
                    'SM_CRUnet':'IEEE/SM_CRUnet', }
        save_dir = 'IEEE_curve'
        list_xlsfile=['evalpixel.xls', 'curve_xy.xls']
        matchstr = 'split'
    function:
        copy 'evalpixel.xls' and 'curve_xy.xls' to save_dir
    """
    def copyfile(srcdir, dstdir, tailstr):
        mkdir( dstdir )
        for xlsfile in list_xlsfile:
            srcpath = os.path.join(srcdir, xlsfile)
            if os.path.isdir(srcpath):
                if False:             
                    dstpath = os.path.join(dstdir, xlsfile)
                    mkdir( dstpath )
                    os.system('cp -rf ' + srcpath + '/* ' + dstpath)
                else:
                    dstpath = os.path.join(dstdir, xlsfile+tailstr)
                    assert(not os.path.exists(dstpath))
                    os.system('cp -rf ' + srcpath + ' ' + dstpath)
            else:                
                dstpath = os.path.join(dstdir, xlsfile.replace('.xls',tailstr+'.xls') )
                os.system('cp ' + srcpath + ' ' + dstpath)
            
    for xlskey in list_keys:
        print(xlskey)
        expkeys = [key for key in list(dict_dir.keys()) if key in xlskey]
        assert( len(expkeys)==1 )
        expkey, expdir = expkeys[0], dict_dir[expkeys[0]]
        if any(['eval_' in sdir for sdir in os.listdir(expdir)]):
            evaldirs = [sdir for sdir in os.listdir(expdir) \
                        if ('eval_' in sdir) and sdir[sdir.index(matchstr)+len(matchstr):] in xlskey]
            evaldirs = [sdir for sdir in evaldirs if sdir[sdir.index(matchstr)+len(matchstr):]==    \
                        xlskey[xlskey.index(sdir[sdir.index(matchstr)+len(matchstr):]):]    ]
            assert( len(evaldirs)==1 )
            if list_xlsfile[0] in os.listdir( os.path.join(expdir,evaldirs[0]) ):
                tailstr = evaldirs[0][evaldirs[0].index(matchstr)+len(matchstr):]
                copyfile( os.path.join(expdir,evaldirs[0]), os.path.join(save_dir,expkey), tailstr)
            else:
                taskdirs = [sdir for sdir in os.listdir( os.path.join(expdir,evaldirs[0]) ) if sdir in xlskey]
                assert( len(taskdirs)==1 )
                tailstr = evaldirs[0][evaldirs[0].index(matchstr)+len(matchstr):]
                copyfile( os.path.join(expdir,evaldirs[0],taskdirs[0]), os.path.join(save_dir,expkey,taskdirs[0]), tailstr)
        else:
            taskdirs = [sdir for sdir in os.listdir( dict_dir[expkeys[0]] ) if sdir in xlskey]
            assert( len(taskdirs)==1 )
            evaldirs = [sdir for sdir in os.listdir(os.path.join(dict_dir[expkeys[0]],taskdirs[0])) \
                        if ('eval_' in sdir) and sdir[sdir.index(matchstr)+len(matchstr):] in xlskey]
            evaldirs = [sdir for sdir in evaldirs if sdir[sdir.index(matchstr)+len(matchstr):]==    \
                        xlskey[xlskey.index(sdir[sdir.index(matchstr)+len(matchstr):]):]    ]
            assert(len(evaldirs)==1)
            tailstr = evaldirs[0][evaldirs[0].index(matchstr)+len(matchstr):]
            copyfile( os.path.join(expdir,taskdirs[0],evaldirs[0]), os.path.join(save_dir,expkey,taskdirs[0]), tailstr)

def check_search(root_dir, savefilepath, matchstr='evalpixel'):
    dict_all = dict()
    for method in os.listdir(root_dir):
        if any(['.xls' in sfile for sfile in os.listdir(os.path.join(root_dir,method))]):
            for evalxls in [sfile for sfile in os.listdir(os.path.join(root_dir,method)) if matchstr in sfile]:
                key = method+evalxls[evalxls.index(matchstr)+len(matchstr):].strip('.xls')
                filepath = os.path.join(root_dir, method, evalxls)
                dict_all[key] = readexcel(filepath, 'dict', 'vertical', 'float')
        else:
            for task in os.listdir(os.path.join(root_dir,method)):
                assert( any(['.xls' in sfile for sfile in os.listdir(os.path.join(root_dir,method,task))]) )
                for evalxls in [sfile for sfile in os.listdir(os.path.join(root_dir,method,task)) if matchstr in sfile]:
                    key = method+'_'+task+evalxls[evalxls.index(matchstr)+len(matchstr):].strip('.xls')
                    filepath = os.path.join(root_dir,method,task, evalxls)
                    dict_all[key] = readexcel(filepath, 'dict', 'vertical', 'float')
    save2excel(dict_all, savefilepath)