import os
import numpy as np
import scipy.stats as stats
from datetime import datetime

from Ailib.basic import mkdir
from Ailib.module.log import Loger
from Ailib.datasets import NumpyDataset
from Ailib.io import readexcel, save2excel
from Ailib.analyze.graphic import plot_datecurve, plot_pairdatecurve

from network.cnnlogCosh import Aider

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

is_train = True
is_test = True

sname = 'GC'
batch_size = 32
img_h, img_w = 4, 4
iptlev = 44
outlev = 6
lr = 0.0005
epoch_start = 1
epoch_total = 100
freq_save = 500


list_mete = ['no2', 'pressure', 'tempc', 'ua', 'va', 'wa']
list_noise = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]

def adjust( dict_input, dict_label ):
    times = dict_label['time']
    label = dict_label['data']/1E12
    # print('label', label.shape, np.min(label), np.max(label))

    arr_tenv = np.zeros((batch_size, 24))
    for b in range( batch_size ):
        arr_tenv[b, int(times[b][8:10])] = 1
        
    b, h, w, c = batch_size, img_h, img_w, iptlev
    list_msk = [np.ones((b, h, w, c)), ]
    for k in range( min(img_h//2,img_w//2) ):
        msk = np.zeros( (b, h, w, 64) )
        msk[:,   k:h-k,     k:w-k,  :] = 1
        msk[:, k+1:h-k-1, k+1:w-k-1,:] = 0
        list_msk.append(msk)
    list_msk.append( np.ones((b,2)) )
    
    list_input = [ ]
    for i, atm in enumerate(list_mete):
        arr = dict_input[atm][...,:iptlev]
        # print(atm, arr.shape, np.min(arr), np.max(arr))
        list_input.append( arr + np.random.normal(0, list_noise[i], size=arr.shape) )
    list_input.append( arr_tenv )
    return times, list_input, label, list_msk


def train(aider, trainset, testset, traindir, epoch_start, epoch_total):
    
    train_dir = lambda file : os.path.join(traindir, file)
    loger  = Loger(['loss'], True)
    if epoch_start > 1:
        aider.load( train_dir('model_best.ckpt') )
        loger.load_epochlog( train_dir('log_train.xls'), train_dir('log_valid.xls') )
            
    for epoch in range(epoch_start, epoch_total+1):
        trainset.reset(is_shuffle=True)
        iter = 0
        while( not trainset.read_EOF ):
            iter += 1
            _, dict_input, dict_label = trainset.next_batch( batch_size )
            times, list_input, label, list_msk = adjust( dict_input, dict_label )
            loss, preds = aider.train( list_input, label, list_msk, lr)
            loger.update_batchlog(False, loss.mean())
            print('epoch(%d), iter(%d),\ttrain loss(%f)'%(epoch, iter, loss.mean()))
        
        if epoch % freq_save == 0:
            mkdir( train_dir('%05d'%epoch) )
            aider.save( train_dir('%05d/model.ckpt'%epoch) )
            loger.plot_log(['loss'], 'loss-curve', 'epoch', 'loss', train_dir('loss_%d.png'%epoch), 'both')
        
        # validset
        testset.reset(is_shuffle=False)
        while( not testset.read_EOF ):
            _, dict_input, dict_label = testset.next_batch( batch_size )
            times, list_input, label, list_msk = adjust( dict_input, dict_label )
            loss, _ = aider.test(list_input, label, list_msk)
            loger.update_batchlog(True, loss.mean())
            print('epoch(%d),\tvalid loss(%f)'%(epoch, loss.mean()))
        loger.update_epochlog()
        loger.save_log( train_dir('log_train.xls'), train_dir('log_valid.xls') )
        
        if loger.is_updateweight('loss'):
            aider.save( train_dir('best/model.ckpt') )
    loger.plot_log(['loss'], 'loss-curve', 'epoch', 'loss', train_dir('loss.png'), 'both')


def test(aider, testset):
    list_times = []
    list_label = []
    list_preds = []
    testset.reset(is_shuffle=False)
    while( not testset.read_EOF ):
        _, dict_input, dict_label = testset.next_batch( batch_size )
        times, list_input, label, list_msk = adjust( dict_input, dict_label )
        loss, preds = aider.test(list_input, label, list_msk)
        
        list_times.append( times )
        list_label.append( label )
        list_preds.append( preds )
        
    arr_times = np.concatenate(list_times, axis=0)
    arr_label = np.concatenate(list_label, axis=0)
    arr_preds = np.concatenate(list_preds, axis=0)
    return arr_times, arr_label, arr_preds


if __name__ == '__main__':
    for RP in [7]:
        readdir = lambda file:os.path.join('/home/tpde/project/DataBase/Alldata/1_我的数据处理/3_forTrain', sname+'_N%d'%1, file)
        train_dir = lambda file:os.path.join('train%d'%RP, file)
        
        dict_doas= {'time':np.load( readdir('metetime.npy') ),
                    'data':np.load( readdir('maxdoas.npy') ) }
        dict_atm = {atm:np.load(readdir(atm+'.npy')) for atm in list_mete}
        print( stats.pearsonr( dict_atm['no2'].sum(axis=(1,2,3)),  dict_doas['data'].sum(axis=-1)) )
        
        dict_idx = readexcel( readdir('idx_0.75.xls'), 'dictlist', dtype='int')
        trainset= NumpyDataset( task_type='cls',
                                input={key:value[dict_idx['train']] for key,value in dict_atm.items()},
                                target={key:value[dict_idx['train']] for key,value in dict_doas.items()})
        testset = NumpyDataset( task_type='cls',
                                input={key:value[dict_idx['test']] for key,value in dict_atm.items()},
                                target={key:value[dict_idx['test']] for key,value in dict_doas.items()}) 
        # Initializing model
        aider = Aider(batch_size, img_h, img_w, iptlev, outlev)
        if is_train:
            train(aider, trainset, testset, train_dir(''), epoch_start, epoch_total)
        
        list_noise = [0, 0, 0, 0, 0, 0]
        for which_weight in ['best']+list(range(freq_save,epoch_total+1,freq_save)):
            if which_weight!='best':
                which_weight = '%05d'%which_weight
                if not os.path.exists(train_dir('%s'%which_weight)):
                    break
             
            eval_dir = lambda file:os.path.join('eval_%s%d'%(which_weight,RP), file)
            
            if is_test:
                mkdir( eval_dir(''))
                aider.load( train_dir('%s/model.ckpt'%which_weight) )
                arr_time, arr_label, arr_pred = test(aider, testset)
                print('predict:', arr_time.shape, arr_label.shape, arr_pred.shape)
                
                list_i= list( range(arr_time.shape[0]) )
                itime = zip(list_i, arr_time.tolist())
                d = sorted(itime, key=lambda x:x[1])
                list_i, list_time = list( zip(*d) )
                list_i, list_time = list(list_i), list(list_time)
                arr_label=arr_label[list_i]
                arr_pred = arr_pred[list_i]
                
                dict_scater = {}
                dict_eval = {}
                for level in range(arr_label.shape[-1]):
                    p = stats.pearsonr(arr_label[:,level], arr_pred[:,level])[0]
                    dict_eval['pearsonr_%d'%level] = p
                    dict_scater[sname+'_level%d(%.4f)'%(level,p)] = {'x':arr_label[:,level], 'y':arr_pred[:,level]}
                dict_eval['pearsonr'] = np.mean([dict_eval['pearsonr_%d'%level] for level in range(arr_label.shape[-1])])
                save2excel(dict_eval, eval_dir('evalall.xls'))
                
#                 from Ailib.analyze.graphic import plot_curves
#                 plot_curves( dict_scater, 'maxdoas prediction(%s)'%sname, 'maxdoas', 'predict', 
#                                     eval_dir('scater.png'), is_scatter=True, fsize=(20,20))
                
                for month in range(4,13):
                    t1 = '2018-%02d-01 00:00:00'%month
                    t2 = '2018-%02d-01 00:00:00'%(month+1)
                    if month == 12:
                        t2 = '2019-01-01 00:00:00'
                    trange = [datetime.strptime(t1, '%Y-%m-%d %H:%M:%S'),
                              datetime.strptime(t2, '%Y-%m-%d %H:%M:%S')]
                    
                    dict_curves = {}
                    for lev in range(6):
                        dict_curves.update({'level%d'%lev: { 'date':list_time,
                                                            'data1':arr_label[:,lev],
                                                            'data2':arr_pred[:,lev]}  })
                        
                    plot_pairdatecurve(dict_curves, eval_dir('label_pred_%d.png'%month), 'label_pred(%s)'%sname,
                                    'date', 'NO2', struct='%Y%m%d%H%M%S', fsize=(50,30), ymax=-1, trange=trange)
                    
#                     for lev in range(6):
#                         dict_curves = { 'label(level%d)'%lev: { 'date':list_time, 'data':arr_label[:,lev]},
#                                         'predict(level%d)'%lev:{ 'date':list_time,'data':arr_pred[:,lev]}  }
#                       
#                         plot_datecurve(dict_curves, eval_dir('label_pred_lev%d_%d.png'%(lev,month)), 'label_pred(%s level%d)'%(sname, lev),
#                                     'date', 'NO2', struct='%Y%m%d%H%M%S', fsize=(50,10), ymax=-1, trange=trange)
        

