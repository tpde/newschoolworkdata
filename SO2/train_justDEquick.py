#! -*- coding:utf-8 -*-
import os
import numpy as np
from Ailib.basic import *
from Ailib.imglib import array2img, combineimages, write_images, read_images
from Ailib.nparray import stack_array, arrs2dict, save_npy, load_npy, normalize, thresh_otsu_arrs, groZeroOne
from Ailib.datasets import Dataset
from Ailib.transforms import usepil as tsform
from Ailib.module import Loger
from cal_so2 import cal_so2_vessel_AV
from keras import backend as K
K.set_image_dim_ordering('tf')

os.environ['CUDA_VISIBLE_DEVICES']='0'

from so2net import get_model



NET = 'SO2net'
img_w = 576
img_h = 576
img_ch = 3
finetuning = 0
epoch_start = 0
epoch_total = 1000
freq_save = 50
freq_sample=1000
batch_size = 2
is_train = False
is_test = True
is_eval = True
       
    
ipt_RGB = np.zeros([batch_size,img_h,img_w,img_ch], np.float)
ipt_610 = np.zeros([batch_size,img_h,img_w,1], np.float)
ipt_570 = np.zeros([batch_size,img_h,img_w,1], np.float)
iptPreBV= np.zeros([batch_size,img_h,img_w,1], np.float)
ipt_SO2 = np.zeros([batch_size,img_h,img_w,1], np.float)


def adjust(dict_batch):
    normalize(dict_batch['RGB'], ipt_RGB)
    normalize(dict_batch['610'], ipt_610)
    normalize(dict_batch['570'], ipt_570)
    iptPreBV[:]= dict_batch['preBV']/255.
    # iptPreV[:] = dict_batch_['possi_vein']/255
    # iptPreA[:] = dict_batch_['possi_artery']/255
    ipt_SO2[:] = dict_batch['SO2']/255.
    print( np.unique(ipt_SO2[:]) )
    batch_x = [ipt_RGB, ipt_610, ipt_570, ipt_SO2, iptPreBV]
    batch_y = [dict_batch['groV']/255, dict_batch['groA']/255]
    return batch_x, batch_y

def sample(dict_batch, batch_preV, batch_preA, sampledir, offset):    
    dict_bv = combineimages([array2img(dict_batch['RGB']), array2img(dict_batch['preBV']), array2img(dict_batch['SO2'])])
    write_images(dict_bv, sampledir+'_vessel', offset)
    
    dict_vein = combineimages([array2img(dict_batch['610']), array2img(dict_batch['groV']), array2img(batch_preV)])
    write_images(dict_vein, sampledir+'_vein', offset)
    
    dict_artery = combineimages([array2img(dict_batch['570']), array2img(dict_batch['groA']), array2img(batch_preA)])
    write_images(dict_artery, sampledir+'_artery', offset)


def train(model, trainset, validset, epoch_start, epoch_total, traindir):
    
    train_dir = lambda file : os.path.join(traindir, file)
    
    loger  = Loger(['loss','loss1','loss2',  'acc1','acc2'], True)
    if epoch_start>0:
        loger.load_epochlog( train_dir('log_train.xls'), train_dir('log_valid.xls') )
    
    n_iter = 0
    for epoch in range(epoch_start, epoch_total):
        trainset.reset(is_shuffle=True)
        while( not trainset.read_EOF ):
            files, dict_batch = trainset.next_batch( batch_size )
            batch_x, batch_y = adjust( dict_batch )
            loss, loss1, loss2, acc1, acc2 = model.SO2net.train_on_batch(batch_x, batch_y)
            loger.update_batchlog(False, loss, loss1, loss2, acc1, acc2)
            print('epoch(%d), iter(%d),\t'%(epoch, n_iter), loss, loss1, loss2, acc1, acc2 )
            
            if n_iter % freq_sample==0:
                batch_preV, batch_preA = model.SO2net.predict_on_batch( batch_x )
                sample(dict_batch, batch_preV, batch_preA, train_dir('sample_train'), batch_size*n_iter/freq_sample)
            
            n_iter += 1
            
        if (epoch+1) % freq_save == 0:
            model.Dnet.save_weights(train_dir('%04d_Dnet.h5'%(epoch+1)), overwrite=True)
            model.Enet.save_weights(train_dir('%04d_Enet.h5'%(epoch+1)), overwrite=True)
            # model.SO2net.save_weights(train_dir('%04d_SO2net.h5'%epoch), overwrite=True)
            loger.plot_log(['loss'], 'loss-curve', 'epoch', 'loss', train_dir('loss_%d.png'%epoch), 'both')
            
        # =============================== valid ==============================
        validset.reset(is_shuffle=False)
        while( not validset.read_EOF ):
            files, dict_batch = validset.next_batch( batch_size )
            batch_x, batch_y = adjust( dict_batch )
                
            loss, loss1, loss2, acc1, acc2 = model.SO2net.evaluate(batch_x, batch_y, batch_size=batch_size, verbose=0)
            loger.update_batchlog(True, loss,loss1,loss2, acc1,acc2)
            print('valid iter:',loss, loss1, loss2, acc1, acc2)
            
            if n_iter%freq_sample==0:
                batch_preV, batch_preA = model.SO2net.predict_on_batch( batch_x )
                sample(dict_batch, batch_preV, batch_preA, train_dir('sample_valid'), batch_size*n_iter/freq_sample)
                
        loger.update_epochlog()
        loger.save_log( train_dir('log_train.xls'), train_dir('log_valid.xls') )
        
        if loger.is_updateweight('loss'):
            model.Dnet.save_weights(train_dir('best_Dnet.h5'), overwrite=True)
            model.Enet.save_weights(train_dir('best_Enet.h5'), overwrite=True)
            
    loger.plot_log(['loss'], 'loss-curve', 'epoch', 'loss', train_dir('loss.png'), 'both')
    

def test(model, testset):
    list_files, list_SO2 = [], []
    list_vessel, list_vein, list_artery = [], [], []
    
    testset.reset(is_shuffle=False)
    while( not testset.read_EOF ):
        files, dict_batch = testset.next_batch( batch_size )
        batch_x, batch_y = adjust( dict_batch )
        
        batch_preV, batch_preA = model.SO2net.predict_on_batch( batch_x )

        for iter in range(0):
            bw_BV= thresh_otsu_arrs(dict_batch['preBV'])
            bw_V = thresh_otsu_arrs(batch_preV)
            bw_A = thresh_otsu_arrs(batch_preA)
            dict_batch['SO2'] = cal_so2_vessel_AV(bw_BV, bw_V, bw_A, dict_batch['RGB'], dict_batch['610'], dict_batch['570'])
            batch_x[-2] = dict_batch['SO2']/255
            batch_preV, batch_preA = model.SO2net.predict_on_batch( batch_x )
        
        list_files += files
        list_vessel.append(dict_batch['preBV'])
        list_vein.append(batch_preV)
        list_artery.append(batch_preA)
        list_SO2.append( dict_batch['SO2'] )        
        
    arr_vessel= np.concatenate(list_vessel,axis=0)[:len(testset)]
    arr_vein  = np.concatenate(list_vein,  axis=0)[:len(testset)]
    arr_artery= np.concatenate(list_artery,axis=0)[:len(testset)]
    arr_SO2   = np.concatenate(list_SO2,   axis=0)[:len(testset)]
    
    return list_files, arr_vessel, arr_vein, arr_artery, arr_SO2


if __name__ == '__main__':
    
    rate = 0.6
#     for DB in ['split1', 'split2', 'split3', 'split4', 'split5', ]:
    for DB in ['split5' ]:
        image_dir = lambda file:os.path.join('../DB3th_noOD/%s'%DB, file)
        train_dir = lambda file:os.path.join('train_%s_%s'%(NET,DB), file)
        test_dir = lambda file:os.path.join('test_%s_%s'%(NET,DB), file)
        
        if is_train:   # ==================================================================
            # ------------------- train ----------------
            trainset = Dataset( datas ={'RGB':image_dir('train/RGB'),
                                        '610':image_dir('train/610'), 
                                        '570':image_dir('train/570'),
                                        'preBV':image_dir('train/possi_vessel'),
                                        'SO2':image_dir('train/possi_SO2'), 
                                        'groBV':image_dir('train/vessel'),
                                        'groV':image_dir('train/vein'),
                                        'groA':image_dir('train/artery')},
                                list_tsf = [[None, tsform.Compose([tsform.ImageRead(), tsform.RandomFlip(), tsform.RandomRotate(),])],
                                            ['RGB']],
                                task_type='seg')
            
             
            # valid set
            validset = Dataset( datas ={'RGB':image_dir('test/RGB'),
                                        '610':image_dir('test/610'), 
                                        '570':image_dir('test/570'),
                                        'preBV':image_dir('test/possi_vessel'),
                                        'SO2':image_dir('test/possi_SO2'), 
                                        'groBV':image_dir('test/vessel'),
                                        'groV':image_dir('test/vein'),
                                        'groA':image_dir('test/artery')},
                                list_tsf = [[None, tsform.Compose([tsform.ImageRead(),])],
                                            ],
                                task_type='seg')
            
            model = get_model(img_h, img_w, rate)
            if epoch_start>0:
                print('--------------- load weights from:',train_dir('last_*net.h5'))
                model.Dnet.load_weights(train_dir('last_Dnet.h5'))
                model.Enet.load_weights(train_dir('last_Enet.h5'))
            
            mkdir(train_dir(''))
            train(model, trainset, validset, epoch_start, epoch_total, train_dir(''))



        if is_test: # ==================================================================
            mkdir(test_dir(''))
            for which_weight in ['0350', '0450', '0550', '0650', '0750', '0850', '0950']:
#             for which_weight in ['best', '0200','0300','0400','0500','0600','0700','0800', '0900', '1000']:    
#             for which_weight in [ 'best', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000']:
                eval_dir = lambda file:os.path.join('eval_%s_%s_%s'%(NET,DB,which_weight), file)
                
                # ------------------- test -----------------                
                testset = Dataset( datas = {'RGB':image_dir('test/RGB'),
                                            '610':image_dir('test/610'), 
                                            '570':image_dir('test/570'),
                                            'preBV':image_dir('test/possi_vessel'),
                                            'SO2':image_dir('test/possi_SO2'), 
                                            'groBV':image_dir('test/vessel'),
                                            'groV':image_dir('test/vein'),
                                            'groA':image_dir('test/artery')},
                                list_tsf = [[None, tsform.Compose([tsform.ImageRead(),])],
                                            ],
                                task_type='seg')
                
                model = get_model(img_h, img_w, rate)
                model.Dnet.load_weights(train_dir('%s_Dnet.h5'%which_weight))
                model.Enet.load_weights(train_dir('%s_Enet.h5'%which_weight))
                list_files, arr_vessel, arr_vein, arr_artery, arr_SO2 = test(model, testset)
                
                save_npy(arrs2dict(arr_vessel, list_files), test_dir('npy_vessel'))
                save_npy(arrs2dict(arr_vein, list_files), test_dir('npy_vein'))
                save_npy(arrs2dict(arr_artery, list_files), test_dir('npy_artery'))
                save_npy(arrs2dict(arr_SO2, list_files), test_dir('npy_SO2'))
                
#                     uncer_vessel = (- arr_pre_vessel*np.log(arr_pre_vessel+1E-7) ) + (- (1-arr_pre_vessel)*np.log(1-arr_pre_vessel+1E-7) )
#                     uncer_vein = (- arr_pre_vein*np.log(arr_pre_vein+1E-7) ) + (- (1-arr_pre_vein)*np.log(1-arr_pre_vein+1E-7) )
#                     uncer_artery = (- arr_pre_artery*np.log(arr_pre_artery+1E-7) ) + (- (1-arr_pre_artery)*np.log(1-arr_pre_artery+1E-7) )
#                     uncer_vessel = uncer_vessel/(np.max(uncer_vessel)+1E-7)*256
#                     uncer_vein = uncer_vein/(np.max(uncer_vein)+1E-7)*256
#                     uncer_artery = uncer_artery/(np.max(uncer_artery)+1E-7)*256
#                     
#                     dict_uncer_vessel = arrs2dict(uncer_vessel, list_name)
#                     dict_uncer_vein = arrs2dict(uncer_vein,   list_name)
#                     dict_uncer_artery = arrs2dict(uncer_artery, list_name)
#                     write_images(dict_uncer_vessel, test_dir('uncer_vessel'))
#                     write_images(dict_uncer_vein, test_dir('uncer_vein'))
#                     write_images(dict_uncer_artery, test_dir('uncer_artery'))
                
                
#                 # ------------------- eval -----------------
                if is_eval:
                    mkdir(eval_dir(''))
                    from Ailib.analyze.common import plot_evalpixel
                    
                    dict_gro_vessel = groZeroOne(read_images(image_dir('test/vessel')))
                    dict_gro_vein = groZeroOne(read_images(image_dir('test/vein')))
                    dict_gro_artery = groZeroOne(read_images(image_dir('test/artery')))

                    dict_pre_vessel = load_npy(test_dir('npy_vessel'))
                    dict_pre_vein = load_npy(test_dir('npy_vein'))
                    dict_pre_artery = load_npy(test_dir('npy_artery'))
                     
                    plot_evalpixel(dict_gro_vessel, dict_pre_vessel, eval_dir('vessel'))
                    plot_evalpixel(dict_gro_vein, dict_pre_vein, eval_dir('vein'))
                    plot_evalpixel(dict_gro_artery, dict_pre_artery, eval_dir('artery'))
                     

