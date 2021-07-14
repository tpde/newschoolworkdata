#! -*- coding:utf-8 -*-
import os
import numpy as np
from Ailib.basic import *
from Ailib.imglib import array2img, combineimages, write_images 
from Ailib.nparray import stack_array, arrs2dict, save_npy, load_npy
from Ailib.io import savelist, readlist
from Ailib.datasets import DataIter
from Ailib.analyze.graphic import plot_curves
from Ailib.datasets.common import prepare_images
from Ailib.nparray import thresh_otsu_arrs
from Ailib.module import Loger
from cal_so2 import *
from keras import backend as K
K.set_image_dim_ordering('tf')

os.environ['CUDA_VISIBLE_DEVICES']='0'

from so2net import get_model

isDE = False

NET = 'SO2net'
img_w = 576
img_h = 576
img_ch = 3
finetuning = 0
epoch_start = 0
epoch_total = 1000
frequece_save=100
frequece_sample=1000
batch_size = 2
is_train = True
is_test = True
is_eval = False

dict_batch_ = {'RGB':np.zeros([batch_size,img_h,img_w,img_ch],dtype=float),
               '610':np.zeros([batch_size,img_h,img_w,1],dtype=float),
               '570':np.zeros([batch_size,img_h,img_w,1],dtype=float),
               'vessel' :np.zeros([batch_size,img_h,img_w,1],dtype=float),
               'vein'   :np.zeros([batch_size,img_h,img_w,1],dtype=float),
               'artery' :np.zeros([batch_size,img_h,img_w,1],dtype=float),
               'possi_vessel':np.zeros([batch_size,img_h,img_w,1],dtype=float),
               'possi_vein':  np.zeros([batch_size,img_h,img_w,1],dtype=float),
               'possi_artery':np.zeros([batch_size,img_h,img_w,1],dtype=float),
               'possi_SO2':   np.zeros([batch_size,img_h,img_w,1],dtype=float)}        
    
ipt_RGB = np.zeros([batch_size,img_h,img_w,img_ch], np.float)
ipt_610 = np.zeros([batch_size,img_h,img_w,1], np.float)
ipt_570 = np.zeros([batch_size,img_h,img_w,1], np.float)
iptPreBV= np.zeros([batch_size,img_h,img_w,1], np.float)
iptPreV = np.zeros([batch_size,img_h,img_w,1], np.float)
iptPreA = np.zeros([batch_size,img_h,img_w,1], np.float)
ipt_SO2 = np.zeros([batch_size,img_h,img_w,1], np.float)


def train(model, dataset, traindir, epoch_start, epoch_total):
    mkdir(traindir)
    train_dir = lambda file : os.path.join(traindir, file)
    
    list_losstrain = []
    list_acctrain = []
    list_lossvalid = []
    list_accvalid = []
    
    if epoch_start>0:
        list_losstrain= [float(y) for y in readlist( train_dir('list_losstrain.txt'))]
        list_lossvalid= [float(y) for y in readlist( train_dir('list_lossvalid.txt'))]
        list_acctrain = [float(y) for y in readlist( train_dir('list_acctrain.txt'))]
        list_accvalid = [float(y) for y in readlist( train_dir('list_accvalid.txt'))]
    
    sample_idx=0
    for epoch in range(epoch_start, epoch_total, 1):
        sum_losstrain = []
        sum_acctrain = []
        dataset.trainEOF = False
        while(dataset.trainEOF is False):
            dict_batch = dataset.next_train(batch_size, False)
            dataset.dataaug( dict_batch, dict_batch_, {'RGB':True,'610':True,'570':True,
                            'vessel':False, 'vein':False, 'artery':False, 'possi_vessel':True,
                            'possi_vein':True, 'possi_artery':True, 'possi_SO2':True} )
            dataset.normalize(dict_batch_['RGB'], ipt_RGB)
            dataset.normalize(dict_batch_['610'], ipt_610)
            dataset.normalize(dict_batch_['570'], ipt_570)
            iptPreBV[:]= dict_batch_['possi_vessel']/256
            # iptPreV[:] = dict_batch_['possi_vein']/256
            # iptPreA[:] = dict_batch_['possi_artery']/256
            ipt_SO2[:] = dict_batch_['possi_SO2']/256
                
            batch_x = [ipt_RGB, ipt_610, ipt_570, ipt_SO2, iptPreBV]
            batch_y = [dict_batch_['vein'], dict_batch_['artery']]
            
            loss_train, loss1, loss2, acc1, acc2 = model.SO2net.train_on_batch(batch_x, batch_y)
            acc_train = (acc1 + acc2 ) / 2
            sum_losstrain.append(loss_train)
            sum_acctrain.append(acc_train)
            print('train iter:', loss_train, loss1, loss2, acc1, acc2 )
            
            sample_idx += 1
            if sample_idx%frequece_sample==0:
                batch_preV, batch_preA = model.SO2net.predict_on_batch(batch_x)
                    
                dict_bv = combineimages([array2img(dict_batch_['RGB']), array2img(dict_batch_['possi_vessel']), array2img(dict_batch_['possi_SO2'])])
                write_images(dict_bv, train_dir('train_vessel'), batch_size*sample_idx/frequece_sample)
                
                dict_vein = combineimages([array2img(dict_batch_['610']), array2img(dict_batch_['vein']), array2img(batch_preV)])
                write_images(dict_vein, train_dir('train_vein'), batch_size*sample_idx/frequece_sample)
                
                dict_artery = combineimages([array2img(dict_batch_['570']), array2img(dict_batch_['artery']), array2img(batch_preA)])
                write_images(dict_artery, train_dir('train_artery'), batch_size*sample_idx/frequece_sample)
        
            
        curr_loss = sum(sum_losstrain)/len(sum_losstrain)
        curr_acc = sum(sum_acctrain)/len(sum_losstrain)
        print('epoch:%d, average: loss_train:%03f, acc_train:%03f'%(epoch, curr_loss, curr_acc) )
        list_losstrain.append(curr_loss)
        list_acctrain.append(curr_acc)
            
        if (epoch+1)%frequece_save==0:
            model.Dnet.save_weights(train_dir('%04d_Dnet.h5'%(epoch+1)), overwrite=True)
            model.Enet.save_weights(train_dir('%04d_Enet.h5'%(epoch+1)), overwrite=True)
            model.SO2net.save_weights(train_dir('%04d_SO2net.h5'%(epoch+1)), overwrite=True)
        

        # =============================== valid ==============================
        sum_lossvalid = []
        sum_accvalid = []
        for i in range(1*dataset.num_valid):
            dict_batch = dataset.next_valid(batch_size, False)
            dataset.dataaug( dict_batch, dict_batch_, {'RGB':True,'610':True,'570':True,
                            'vessel':False, 'vein':False, 'artery':False, 'possi_vessel':True,
                            'possi_vein':True,'possi_artery':True, 'possi_SO2':True} )
            dataset.normalize(dict_batch_['RGB'], ipt_RGB)
            dataset.normalize(dict_batch_['610'], ipt_610)
            dataset.normalize(dict_batch_['570'], ipt_570)
            iptPreBV[:]= dict_batch_['possi_vessel']/256
            # iptPreV[:] = dict_batch_['possi_vein']/256
            # iptPreA[:] = dict_batch_['possi_artery']/256
            ipt_SO2[:] = dict_batch_['possi_SO2']/256
                
            batch_x = [ipt_RGB, ipt_610, ipt_570, ipt_SO2, iptPreBV]
            batch_y = [dict_batch_['vein'], dict_batch_['artery']]
                
            loss_valid, loss1, loss2, acc1, acc2 = model.SO2net.evaluate(batch_x, batch_y, batch_size=batch_size, verbose=0)
            acc_valid = (acc1+acc2)/2
            sum_lossvalid.append(loss_valid)
            sum_accvalid.append(acc_valid)
            print('valid iter:',loss_valid,loss1, loss2, acc1, acc2)
            
            sample_idx += 1
            if sample_idx%frequece_sample==0:
                batch_preV, batch_preA = model.SO2net.predict_on_batch(batch_x)                        
                             
                dict_bv = combineimages([array2img(dict_batch_['RGB']), array2img(dict_batch_['possi_vessel']), array2img(dict_batch_['possi_SO2'])])
                write_images(dict_bv, train_dir('valid_vessel'), batch_size*sample_idx/frequece_sample)
                
                dict_vein = combineimages([array2img(dict_batch_['610']), array2img(dict_batch_['vein']), array2img(batch_preV)])
                write_images(dict_vein, train_dir('valid_vein'), batch_size*sample_idx/frequece_sample)
                
                dict_artery = combineimages([array2img(dict_batch_['570']), array2img(dict_batch_['artery']), array2img(batch_preA)])
                write_images(dict_artery, train_dir('valid_artery'), batch_size*sample_idx/frequece_sample)
                
        curr_loss = sum(sum_lossvalid)/len(sum_lossvalid)
        curr_acc = sum(sum_accvalid)/len(sum_accvalid)
        print('epoch:%d, average: loss_valid:%03f, acc_valid:%03f'%(epoch, curr_loss, curr_acc) )
        list_lossvalid.append(curr_loss)
        list_accvalid.append(curr_acc)
        
        if min(list_lossvalid) >= curr_loss:
            model.Dnet.save_weights(train_dir('best_Dnet.h5'), overwrite=True)
            model.Enet.save_weights(train_dir('best_Enet.h5'), overwrite=True)
            model.SO2net.save_weights(train_dir('best_SO2net.h5'), overwrite=True)
            
        savelist(list_losstrain, train_dir('list_losstrain.txt'))
        savelist(list_lossvalid, train_dir('list_lossvalid.txt'))
        savelist(list_acctrain, train_dir('list_acctrain.txt'))
        savelist(list_accvalid, train_dir('list_accvalid.txt'))
    
    plot_curves( {'loss_train':list_losstrain,'loss_valid':list_lossvalid},
                 'loss-curve', 'epoch', 'loss', train_dir('loss.png'), True)
    plot_curves( {'acc_train':list_acctrain,'acc_valid':list_accvalid},
                 'acc-curve', 'epoch', 'acc', train_dir('acc.png'), True)
    

def test(model, dataset):
    list_pre_vessel = list()
    list_pre_vein   = list()
    list_pre_artery = list()
    list_pre_SO2 = list()
    
    arr_gro_vessel = np.zeros( (dataset.num_valid, img_h,img_w,1), dtype=np.uint8 )
    arr_gro_vein   = np.zeros( (dataset.num_valid, img_h,img_w,1), dtype=np.uint8 )
    arr_gro_artery = np.zeros( (dataset.num_valid, img_h,img_w,1), dtype=np.uint8 )

    sample_idx = 1
    for i in range(dataset.num_valid//batch_size):
        dict_batch = dataset.next_test(batch_size, False)
        dataset.dataaug( dict_batch, dict_batch_, {'RGB':True,'610':True,'570':True, 'vessel':False,
                        'vein':False, 'artery':False, 'possi_vessel':True, 'possi_vein':True,
                        'possi_artery':True, 'possi_SO2':True}, angle=0, is_random=False )
        dataset.normalize(dict_batch_['RGB'], ipt_RGB)
        dataset.normalize(dict_batch_['610'], ipt_610)
        dataset.normalize(dict_batch_['570'], ipt_570)
        iptPreBV[:]= dict_batch_['possi_vessel']/256
        # iptPreV[:] = dict_batch_['possi_vein']/256
        # iptPreA[:] = dict_batch_['possi_artery']/256
        ipt_SO2[:] = dict_batch_['possi_SO2']/256
        
        for iter in range(1):
            batch_x = [ipt_RGB, ipt_610, ipt_570, ipt_SO2, iptPreBV]
            batch_y = [dict_batch_['vein'], dict_batch_['artery']]
            loss_valid, loss1, loss2, acc1, acc2 = model.SO2net.evaluate(batch_x, batch_y, batch_size=batch_size, verbose=0)
            acc_valid = (acc1+acc2)/2
            print('valid loss:', loss_valid,loss1, loss2)
            print('valid acc:', acc_valid, acc1, acc2)

            # ============================ sample ==========================
            batch_preV, batch_preA = model.SO2net.predict_on_batch(batch_x)                        

#             bw_BV= thresh_otsu_arrs(dict_batch_['possi_vessel'])
#             bw_V = thresh_otsu_arrs(batch_preV)
#             bw_A = thresh_otsu_arrs(batch_preA)
#             arr_SO2 = cal_so2_vessel_AV(bw_BV, bw_V, bw_A, dict_batch_['RGB'], dict_batch_['610'], dict_batch_['570'])
#             ipt_SO2[:] = dict_batch_['possi_SO2']/256
            
            
        sample_idx += 1
        list_pre_SO2.append( dict_batch_['possi_vessel'])
            
        list_pre_vessel.append(dict_batch_['possi_vessel'])
        list_pre_vein.append(batch_preV)
        list_pre_artery.append(batch_preA)
        
        arr_gro_vessel[i*batch_size:(i+1)*batch_size] = dict_batch_['vessel'].astype(np.uint8)
        arr_gro_vein[  i*batch_size:(i+1)*batch_size] = dict_batch_['vein'].astype(np.uint8)
        arr_gro_artery[i*batch_size:(i+1)*batch_size] = dict_batch_['artery'].astype(np.uint8)
        print('===****===', np.unique(arr_gro_vessel), np.unique(arr_gro_vein), np.unique(arr_gro_artery), )
     
    arr_pre_vessel= np.concatenate(list_pre_vessel, axis=0)[:dataset.num_valid]
    arr_pre_vein = np.concatenate(list_pre_vein, axis=0)[:dataset.num_valid]
    arr_pre_artery = np.concatenate(list_pre_artery, axis=0)[:dataset.num_valid]
    arr_pre_SO2 = np.concatenate(list_pre_SO2, axis=0)[:dataset.num_valid]
    
    return arr_pre_vessel, arr_pre_vein, arr_pre_artery, arr_pre_SO2, \
                    arr_gro_vessel, arr_gro_vein, arr_gro_artery
    


if __name__ == '__main__':
    
    rate = 0.6
    for DB in ['split1']: # 'split5', 'split4', 'split3', 'split2', 
        if DB=='split1':
            loadpath = 'best0'
        elif DB=='split2':
            loadpath = '0800'
        elif DB=='split3':
            loadpath = '0600'
        elif DB=='split4':
            loadpath = '0900'
        elif DB=='split5':
            loadpath = '0700'
        image_dir = lambda file:os.path.join('../newdb/%s'%DB, file)
        train_dir = lambda file:os.path.join('train_%s_%s'%(NET,DB), file)
        test_dir = lambda file:os.path.join('test_%s_%s'%(NET,DB), file)
        
        if is_train:   # ==================================================================
            # ------------------- train ----------------
            dataset = DataIter()
            list_dirs = [image_dir('train/RGB'), image_dir('train/610'), image_dir('train/570'),
                         image_dir('train/vessel'), image_dir('train/vein'), image_dir('train/artery'),
                         image_dir('train/possi_vessel'), image_dir('train/possi_vein'), 
                         image_dir('train/possi_artery'), image_dir('train/possi_SO2')]
            list_dict_img = prepare_images(list_dirs, [False, False, False, True, True, True, False, False, False, False],
                                    -1, img_w, [img_h], [None, None, None,  None, None, None,  None, None, None, None])
            list_array, _ = stack_array(list_dict_img)
            print('train:: RGB',list_array[0].shape, np.min(list_array[0]), np.max(list_array[0]))
            print('train:: 610',list_array[1].shape, np.min(list_array[1]), np.max(list_array[1]))
            print('train:: 570',list_array[2].shape, np.min(list_array[2]), np.max(list_array[2]))
            print('train:: vessel', list_array[3].shape, np.unique(list_array[3]))
            print('train:: vein',   list_array[4].shape, np.unique(list_array[4]))
            print('train:: artery', list_array[5].shape, np.unique(list_array[5]))
            print('train:: possi_vessel',list_array[6].shape, np.min(list_array[6]), np.max(list_array[6]))
            print('train:: possi_vein  ',list_array[7].shape, np.min(list_array[7]), np.max(list_array[7]))
            print('train:: possi_artery',list_array[8].shape, np.min(list_array[8]), np.max(list_array[8]))
            print('train:: possi_SO2',   list_array[9].shape, np.min(list_array[9]), np.max(list_array[9]))
            dataset.set_trainset( {'RGB':list_array[0], '610':list_array[1], '570':list_array[2],
                                   'vessel':list_array[3], 'vein':list_array[4], 'artery':list_array[5],
                                   'possi_vessel':list_array[6], 'possi_vein':list_array[7],
                                   'possi_artery':list_array[8], 'possi_SO2':list_array[9]}, 0.)
             
            # valid set
            list_dirs = [image_dir('test/RGB'), image_dir('test/610'), image_dir('test/570'),
                         image_dir('test/vessel'), image_dir('test/vein'), image_dir('test/artery'),
                         image_dir('test/possi_vessel'), image_dir('test/possi_vein'), 
                         image_dir('test/possi_artery'), image_dir('test/possi_SO2')]
            list_dict_img = prepare_images(list_dirs, [False, False, False, True, True, True, False, False, False, False],
                                    -1, img_w, [img_h], [None, None, None,  None, None, None,  None, None, None, None])
            list_array, _ = stack_array(list_dict_img)
            print('test:: RGB',list_array[0].shape, np.min(list_array[0]), np.max(list_array[0]))
            print('test:: 610',list_array[1].shape, np.min(list_array[1]), np.max(list_array[1]))
            print('test:: 570',list_array[2].shape, np.min(list_array[2]), np.max(list_array[2]))
            print('test:: vessel', list_array[3].shape, np.unique(list_array[3]))
            print('test:: vein',   list_array[4].shape, np.unique(list_array[4]))
            print('test:: artery', list_array[5].shape, np.unique(list_array[5]))
            print('test:: possi_vessel',list_array[6].shape, np.min(list_array[6]), np.max(list_array[6]))
            print('test:: possi_vein  ',list_array[7].shape, np.min(list_array[7]), np.max(list_array[7]))
            print('test:: possi_artery',list_array[8].shape, np.min(list_array[8]), np.max(list_array[8]))
            print('test:: possi_SO2',   list_array[9].shape, np.min(list_array[9]), np.max(list_array[9]))
            dataset.set_validset( {'RGB':list_array[0], '610':list_array[1], '570':list_array[2],
                                   'vessel':list_array[3], 'vein':list_array[4], 'artery':list_array[5],
                                   'possi_vessel':list_array[6], 'possi_vein':list_array[7],
                                   'possi_artery':list_array[8], 'possi_SO2':list_array[9]} )
            model = get_model(img_h, img_w, rate)
            if epoch_start>0:
                print('--------------- load weights from:',train_dir('%s_*net.h5'%loadpath))
                model.Dnet.load_weights(train_dir('%s_Dnet.h5'%loadpath))
                model.Enet.load_weights(train_dir('%s_Enet.h5'%loadpath))

            train(model, dataset, train_dir(''), epoch_start, epoch_total)



        if is_test: # ==================================================================
            for which_weight in ['best',]:# '0200','0300','0400','0500','0600','0700','0800', '0900', '1000']:    
#             for which_weight in [ 'best', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000']:
                eval_dir = lambda file:os.path.join('eval_%s_%s_%s'%(NET,DB,which_weight), file)
                
                # ------------------- test -----------------
                dataset = DataIter()
                list_dirs = [image_dir('test/RGB'), image_dir('test/610'), image_dir('test/570'),
                             image_dir('test/vessel'), image_dir('test/vein'), image_dir('test/artery'),
                             image_dir('test/possi_vessel'), image_dir('test/possi_vein'), 
                             image_dir('test/possi_artery'), image_dir('test/possi_SO2')]
                list_dict_img = prepare_images(list_dirs, [False, False, False, True, True, True, False, False, False, False],
                                        -1, img_w, [img_h], [None, None, None,  None, None, None,  None, None, None, None])
                list_array, list_name = stack_array(list_dict_img)
                print('test:: RGB',list_array[0].shape, np.min(list_array[0]), np.max(list_array[0]))
                print('test:: 610',list_array[1].shape, np.min(list_array[1]), np.max(list_array[1]))
                print('test:: 570',list_array[2].shape, np.min(list_array[2]), np.max(list_array[2]))
                print('test:: vessel', list_array[3].shape, np.unique(list_array[3]))
                print('test:: vein',   list_array[4].shape, np.unique(list_array[4]))
                print('test:: artery', list_array[5].shape, np.unique(list_array[5]))
                print('test:: possi_vessel',list_array[6].shape, np.min(list_array[6]), np.max(list_array[6]))
                print('test:: possi_vein  ',list_array[7].shape, np.min(list_array[7]), np.max(list_array[7]))
                print('test:: possi_artery',list_array[8].shape, np.min(list_array[8]), np.max(list_array[8]))
                print('test:: possi_SO2',   list_array[9].shape, np.min(list_array[9]), np.max(list_array[9]))
                dataset.set_testset( {'RGB':list_array[0], '610':list_array[1], '570':list_array[2],
                                       'vessel':list_array[3], 'vein':list_array[4], 'artery':list_array[5],
                                       'possi_vessel':list_array[6], 'possi_vein':list_array[7],
                                       'possi_artery':list_array[8], 'possi_SO2':list_array[9]} )
                model = get_model(img_h, img_w, rate)
        #         from keras.models import model_from_json
        #         model = model_from_json(open(train_dir('architecture.json')).read())
                model.Dnet.load_weights(train_dir('%s_Dnet.h5'%which_weight))
                model.Enet.load_weights(train_dir('%s_Enet.h5'%which_weight))
                
                arr_pre_vessel, arr_pre_vein, arr_pre_artery, arr_pre_SO2, arr_gro_vessel, arr_gro_vein, arr_gro_artery = test(model, dataset)
                
                dict_pre_vessel = arrs2dict(arr_pre_vessel, list_name)
                dict_pre_vein = arrs2dict(arr_pre_vein,   list_name)
                dict_pre_artery = arrs2dict(arr_pre_artery, list_name)
                dict_pre_SO2 = arrs2dict(arr_pre_SO2, list_name)
                save_npy(dict_pre_vessel, test_dir('npy_vessel'))
                write_images(dict_pre_vessel, test_dir('possi_vessel'))
                save_npy(dict_pre_vein, test_dir('npy_vein'))
                write_images(dict_pre_vein, test_dir('possi_vein'))
                save_npy(dict_pre_artery, test_dir('npy_artery'))
                write_images(dict_pre_artery, test_dir('possi_artery'))
                write_images(dict_pre_SO2, test_dir('possi_SO2'))
                print('===****===', np.unique(arr_gro_vessel), np.unique(arr_gro_vein), np.unique(arr_gro_artery), )
                save_npy(arrs2dict(arr_gro_vessel, list_name), test_dir('npy_gro_vessel'))
                save_npy(arrs2dict(arr_gro_vein, list_name), test_dir('npy_gro_vein'))
                save_npy(arrs2dict(arr_gro_artery, list_name), test_dir('npy_gro_artery'))
            
                dict_cmp_vessel = combineimages([array2img(list_dict_img[0]), array2img(dict_pre_vessel), array2img(list_dict_img[3])])
                dict_cmp_vein = combineimages([array2img(list_dict_img[1]), array2img(dict_pre_vein), array2img(list_dict_img[4])])
                dict_cmp_artery = combineimages([array2img(list_dict_img[2]), array2img(dict_pre_artery), array2img(list_dict_img[5])]) 
                dict_cmp_SO2 = combineimages([array2img(list_dict_img[0]), array2img(list_dict_img[4]), array2img(dict_pre_SO2)]) 
                write_images(dict_cmp_vessel, test_dir('compare_vessel'))
                write_images(dict_cmp_vein, test_dir('compare_vein'))
                write_images(dict_cmp_artery, test_dir('compare_artery'))
                write_images(dict_cmp_SO2, test_dir('compare_SO2'))
                    
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
                    from Ailib.analyze.common import plot_evalresult, plot_evalpixel, eval_perimg
                    
                    dict_gro_vessel = load_npy( test_dir('npy_gro_vessel'))
                    dict_gro_vein = load_npy( test_dir('npy_gro_vein'))
                    dict_gro_artery = load_npy( test_dir('npy_gro_artery'))
                    
                    dict_pre_vessel = load_npy(test_dir('npy_vessel'))
                    dict_pre_vein = load_npy(test_dir('npy_vein'))
                    dict_pre_artery = load_npy(test_dir('npy_artery'))
                     
                    plot_evalpixel(dict_gro_vessel, dict_pre_vessel, eval_dir('vessel'))
                    plot_evalpixel(dict_gro_vein, dict_pre_vein, eval_dir('vein'))
                    plot_evalpixel(dict_gro_artery, dict_pre_artery, eval_dir('artery'))
                    
                    

