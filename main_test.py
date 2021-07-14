import os
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from network.aider import Aider

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


list_stest = ['GC', 'GKD', 'NC', 'QKY']
dict_P  = {s:{} for s in list_stest}
dict_spm= {s:{} for s in list_stest}
dict_MAE ={s:{} for s in list_stest}
dict_RMSE={s:{} for s in list_stest}

batch_size = 32
img_h, img_w = 4, 4
iptlev = 44
outlev = 6
epoch_total = 100
freq_save = 10

wrf_mete = ['no2', 'pressure', 'tempc', 'ua', 'va', 'wa']
wrf_noise = [0, 0, 0, 0, 0, 0]

def adjust( datas ):
    datas = [np.array(v) for v in datas]
    arr_tenv = np.zeros((datas[0].shape[0], 24))
    times = datas[-2].tolist()
    for b in range( datas[0].shape[0] ):
        arr_tenv[b, int(times[b][8:10])] = 1
    
    inputs = []
    for i, atm in enumerate(wrf_mete):
        arr = datas[i][...,:iptlev]
        # print(atm, arr.shape, np.min(arr), np.max(arr))
        inputs.append( arr + np.random.normal(0, wrf_noise[i], size=arr.shape) )
    inputs.append( arr_tenv )
    label = datas[-1]/1E12
    return times, inputs, label


from torch.utils.data import Dataset, DataLoader

class MyDataSet(Dataset):
    def __init__(self, datas ):
        self.datas = datas
            
    def __getitem__(self, index):
        return [data[index] for data in self.datas]
 
    def __len__(self):
        return len(self.datas[0])


def test(aider, testset):
    list_times = []
    list_label = []
    list_preds = []
    for _, datas in enumerate( testset ):
        if datas[0].shape[0]!=batch_size:
            break
        times, inputs, label = adjust( datas )
        loss, preds = aider.test(inputs, label)
        
        list_times.append( times )
        list_label.append( label )
        list_preds.append( preds )
        
    arr_times = np.concatenate(list_times, axis=0)
    arr_label = np.concatenate(list_label, axis=0)
    arr_preds = np.concatenate(list_preds, axis=0)
    return arr_times, arr_label, arr_preds


def evaluate(stest, arr_gro, arr_pre):
    for lev in range(6):
        dict_P[stest]['P_%d'%(lev+1)] = stats.pearsonr( arr_gro[:,lev], arr_pre[:,lev] )[0]
        dict_spm[stest]['spm_%d'%(lev+1)] = stats.spearmanr( arr_gro[:,lev], arr_pre[:,lev] )[0]
        dict_MAE[stest]['MAE_%d'%(lev+1)] = mean_absolute_error( arr_gro[:,lev], arr_pre[:,lev])
#         dict_MAE[stest]['MAE_%d'%(lev+1)] = np.mean(np.absolute(arr_gro[:,lev], arr_pre[:,lev]))
        dict_RMSE[stest]['RMSE_%d'%(lev+1)] = np.sqrt( mean_squared_error(arr_gro[:,lev], arr_pre[:,lev]) )
#         dict_RMSE[stest]['RMSE_%d'%(lev+1)] = np.sqrt( np.mean(np.square(arr_gro[:,lev], arr_pre[:,lev])) )

    dict_P[stest]['P_Avg'] = np.mean( [dict_P[stest]['P_%d'%(lev+1)] for lev in range(6)] )
    dict_spm[stest]['spm_Avg'] = np.mean( [dict_spm[stest]['spm_%d'%(lev+1)] for lev in range(6)] )
    dict_MAE[stest]['MAE_Avg'] = np.mean( [dict_MAE[stest]['MAE_%d'%(lev+1)] for lev in range(6)] )
    dict_RMSE[stest]['RMSE_Avg'] = np.mean( [dict_RMSE[stest]['RMSE_%d'%(lev+1)] for lev in range(6)] )
    
    list_P3D = []
    for t in range(arr_pre.shape[0]):
        list_P3D.append( stats.pearsonr(arr_gro[t], arr_pre[t])[0] )
    dict_P[stest]['P_3D'] = np.mean( list_P3D )
    
    return dict_P, dict_spm, dict_MAE, dict_RMSE
    
if __name__ == '__main__':
    if False:
        aider = Aider(batch_size, img_h, img_w, iptlev, outlev)
        for stest in list_stest:
            train_sites = [s for s in ['GC', 'GKD', 'NC', 'QKY'] if s!=stest]
            readdir = lambda file:os.path.join('Data/', stest, file)
            list_pre = []
            for strain in train_sites:
                for epoch in range(freq_save, epoch_total+1, freq_save):
                    aider.load( 'train_%s/epoch_%d/model.ckpt'%(strain, epoch) )
                    datas = []
                    for key in wrf_mete + ['doastime', 'maxdoas']:
                        datas.append( np.load(readdir('%s.npy'%key)) )
                    loader_test = DataLoader(MyDataSet(datas), batch_size, shuffle=False)
                    
                    arr_time, arr_label, arr_pred = test(aider, loader_test)
                    print('predict:', arr_time.shape, arr_label.shape, arr_pred.shape)
                    list_pre.append( arr_pred[ : datas[0].shape[0] ] )
            arr_pre = np.mean(list_pre, axis=0)
            if not os.path.exists( 'Prediction' ):
                os.mkdir( 'Prediction' )
            np.save('Prediction/Prediction_'+stest, arr_pre)
        
    
    # ======================== evaluation =========================
    if True:
        eval_NetOrWRF = False #True: evaluate MF-net,   False: evaluate WRF-CHEM
        
        for stest in list_stest:
            if eval_NetOrWRF: # ===== load MF-net prediction ======
                arr_pred = np.load( 'Prediction/Prediction_%s.npy'%stest )
                tail ='MF-net'
            else:   # ===== load WRF-CHEM prediction ======
                arr_wrf = np.load( 'Data/%s/n_no2.npy'%stest )
                list_lev = [0, 7, 10, 11, 12, 13]
#                 arr_wrf = np.load( 'Data/%s/no2.npy'%stest )    #以前单位没统一时错误采用的方法
#                 list_lev = [0, 2, 4, 6, 8, 10]
                
                arr_pred = arr_wrf[:, 1:3, 1:3, list_lev].mean(axis=(1,2))
                tail ='WRF-CHEM'
            print( stest, 'predict', arr_pred.shape)
                
            arr_label = np.load( 'Data/%s/maxdoas.npy'%stest ) / 1E12
            print( stest, 'label', arr_label.shape)
            arr_label = arr_label[:arr_pred.shape[0]]
            dict_P, dict_spm, dict_MAE, dict_RMSE = evaluate(stest, arr_label, arr_pred)
        
        # ========= save the evaluation result to file =========
        with pd.ExcelWriter('eval_%s.xls'%tail) as writer:
            evalres = {key : [dict_P[site][key] for site in list_stest] for key in dict_P[stest]}
            evalres["name"] = list_stest
            df = pd.DataFrame( evalres )
            df.to_excel(writer, 'Pearson')
            
            evalres = {key : [dict_spm[site][key] for site in list_stest] for key in dict_spm[stest]}
            evalres["name"] = list_stest
            df = pd.DataFrame( evalres )
            df.to_excel(writer, 'Spearman')
            
            evalres = {key : [dict_MAE[site][key] for site in list_stest] for key in dict_MAE[stest]}
            evalres["name"] = list_stest
            df = pd.DataFrame( evalres )
            df.to_excel(writer, 'MAE')
            
            evalres = {key : [dict_RMSE[site][key] for site in list_stest] for key in dict_RMSE[stest]}
            evalres["name"] = list_stest
            df = pd.DataFrame( evalres )
            df.to_excel(writer, 'RMSE')

            