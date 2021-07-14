import os, random
import numpy as np
import scipy.stats as stats
from network.aider import Aider

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


site = 'GC'    # 'GC', 'GKD', 'NC', 'QKY' 
is_train = True
is_test = True
rate_split = 0.75

batch_size = 32
img_h, img_w = 4, 4
iptlev = 44
outlev = 6
lr = 0.0005
epoch_start = 0
epoch_total = 100
freq_save = 10

wrf_mete = ['no2', 'pressure', 'tempc', 'ua', 'va', 'wa']
wrf_noise = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
  

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


def train(traindir, aider, trainset, validset, epoch_start, epoch_total):
    if not os.path.exists( traindir ):
        os.mkdir( traindir )
    train_dir = lambda file : os.path.join(traindir, file)
    
    for epoch in range(epoch_start, epoch_total//freq_save):
        aider.update()
        
        for iter in range(epoch_total):
            iteration = (iter+epoch_total*epoch)//(epoch_total//freq_save)
            for _, datas in enumerate( trainset ):
                if datas[0].shape[0]!=batch_size:
                    break
                _, inputs, label = adjust( datas )
                loss, preds = aider.train( inputs, label, lr )
                print('epoch(%d),\ttrain loss(%f)'%(iteration, loss.mean()))
            
            # validset
            for _, datas in enumerate( validset ):
                if datas[0].shape[0]!=batch_size:
                    break
                _, inputs, label = adjust( datas )
                loss, _ = aider.test(inputs, label)
                print('epoch(%d),\tvalid loss(%f)'%(iteration, loss.mean()))
            
        aider.save( train_dir('epoch_%d/model.ckpt'%((epoch+1)*freq_save)) )
    

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


if __name__ == '__main__':
    
    datas = []
    for key in wrf_mete + ['doastime', 'maxdoas']:
        datas.append( np.load('Data/%s/%s.npy'%(site, key)) )
    
    num = datas[0].shape[0]
    idx_all = list( range(num) )
    random.shuffle(idx_all)
    idx_train= idx_all[:int(num*rate_split)]
    idx_test = idx_all[int(num*rate_split):]
    
    loader_train = DataLoader(MyDataSet([arr[idx_train] for arr in datas]), batch_size, shuffle=False)
    loader_test = DataLoader(MyDataSet([arr[idx_test] for arr in datas]), batch_size, shuffle=False)
    
    aider = Aider(batch_size, img_h, img_w, iptlev, outlev)
    train('train_%s'%site, aider, loader_train, loader_test, epoch_start, epoch_total)
    
    
#     aider = Aider(batch_size, img_h, img_w, iptlev, outlev)
#     aider.load( 'train_%s/epoch_%d/model.ckpt'%(site, epoch_total) )
    list_noise = [0, 0, 0, 0, 0, 0]
    arr_time, arr_label, arr_pred = test(aider, loader_test)
    print('predict:', arr_time.shape, arr_label.shape, arr_pred.shape)
    
    for level in range(arr_label.shape[-1]):
        p = stats.pearsonr(arr_label[:,level], arr_pred[:,level])[0]
        print( 'level(%d):'%level, 'pearsonr:', p )


