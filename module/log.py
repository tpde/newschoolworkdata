from ..io import readexcel, save2excel
from ..analyze import graphic

class Loger(object):
    def __init__(self, list_key=[], is_epoch=True):
        self.list_key = list_key
        self.epochlog_train = {key:[] for key in list_key}
        self.batchlog_train = {key:[] for key in list_key}
        self.epochlog_valid = {key:[] for key in list_key}
        self.batchlog_valid = {key:[] for key in list_key}
        self.is_epoch = is_epoch
    
    def update_batchlog(self, is_valid, *input):
        if is_valid:
            for i,key in enumerate(self.list_key):
                self.batchlog_valid[key].append(input[i])
        else:
            for i,key in enumerate(self.list_key):
                self.batchlog_train[key].append(input[i])
    
    def update_epochlog(self, ):
        if not self.is_epoch:
            self.epochlog_train = self.batchlog_train
        else:
            for key in self.list_key:
                if len(self.batchlog_train[key])>0:
                    self.epochlog_train[key].append( sum(self.batchlog_train[key])/len(self.batchlog_train[key]) )
                    self.batchlog_train[key] = []
                    
        for key in self.list_key:
            if len(self.batchlog_valid[key])>0:
                self.epochlog_valid[key].append( sum(self.batchlog_valid[key])/len(self.batchlog_valid[key]) )
                self.batchlog_valid[key] = []
        self.print_log()
            
    def load_log(self, logfile_train, logfile_valid):
        self.epochlog_train = readexcel(logfile_train,'dictlist','h','float')
        self.epochlog_valid = readexcel(logfile_valid,'dictlist','h','float')
        if self.is_epoch:
            self.batchlog_train = {key:[] for key in self.list_key}
            self.batchlog_valid = {key:[] for key in self.list_key}
        else:
            self.batchlog_train = self.epochlog_train
            self.batchlog_valid = self.epochlog_valid
        
    def save_log(self, logfile_train, logfile_valid):
        save2excel(self.epochlog_train, logfile_train)
        save2excel(self.epochlog_valid, logfile_valid)
            
    def print_log(self, ): 
        if self.is_epoch:
            print('epoch:%d'%len(self.epochlog_train[self.list_key[0]]))
        else:
            print('Iter:%d'%len(self.epochlog_train[self.list_key[0]]))
            
        dict_log = {key:self.epochlog_train[key][-1] if len(self.epochlog_train[key])>0 else None for key in self.list_key}
        print('log_train:', dict_log)
        
        dict_log = {key:self.epochlog_valid[key][-1] if len(self.epochlog_valid[key])>0 else None for key in self.list_key}
        print('log_valid:', dict_log)
        
    def is_updateweight(self, key, is_min=True):
        if is_min:
            return min(self.epochlog_valid[key]) >= self.epochlog_valid[key][-1]
        else:
            return max(self.epochlog_valid[key]) <= self.epochlog_valid[key][-1]
    
    def get_epochlog(self, is_valid, key):
        if is_valid:
            return self.epochlog_valid[key]            
        else:
            return self.epochlog_train[key]
        
    def plot_log(self, list_key, title, xtick, ytick, filepath, mode='both'):
        if mode=='train':
            dict_curves = {'train_'+key:self.epochlog_train[key] for key in list_key}
        elif mode=='valid':
            dict_curves = {'valid_'+key:self.epochlog_valid[key] for key in list_key}
        else:
            dict_curves = {'train_'+key:self.epochlog_train[key] for key in list_key}
            dict_valid = {'valid_'+key:self.epochlog_valid[key] for key in list_key}
            dict_curves.update( dict_valid )
        graphic.plot_curves( dict_curves, title, xtick, ytick, filepath, False)
        
    

        