from torch.utils.data import Dataset, DataLoader

class MyDataSet(Dataset):
    def __init__(self, datas ):
        self.datas = datas
            
    def __getitem__(self, index):
        return [data[index] for data in self.datas]
 
    def __len__(self):
        return len(self.datas[0])
    
# loader_train= DataLoader(MyDataSet([trX, trY]), batch_size, shuffle=False)
# for _, datas in enumerate( loader_train ):


# class DataLoader(object):
#  
#     def __init__(self, dataset, batch_size, is_shuffle=False):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.is_shuffle = is_shuffle
#         self.dataset.reset(is_shuffle) 
# 
#     def __len__(self):
#         return (len(self.dataset) + self.batch_size - 1) // self.batch_size
# 
#     def __next__(self):
#         batch_data = self.dataset.next_batch(self.batch_size, self.is_shuffle)
#         return batch_data[1][0], batch_data[2][0]
#         
#     def __iter__(self):
#         return self







