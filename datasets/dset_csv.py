# 
# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division
# 
# from torchlib.datasets import BaseDataset
# 
# 
# 
# class CSVDataset(BaseDataset):
# 
#     def __init__(self,
#                  csv,
#                  input_cols=[0],
#                  target_cols=[1],
#                  input_transform=None,
#                  target_transform=None,
#                  co_transform=None):
#         """
#         Initialize a Dataset from a CSV file/dataframe. This does NOT
#         actually load the data into memory if the CSV contains filepaths.
# 
#         Arguments
#         ---------
#         csv : string or pandas.DataFrame
#             if string, should be a path to a .csv file which
#             can be loaded as a pandas dataframe
#         
#         input_cols : int/list of ints, or string/list of strings
#             which columns to use as input arrays.
#             If int(s), should be column indicies
#             If str(s), should be column names 
#         
#         target_cols : int/list of ints, or string/list of strings
#             which columns to use as input arrays.
#             If int(s), should be column indicies
#             If str(s), should be column names 
# 
#         input_transform : class which implements a __call__ method
#             tranform(s) to apply to inputs during runtime loading
# 
#         target_tranform : class which implements a __call__ method
#             transform(s) to apply to targets during runtime loading
# 
#         co_transform : class which implements a __call__ method
#             transform(s) to apply to both inputs and targets simultaneously
#             during runtime loading
#         """
#         self.input_cols = _process_cols_argument(input_cols)
#         self.target_cols = _process_cols_argument(target_cols)
#         
#         self.df = _process_csv_argument(csv)
# 
#         self.inputs = _select_dataframe_columns(self.df, input_cols)
#         self.num_inputs = self.inputs.shape[1]
#         self.input_return_processor = _return_first_element_of_list if self.num_inputs==1 else _pass_through
# 
#         if target_cols is None:
#             self.num_targets = 0
#             self.has_target = False
#         else:
#             self.targets = _select_dataframe_columns(self.df, target_cols)
#             self.num_targets = self.targets.shape[1]
#             self.target_return_processor = _return_first_element_of_list if self.num_targets==1 else _pass_through
#             self.has_target = True
#             self.min_inputs_or_targets = min(self.num_inputs, self.num_targets)
# 
#         self.input_loader = default_file_reader
#         self.target_loader = default_file_reader
#         
#         self.input_transform = _process_transform_argument(input_transform, self.num_inputs)
#         if self.has_target:
#             self.target_transform = _process_transform_argument(target_transform, self.num_targets)
#             self.co_transform = _process_co_transform_argument(co_transform, self.num_inputs, self.num_targets)
# 
#     def __getitem__(self, index):
#         """
#         Index the dataset and return the input + target
#         """
#         input_sample = [self.input_transform[i](self.input_loader(self.inputs[index, i])) for i in range(self.num_inputs)]
# 
#         if self.has_target:
#             target_sample = [self.target_transform[i](self.target_loader(self.targets[index, i])) for i in range(self.num_targets)]
#             for i in range(self.min_inputs_or_targets):
#                 input_sample[i], input_sample[i] = self.co_transform[i](input_sample[i], target_sample[i])
# 
#             return self.input_return_processor(input_sample), self.target_return_processor(target_sample)
#         else:
#             return self.input_return_processor(input_sample)
# 
#     def split_by_column(self, col):
#         """
#         Split this dataset object into multiple dataset objects based on 
#         the unique factors of the given column. The number of returned
#         datasets will be equal to the number of unique values in the given
#         column. The transforms and original dataframe will all be transferred
#         to the new datasets 
# 
#         Useful for splitting a dataset into train/val/test datasets.
# 
#         Arguments
#         ---------
#         col : integer or string
#             which column to split the data on. 
#             if int, should be column index
#             if str, should be column name
# 
#         Returns
#         -------
#         - list of new datasets with transforms copied
#         """
#         if isinstance(col, int):
#             split_vals = self.df.iloc[:,col].values.flatten()
# 
#             new_df_list = []
#             for unique_split_val in np.unique(split_vals):
#                 new_df = self.df[:][self.df.iloc[:,col]==unique_split_val]
#                 new_df_list.append(new_df)
#         elif isinstance(col, str):
#             split_vals = self.df.loc[:,col].values.flatten()
# 
#             new_df_list = []
#             for unique_split_val in np.unique(split_vals):
#                 new_df = self.df[:][self.df.loc[:,col]==unique_split_val]
#                 new_df_list.append(new_df)
#         else:
#             raise ValueError('col argument not valid - must be column name or index')
# 
#         new_datasets = []
#         for new_df in new_df_list:
#             new_dataset = self.copy(new_df)
#             new_datasets.append(new_dataset)
# 
#         return new_datasets
# 
#     def train_test_split(self, train_size):
#         if train_size < 1:
#             train_size = int(train_size * len(self))
# 
#         train_indices = np.random.choice(len(self), train_size, replace=False)
#         test_indices = np.array([i for i in range(len(self)) if i not in train_indices])
#         
#         train_df = self.df.iloc[train_indices,:]
#         test_df = self.df.iloc[test_indices,:]
# 
#         train_dataset = self.copy(train_df)
#         test_dataset = self.copy(test_df)
# 
#         return train_dataset, test_dataset
# 
#     def copy(self, df=None):
#         if df is None:
#             df = self.df
# 
#         return CSVDataset(df,
#                           input_cols=self.input_cols, 
#                           target_cols=self.target_cols,
#                           input_transform=self.input_transform,
#                           target_transform=self.target_transform,
#                           co_transform=self.co_transform)


