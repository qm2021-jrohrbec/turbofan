from tracemalloc import start
import numpy as np
import pandas as pd

from rul_dataframe import RUL_DataFrame

def getTFDataset(path = "dataset", set = 1):
    """
    Function for loading the NASA Turbofan Dataset. 
    Returns pandas dataframes train, test(, rul)
    """

    colnames = ['id', 'dt',
                'set1', 'set2', 'set3',
                's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 
                's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
        
    train = pd.read_table(f'{path}/train_FD00{set}.txt', header=None, delim_whitespace=True)
    test = pd.read_table(f'{path}/test_FD00{set}.txt', header=None, delim_whitespace=True)

    train.columns = colnames
    test.columns = colnames
        
    rul = pd.read_table(f'{path}/RUL_FD00{set}.txt', header=None, delim_whitespace=True)
    rul.columns = ['rul'] # rul_col
            
    return train, test, rul

def addTFlinear(train, test, rul):
    func_help = lambda c: c[::-1]
    train['linear'] = train[['id', 'dt']].groupby('id').transform(func_help) - 1
    test['linear'] = test[['id', 'dt']].groupby('id').transform(func_help) - 1
    for j in test['id'].unique():
        test.loc[test['id'] == j, 'linear'] = test.loc[test['id'] == j, 'linear'] + rul.loc[j - 1, 'rul']

def add_label(train, label, name):
    train[name] = label

# For PyTorch
import torch
from torch.utils.data import Dataset
from typing import Union

class TorchRULIterable():

    def __init__(self, rul_df, label, start = 1) -> None:
        self.current_id = start
        self.last_id = rul_df.df[rul_df.id_col].iloc[-1]
        self.label = label
        self.rul_df = rul_df

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_id <= self.last_id:
            x = self.rul_df.df.loc[self.rul_df.id_col == self.current_id, self.rul_df.data_cols]
            y = self.rul_df.df.loc[self.rul_df.id_col == self.current_id, self.label]
            self.current_id += 1
            return torch.tensor(x.values.astype(np.float32)), torch.tensor(y.values.astype(np.float32))
        else:
            raise StopIteration

class TorchRULDataset(Dataset):

    def __init__(self, rul_df: RUL_DataFrame, label: str, sequence_length: Union[None, int] = None):
        self.rul_df = rul_df
        self.label = label
        self.seq_l = sequence_length
        self.ids = rul_df.df[rul_df.id_col].unique()

        self.items = []
        for i in self.ids:
            trajectory = rul_df.df.loc[rul_df.df[rul_df.id_col] == i].index
            start = trajectory[0]
            end = trajectory[-1]
            self.items.append([i, start, end])

        self.id_col = rul_df.id_col
        self.len = len(self.ids)
        if self.seq_l:
            # if fixed sequence length
            self.len = 0
            self.items = []
            current_item = 0
            for i in self.ids:    
                trajectory = rul_df.df.loc[rul_df.df[rul_df.id_col] == i].index
                leni = len(trajectory)
                if leni >= self.seq_l:
                    # calculate how many trajectories with seq_l fit in id
                    n = leni - self.seq_l + 1
                    self.len += n
                    #go through trajectories
                    for j in range(n):
                        start = trajectory[j]
                        end = trajectory[j + self.seq_l - 1]
                        current_item += 1
                        self.items.append([current_item, start, end])
                else:
                   raise Exception('selected seqence legth too long')
        else:
            print('Note due to possibly variing sequence lenghts, select batch size 1 (default) in DataLoader')

    def __getitem__(self, item):
        trajectory = self.rul_df.df.iloc[self.items[item - 1][1]:(self.items[item - 1][2] + 1)]
        if self.items[item - 1][0] == item:
            trajectory = self.rul_df.df.iloc[self.items[item - 1][1]:(self.items[item - 1][2] + 1)]
        else:
            for i in range(len(self.items)):
                if self.items[i][0] == item:
                     trajectory = self.rul_df.df.iloc[self.items[i][1]:(self.items[i][2] + 1)]
        # Transform it to Tensor
        x = torch.tensor(trajectory[self.rul_df.data_cols].values)
        y = torch.tensor(trajectory[self.label].values)
        return x.float(), y.float()

    def __len__(self):
        return self.len