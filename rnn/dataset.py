import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

    
class ElecDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        data = self.data.iloc[idx:idx + self.seq_len]
        x = torch.from_numpy(np.array(data, dtype=np.float32))
        y = torch.from_numpy(np.array(self.data.iloc[idx + self.seq_len], dtype=np.float32))

        return x, y
    

pmj_df = pd.read_csv("./data/kaggle_data/daily_data/pjm_elec_daily.csv")

pmj_cols = ['AEP', 'COMED', 'DAYTON', 'DOM', 'DUQ', 'FE', 'NI', 'PJME', 'PJMW']

pmj_dataset_dict = {}

for col in pmj_cols:
    data = pmj_df.loc[pmj_df[col] != 0, col]

    pmj_dataset_dict[col] = ElecDataset(data, 120)

    print(len(pmj_dataset_dict[col]))