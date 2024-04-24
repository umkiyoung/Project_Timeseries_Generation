import os
import torch
import numpy as np
import FinanceDataReader as fdr

from torch import Tensor
from tqdm.auto import tqdm
from torch.utils.data import Dataset

        
class SineDataset(Dataset):
    def __init__(self, 
                 n_samples=10000, 
                 window=24, 
                 feature_dim=5, 
                 save_ground_truth=True, 
                 seed=2024,
                 period='train',
                 ):
        super().__init__()
        self.dir = './output/ground_truth'
        os.makedirs(self.dir, exist_ok=True)
        self.window = window
        self.feature_dim = feature_dim
        self.data = self.sine_data_generation(n_samples=n_samples, 
                                              window=window, 
                                              feature_dim=feature_dim, 
                                              seed=seed)
        if save_ground_truth:
            np.save(os.path.join(self.dir, f"sine_ground_truth_{window}_{period}.npy"), self.data)

    def sine_data_generation(self,
                             n_samples : int, 
                             window : int, 
                             feature_dim : int, 
                             seed : int, 
                             ):
        np.random.seed(seed)
        sine_data = list()
        for _ in tqdm(range(n_samples), total=n_samples, desc="Sampling sine-dataset"):
            sine = list()
            for _ in range(feature_dim):
                freq, phase = np.random.uniform(0, 0.1, 2)            
                sine_sequence = [np.sin(freq * j + phase) for j in range(window)]
                sine.append(sine_sequence)
            sine_data.append(np.array(sine).T)  
        sine_data = torch.from_numpy(np.array(sine_data)).float()
        
        return sine_data
  
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class StockDataset(Dataset):
    def __init__(self, 
                 symbol : str = "AAPL", 
                 sdate : str = "2000", 
                 edate : str = "2024",
                 window : int = 24,
                 save_ground_truth=True, 
                 normalize=True,
                 period='train',
                 ):
        
        raw_df = fdr.DataReader(symbol, sdate, edate)
        self.data = self.generate_stock_sample(raw_df, window)
        self.window = window
        self.feature_dim = self.data.shape[-1]
        self.dir = './output/ground_truth'
        os.makedirs(self.dir, exist_ok=True)
        
        self.normalize = normalize
        if self.normalize:
            self.data, self.min_val, self.max_val = self._min_max_scale(self.data)
        
        if save_ground_truth:
            np.save(os.path.join(self.dir, f"stock_ground_truth_{window}_{period}.npy"), self.data)
            
    def generate_stock_sample(self, df, window) -> Tensor:
        raw_data = torch.from_numpy(df.to_numpy()).float()
        data = self._extract_sliding_windows(raw_data, window)
    
        return data
    
    def _extract_sliding_windows(self, raw_data, window) -> Tensor:
        sample_n = len(raw_data)-window+1
        data = torch.zeros(sample_n, window, raw_data.shape[-1])
        for i in range(sample_n):
            start = i
            end = i + window    
            data[i, :, :] = raw_data[start:end]
            
        return data

    def _min_max_scale(self, data : Tensor):
        min_val = data.min(dim=1, keepdim=True)[0]
        max_val = data.max(dim=1, keepdim=True)[0]
        scaled_data = (data-min_val)/(max_val - min_val)
        
        return scaled_data, min_val, max_val
    
    def _inverse_min_max_scale(scaled_data : Tensor,
                               min_val : Tensor, 
                               max_val : Tensor):
        origin_data = scaled_data*(max_val-min_val)+ min_val

        return origin_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class SyntheticDataset(Dataset):
    def __init__(self, 
                 n_samples : int = 10000, 
                 window : int = 24, 
                 feature_dim : int = 5,
                 noise_level : float = 2, 
                 save_ground_truth=True, 
                 normalize=True,
                 seed=2024,
                 period='train',
                 ):
        self.window = window
        self.feature_dim = feature_dim
        self.dir = './output/ground_truth'
        os.makedirs(self.dir, exist_ok=True)
        self.data = self.generate_synthetic_ts(n_samples, 
                                               window,
                                               feature_dim,
                                               noise_level,
                                               seed)
        self.normalize = normalize
        if self.normalize:
            self.data, self.min_val, self.max_val = self._min_max_scale(self.data)
        
        if save_ground_truth:
            np.save(os.path.join(self.dir, f"synthetic_ground_truth_{window}_{period}.npy"), self.data)

    def generate_synthetic_ts(self, 
                              n_samples,
                              window,
                              feature_dim,
                              noise_level,
                              seed
                              ) -> Tensor:
        np.random.seed(seed)
        syn_data = list()
        for _ in tqdm(range(n_samples), total=n_samples, desc="Sampling synthetic-dataset"):
            synthetic = list()
            for _ in range(feature_dim):
                freq, phase = np.random.uniform(0, 3, 2)            
                linear_trend = np.linspace(0, np.random.uniform(-6,6), window)
                seasonal = [np.sin(freq * j + phase) for j in range(window)]
                noise = np.random.normal(0, noise_level, window)
                synthetic.append(linear_trend+seasonal+noise)
            syn_data.append(np.array(synthetic).T)    
        syn_data = torch.from_numpy(np.array(syn_data)).float()

        return syn_data
    
    def _min_max_scale(self, data : Tensor):
        min_val = data.min(dim=1, keepdim=True)[0]
        max_val = data.max(dim=1, keepdim=True)[0]
        scaled_data = (data-min_val)/(max_val - min_val)
        
        return scaled_data, min_val, max_val
    
    def _inverse_min_max_scale(scaled_data : Tensor,
                            min_val : Tensor, 
                            max_val : Tensor):
        origin_data = scaled_data*(max_val-min_val)+ min_val

        return origin_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    