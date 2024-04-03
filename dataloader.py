import torch
import numpy as np
import FinanceDataReader as fdr

from torch import Tensor
from utils import min_max_scale_each_sample
from torch.utils.data import DataLoader, Dataset



class Stock(Dataset):
    def __init__(self, 
                 symbols : str | list[str], 
                 sdate : str | int, 
                 edate : str | int,
                 window : int):
        df = fdr.DataReader(symbols, sdate, edate)
        self.data = self.generate_stock_sample(df, window)
        
    def generate_stock_sample(self, df, window) -> Tensor:
        df = torch.from_numpy(df.to_numpy())
        sample_n = len(df)-window+1
        data = torch.zeros(sample_n, window, df.shape[-1], dtype=torch.float64)
        for i in range(sample_n):
            start = i
            end = i + window
            data[i, :, :] = df[start:end]
        
        data, _, _ = min_max_scale_each_sample(data)
        data = data.float()
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class Sine(Dataset):
    def __init__(self, 
                 sample_n : int, 
                 seq_len : int, 
                 feature_n : int):
        '''
        sample_n : number of samples
        seq_len : sequence length
        feature_n : number of features
        '''
        self.data = self.generate_sine_ts(sample_n, seq_len, feature_n)
        
    def generate_sine_ts(self, sample_n, seq_len, feature_n) -> Tensor:
        data = list()
        for i in range(sample_n):
            sines = list()      
            for j in range(feature_n):
                freq = np.random.uniform(0, 0.1)            
                phase = np.random.uniform(0, 0.1)
                sine_sequence = [np.sin(freq * k + phase) for k in range(seq_len)] 
                sines.append(sine_sequence)
                
            sines = np.array(sines).T # align shape to (sequence, feature)
            data.append(sines)
        
        data = np.array(data)
        data = torch.from_numpy(data).float()
        data, _, _ = min_max_scale_each_sample(data)

        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):        
        return self.data[idx]


class SyntheticTS(Dataset):
    def __init__(self, 
                 sample_n : int, 
                 seq_len : int, 
                 freq : int, 
                 noise_level : float, 
                 trend : bool):
        """
        Generates synthetic time series data.
        
        Parameters:
        - num_samples: Number of time series samples to generate.
        - seq_length: Length of each time series.
        - freq: Frequency of the sine wave.
        - noise_level: Standard deviation of Gaussian noise added to the data.
        - trend: If True, adds a linear trend to the data.
        
        Returns:
        - data: Generated synthetic time series data of shape (num_samples, seq_length).
        """
        self.data = self.generate_synthetic_ts(sample_n, seq_len, freq, noise_level, trend)
        
    def generate_synthetic_ts(self, sample_n, seq_len, freq, noise_level, trend) -> Tensor:
        time = np.linspace(0, 2 * np.pi, seq_len)
        sine_wave = np.sin(freq * time)
        
        if trend:
            trend = np.linspace(0, 1, seq_len)
            sine_wave += trend
        
        data = np.zeros((sample_n, seq_len))
        for i in range(sample_n):
            noise = np.random.normal(0, noise_level, seq_len)
            data[i, :] = sine_wave + noise
        
        data = torch.from_numpy(data).float()
        data, _, _ = min_max_scale_each_sample(data)
        
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx]
        
        return X
    

class DataloaderHandler():
    def __init__(self, 
                 batch_size : int = 128, 
                 shuffle : bool = True, 
                 drop_last : bool = True) -> None:
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def synthetic_dataloader(self, 
                             sample_n = 10000, 
                             seq_len = 24, 
                             freq = 1, 
                             noise_level = 0.1, 
                             trend = True):

        dataset = SyntheticTS(sample_n, seq_len, freq, noise_level, trend)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                drop_last=self.drop_last)
        
        return dataloader
    
    def sine_dataloader(self, 
                        sample_n = 10000, 
                        seq_len = 24, 
                        feature_n = 5):
        
        dataset = Sine(sample_n, seq_len, feature_n)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                drop_last=self.drop_last)
        
        return dataloader
    
    def stock_dataloader(self, 
                         symbols=["AAPL", "GOOG"], 
                         sdate="2010", 
                         edate=None,
                         window=24):
        
        dataset = Stock(symbols, sdate, edate, window)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=self.shuffle,
                                drop_last=self.drop_last)
        
        return dataloader
    

