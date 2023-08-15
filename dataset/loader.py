import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import config.config as config

# history_size
history_size = config.history_size

class DatasetFromCSV(IterableDataset):
    def __init__(self, train=False):
        if train:
            if config.target_domain == 'movie':
                self.file_path = 'dataset/amazon_data/movie_train.csv'
            self.length = 313472
                
        else:
            raise ValueError("incorrect dataset")

    def __len__(self):
        return self.length

    def __iter__(self):
        with open(self.file_path, 'r',encoding = "utf-8") as file_obj:
            for line in file_obj:
                line_data = line.strip('\n').split(',')
                line_data = torch.from_numpy(np.array(line_data, dtype='int'))
                yield line_data[history_size + 2], line_data[:history_size], line_data[history_size], line_data[history_size + 3:history_size + history_size + 3], line_data[history_size * 3 + 3:history_size * 4 + 3], line_data[history_size + 1].float()

class TestDataset(IterableDataset):
    def __init__(self):
        if config.target_domain == 'movie':
            self.file_path = 'dataset/amazon_data/movie_test.csv'
        self.length = 3915064

    def __len__(self):
        return self.length

    def __iter__(self):
        with open(self.file_path, 'r') as file_obj:
            next(file_obj)
            for line in file_obj:
                line_data = line.strip('\n').split(',')
                line_data = torch.from_numpy(np.array(line_data, dtype='int'))
                yield line_data[history_size+2], line_data[:history_size], line_data[history_size], line_data[history_size+3:history_size*2 + 3], line_data[history_size*3+3:history_size*4 + 3], line_data[history_size+1].float(), line_data[history_size+2].tolist()

def get_dataloader(batch_size, mode):
    '''
    input:
        batch_size: size of batch
        mode: string, 'train' or 'test' or 'validation'

    Return:
        dataloader
    '''
    dataloader = None
    if mode == 'train':
        dataset = DatasetFromCSV(train=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, prefetch_factor=1,shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)
    elif mode == 'test' or mode == 'val':
        dataset = TestDataset()
        dataloader = DataLoader(dataset, batch_size=batch_size, prefetch_factor=1, shuffle=False, pin_memory=True,num_workers=1, drop_last=False)
    
    return dataloader

def load_pretrained_embedding():
    '''
    Returns:
        query_embedding(np.array): (number of data, embedding_size)
        item_embedding(np.array): (number of data, embedding_size)
    '''
    if config.target_domain == 'movie':
        target_embedding_matrix = np.load('./dataset/amazon_data/movie_item.npy', allow_pickle=True)
        source_embedding_matrix = np.load('./dataset/amazon_data/book_item.npy', allow_pickle=True)

    return torch.tensor(target_embedding_matrix).float(), torch.tensor(source_embedding_matrix).float()

def load_corresponding_iv():
    if config.target_domain == 'movie':
        target_iv = np.load('./dataset/amazon_data/movieIV.npy', allow_pickle=True)
        pinv_target_iv = np.load('./dataset/amazon_data/IV_pinv_movieIV.npy', allow_pickle=True)
        source_iv = np.load('./dataset/amazon_data/bookIV.npy', allow_pickle=True)
        pinv_source_iv = np.load('./dataset/amazon_data/IV_pinv_bookIV.npy', allow_pickle=True)
    target_iv = torch.tensor(target_iv).float()
    target_iv = target_iv.flatten(start_dim=1)
    pinv_target_iv = torch.tensor(pinv_target_iv).float()
    pinv_target_iv = pinv_target_iv.flatten(start_dim=1)
    source_iv = torch.tensor(source_iv).float()
    source_iv = source_iv.flatten(start_dim=1)
    pinv_cource_iv = torch.tensor(pinv_source_iv).float()
    pinv_cource_iv = pinv_cource_iv.flatten(start_dim=1)

    return target_iv, pinv_target_iv, source_iv, pinv_cource_iv

def load_pretrained_user():
    # pass
    user_embedding_matrix = np.load('./dataset/amazon_data/user_embedding.npy', allow_pickle=True)

    return torch.tensor(user_embedding_matrix).float()
