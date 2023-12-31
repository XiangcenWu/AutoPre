import torch
import numpy as np
from monai.data import (
    Dataset,
    DataLoader,
)
from monai.transforms import (
    Compose
)

import h5py
label_number = 10




def h5_to_dict(file_path, label_num=10):
    h5f = h5py.File(file_path,'r')
    data_dict = {
        'image': torch.from_numpy(h5f['image'][:]), 
        'label': torch.from_numpy(h5f['label'][:])
    }
    h5f.close()
    return data_dict

class ReadH5d(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, file_path):
        return h5_to_dict(file_path)


transform = Compose([
    ReadH5d()
])


def create_data_loader(data_list, batch_size, drop_last=True, shuffle=True):
    set = Dataset(data_list, transform)
    return DataLoader(set, num_workers=8, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
