import h5py
import torch




def h5_to_dict(file_path):
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
