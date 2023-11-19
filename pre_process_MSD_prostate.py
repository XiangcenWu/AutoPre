import h5py
import torch
from monai.transforms import (
    SpatialCropd,
    Compose,
    RandShiftIntensityd
)
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import Resized, Compose, LoadImaged, Spacingd, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged, CropForegroundd, SpatialCropd, CenterSpatialCropd, SpatialPadd
import json
import nibabel as nib
import os



def convert_label(input_label):
    input_label_5 = (input_label == 5).float()
    input_label_4 = (input_label == 4).float()
    return input_label_5 + input_label_4
class Load_File(object):
    """load the file dir dict and convert to actualy file dir"""
    
    def load_file(self, input_dict):
    
        data_dict = {
            'image': torch.tensor(nib.load(input_dict['image']).get_fdata()), 
            'label': convert_label(torch.tensor(nib.load(input_dict['label']).get_fdata()))
        }
        return data_dict

    def __call__(self, input_dict):
        return self.load_file(input_dict)


class Access(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data_dict):
        return self._access(data_dict)
    
    def _access(self, data_dict):
        return {'image': data_dict['image'][:, :, :, 0].unsqueeze(0), 'label': data_dict['label'].unsqueeze(0)}

data_reader = Compose(
    [
        Load_File(),
        Access(),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Resized(keys=["image", "label"], spatial_size=(180, 180, 20)),

    ]
)

def convert_label(input_label):
    input_label_5 = (input_label == 1).float()
    input_label_4 = (input_label == 2).float()
    return input_label_5 + input_label_4


json_file = json.load(open('/home/xiangcen/AutoPre/Task05_Prostate/dataset.json'))
data_dir_list = json_file['training']
print(data_dir_list)
print(data_dir_list[0]['image'][-9:-7])



def convert_h5(data_dir_dict_list, des_dir):
    for data_dir in data_dir_dict_list:
        
        image_index = data_dir['image'][-9:-7]
        dir = {
            'image': os.path.join('/home/xiangcen/AutoPre/Task05_Prostate', data_dir['image']),
            'label': os.path.join('/home/xiangcen/AutoPre/Task05_Prostate', data_dir['label'])
        }
        loaded_dict = data_reader(dir)
        with h5py.File(os.path.join(des_dir, image_index + '.h5'), 'w') as hf:
            hf.create_dataset('image', data=loaded_dict['image'])
            hf.create_dataset('label', data=(loaded_dict['label'] > 0.5).float())


convert_h5(data_dir_list, '/home/xiangcen/AutoPre/MSD_prostate_h5')
