
import h5py
import torch
from monai.transforms import (
    SpatialCropd,
    Compose,
    RandShiftIntensityd
)

import os
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import Resized, Compose, LoadImaged, Spacingd, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged, CropForegroundd, SpatialCropd, CenterSpatialCropd, SpatialPadd
import nibabel as nib


import os



# read the insitution.txt file and then 
file1 = open('institution.txt', 'r')
Lines = file1.readlines()
ins_1_data = []
for i in Lines:
    
    
    if int(i[-2]) == 1:
        index = i[:6]
        img_dir = os.path.join('/home/xiangcen/AutoPre/Cimps_data' , index + '_img.nii')
        label_dir = os.path.join('/home/xiangcen/AutoPre/Cimps_data' , index + '_mask.nii')
        ins_1_data.append({'image':img_dir, 'label':label_dir, 'index': index})
print(ins_1_data[0]['image'][-14:-8])








class Load_File(object):
    """load the file dir dict and convert to actualy file dir"""
    
    def load_file(self, input_dict):
    
        data_dict = {
            'image': torch.tensor(nib.load(input_dict['image']).get_fdata()).unsqueeze(0), 
            'label': convert_label(torch.tensor(nib.load(input_dict['label']).get_fdata())).unsqueeze(0)
        }
        return data_dict

    def __call__(self, input_dict):
        return self.load_file(input_dict)


data_reader = Compose(
    [
        Load_File(),

        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 64)),
    ]
)



def convert_label(input_label):
    input_label_5 = (input_label == 5).float()
    input_label_4 = (input_label == 4).float()
    return input_label_5 + input_label_4


def convert_h5(data_dir_dict_list, des_dir):
    for data_dir in data_dir_dict_list:
        
        image_index = data_dir['index']
        loaded_dict = data_reader(data_dir)
        with h5py.File(os.path.join(des_dir, image_index + '.h5'), 'w') as hf:
            hf.create_dataset('image', data=loaded_dict['image'])
            hf.create_dataset('label', data=(loaded_dict['label'] > 0.5).float())



convert_h5(ins_1_data, '/home/xiangcen/AutoPre/Cimps_ins_1_h5')

