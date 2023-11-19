import h5py
import torch
from monai.transforms import (
    SpatialCropd,
    Compose,
    RandShiftIntensityd
)
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import Compose, LoadImaged, Spacingd, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged, CropForegroundd, SpatialCropd, CenterSpatialCropd, SpatialPadd




data_reader = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(2.0, 2.0, 3.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-125,
            a_max=125,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        CenterSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128)),
        SpatialCropd(keys=["image", "label"], roi_center=(50, 74, 80) , roi_size=(96, 96, 96)),
        SpatialPadd(keys=['image', 'label'], spatial_size=(96, 96, 96))
    ]
)



import os

# img_directory = '../amos22/imagesTr'
# label_directory = '../amos22/labelsTr'

def convert_h5(img_dir, label_dir, des_dir):
    for img_name in os.listdir(img_directory):
        # make sure all data are CT data
        if img_name.endswith('.nii.gz') and int(img_name[-4:]) > 500:
            img_dir = os.path.join(img_directory, img_name)
            label_dir = os.path.join(label_directory, img_name)

            dir_dict = {
                'image' : img_dir,
                'label' : label_dir
            }

            data = data_reader(dir_dict)
            print(data['image'].numpy().shape)
            with h5py.File(os.path.join(des_dir, img_name+'.h5'), 'w') as hf:
                hf.create_dataset('image', data=data['image'].numpy())
                hf.create_dataset('label', data=data['label'].numpy())




img_directory = '../amos22/imagesTr'
label_directory = '../amos22/labelsTr'
convert_h5(img_directory, label_directory, '../h5_all_label')



img_directory = '../amos22/imagesVa'
label_directory = '../amos22/labelsVa'
convert_h5(img_directory, label_directory, '../h5_all_label')