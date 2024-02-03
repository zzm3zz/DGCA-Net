import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy.ndimage.morphology import distance_transform_edt

def random_rot_flip(image, label, boundary):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    boundary = np.rot90(boundary, k)
    axis = np.random.randint(0, 2)
    # numpy.flip(m, axis=None)

    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    boundary = np.flip(boundary, axis=axis).copy()
    return image, label, boundary


def random_rotate(image, label, boundary):
    angle = np.random.randint(-20, 20)
    # 是否将图像根据角度旋转
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    boundary = ndimage.rotate(boundary, angle, order=0, reshape=False)
    return image, label, boundary


class RandomGenerator_b(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, boundary = sample['image'], sample['label'], sample['boundary']

        # 根据0.5的概率确定是否将图像进行旋转和翻转
        if random.random() > 0.5:
            image, label, boundary = random_rot_flip(image, label, boundary)
        elif random.random() > 0.5:
            image, label, boundary = random_rotate(image, label, boundary)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            boundary = zoom(boundary, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        boundary = torch.from_numpy(boundary.astype(np.float32)).unsqueeze(0)
        sample = {'image': image, 'label': label.long(), 'boundary': boundary}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        global edge, boundary
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label, boundary = data['image'], data['label'], data['boundary']
            edge = edge.squeeze(0)
            sample = {'image': image, 'label': label, 'boundary': boundary}
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.nii.gz".format(vol_name)
            image = sitk.ReadImage(filepath)
            image = sitk.GetArrayFromImage(image)
            
            labelpath = filepath.replace('test_nii_image', 'test_nii_label')
            label = sitk.ReadImage(labelpath)
            label = sitk.GetArrayFromImage(label)
            
            sample = {'image': image, 'label': label}
          
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
