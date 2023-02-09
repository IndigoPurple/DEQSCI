import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, re, random
from PIL import Image

# yaping
import h5py
import scipy.io as sio
from scipy.io.matlab.mio import _open_file
from scipy.io.matlab.miobase import get_matfile_version

def swap_patches(batch, index1, index2, h,w, patch_top_loc, patch_left_loc):
    tmp = batch[
       index1,
       patch_top_loc:patch_top_loc+h,
       patch_left_loc:patch_left_loc+w, :].clone()

    batch[
         index1,
         patch_top_loc:patch_top_loc+h,
         patch_left_loc:patch_left_loc+w, :] = batch[
             index2,
             patch_top_loc:patch_top_loc+h,
             patch_left_loc:patch_left_loc+w, :].clone()

    batch[
         index1,
         patch_top_loc:patch_top_loc+h,
         patch_left_loc:patch_left_loc+w, :] = tmp
 
    return batch

   


#TODO this should probably be moved to another file as it's a transform
# Although it needs to operate on the batch as opposed to individual images
def batch_patch_swap(batch, h=None, w=None, ):
    '''
    Args:
        batch: batch x h x w x ch of images of type torch tensor
        h: height of patch to be swapped, if greater than batch.size()[1] then will be capped to that
        w: width of patch, similar capping will be done as height

    '''
    #TODO maybe we specify h and w as a range instead?
    num_images, height, width, _ = batch.size()
    #TODO maybe crop from different locations of the image?
    patch_top_loc = random.randint(1, int((3.0/4)*height))
    patch_left_loc = random.randint(1, int((3.0/4)*width))

    if (not h) or (h >= height) or (h + patch_top_loc >= height):
        h = random.randint(1, height-patch_top_loc)
    if (not w) or (w >= width):
        h = random.randint(1, width-patch_left_loc)

    for index in range(num_images):
        swap_index = random.randint(0, num_images-1)
        batch = swap_patches(batch, index, swap_index, h,w, patch_top_loc, patch_left_loc)

    return batch

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def directory_filelist(target_directory):
    file_list = [f for f in sorted(os.listdir(target_directory))
                 if os.path.isfile(os.path.join(target_directory, f))]
    file_list = list(file_list)
    file_list = [f for f in file_list if not f.startswith('.')]
    return file_list

def load_img(file_name):
    with open(file_name,'rb') as f:
        img = Image.open(f).convert("RGB")
    return img

class FolderDataset(Dataset):
    def __init__(self, target_directory, transform=None):
        filelist = directory_filelist(target_directory)
        self.full_filelist = [target_directory + single_file for single_file in filelist]
        self.transform = transform

    def __len__(self):
        return len(self.full_filelist)

    def __getitem__(self, item):
        image_name = self.full_filelist[item]
        data = load_img(image_name)
        if self.transform is not None:
            data = self.transform(data)
        return data

class CelebaDataset(Dataset):
    def __init__(self, target_directory, validation_data=False, transform=None):
        filelist = directory_filelist(target_directory)
        training_data = filelist[:162770]
        val_data = filelist[162770:182638]
        test_data = filelist[182638:]
        if validation_data:
            self.full_filelist = [target_directory + single_file for single_file in val_data]
        else:
            self.full_filelist = [target_directory + single_file for single_file in training_data]

        self.transform = transform

    def __len__(self):
        return len(self.full_filelist)

    def __getitem__(self, item):
        image_name = self.full_filelist[item]
        data = load_img(image_name)
        if self.transform is not None:
            data = self.transform(data)
        return data

class CelebaTrainingDatasetSubset(Dataset):
    def __init__(self, target_directory, subset_indices, transform=None):
        filelist = directory_filelist(target_directory)
        training_data = filelist[:162770]
        try:
            training_data = [training_data[x] for x in subset_indices]
        except TypeError:
            training_data = [training_data[subset_indices]]

        self.full_filelist = [target_directory + single_file for single_file in training_data]

        self.transform = transform

    def __len__(self):
        return len(self.full_filelist)

    def __getitem__(self, item):
        image_name = self.full_filelist[item]
        data = load_img(image_name)
        if self.transform is not None:
            data = self.transform(data)
        return data

# This can be removed I think it's the same class as the one above at least right now.
class CelebaTestDataset(Dataset):
    def __init__(self, target_directory, transform=None):
        filelist = directory_filelist(target_directory)
        training_data = filelist[:162770]
        val_data = filelist[162770:182638]
        test_data = filelist[182638:]
        self.full_filelist = [target_directory + single_file for single_file in test_data]
        self.transform = transform

    def __len__(self):
        return len(self.full_filelist)

    def __getitem__(self, item):
        image_name = self.full_filelist[item]
        data = load_img(image_name)
        if self.transform is not None:
            data = self.transform(data)
        return data

def load_mat(mask_location, key):
    if get_matfile_version(_open_file(mask_location, appendmat=True)[0])[0] < 2: # MATLAB .mat v7.2 or lower versions
        file = sio.loadmat(mask_location) # for '-v7.2' and lower version of .mat file (MATLAB)
        # print(file)
        if key is 'gt':
            if "patch_save" in file:
                file = torch.from_numpy(file['patch_save'] / 255)
            elif "p1" in file:
                file = torch.from_numpy(file['p1'] / 255)
            elif "p2" in file:
                file = torch.from_numpy(file['p2'] / 255)
            elif "p3" in file:
                file = torch.from_numpy(file['p3'] / 255)
            # file = file.permute(2, 0, 1)
        elif key is 'meas':
            # file = torch.from_numpy(file['meas'] / 255).unsqueeze(0)
            file = torch.from_numpy(file['meas'] / 255)
        elif key is 'mask':
            # file = torch.from_numpy(file['mask']).permute(2,0,1)
            file = torch.from_numpy(file['mask'])
        else:
            print('unknown key')
        order = 'K' # [order] keep as the default order in Python/numpy
        data = np.float32(file)

    else: # MATLAB .mat v7.3
        file =  h5py.File(mask_location, 'r')  # for '-v7.3' .mat file (MATLAB)
        # print(file)
        if key is 'gt':
            if "patch_save" in file:
                file = torch.from_numpy(file['patch_save'] / 255)
            elif "p1" in file:
                file = torch.from_numpy(file['p1'] / 255)
            elif "p2" in file:
                file = torch.from_numpy(file['p2'] / 255)
            elif "p3" in file:
                file = torch.from_numpy(file['p3'] / 255)
            # file = file.permute(2, 0, 1)
        elif key is 'meas':
            # file = torch.from_numpy(file['meas'] / 255).unsqueeze(0)
            file = torch.from_numpy(file['meas'] / 255)
        elif key is 'mask':
            # file = torch.from_numpy(file['mask']).permute(2,0,1)
            file = torch.from_numpy(file['mask'])
        else:
            print('unknown key')
        order = 'F' # [order] switch to MATLAB array order
        data = np.float32(file, order=order).transpose()

    # print(data.shape)
    # data = torch.from_numpy(data)
    return data



class SCITrainingDatasetSubset(Dataset):
    def __init__(self, gt_directory, meas_directory, mask_location):
        training_data = directory_filelist(gt_directory)
        # training_data = training_data[:10]

        self.full_gt_filelist = [gt_directory + single_file for single_file in training_data]
        self.full_meas_filelist = [meas_directory + single_file for single_file in training_data]

        self.mask = load_mat(mask_location, 'mask')

    def __len__(self):
        return len(self.full_gt_filelist)

    def __getitem__(self, item):
        gt_image_name = self.full_gt_filelist[item]
        meas_image_name = self.full_meas_filelist[item]
        gt = load_mat(gt_image_name, 'gt')
        meas = load_mat(meas_image_name,'meas')
        data = {'gt': gt,
                     'mask': self.mask,
                     'meas': meas}
        return data

def load_test_data(matfile):
    # [1] load data
    if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2:  # MATLAB .mat v7.2 or lower versions
        file = sio.loadmat(matfile)  # for '-v7.2' and lower version of .mat file (MATLAB)
        order = 'K'  # [order] keep as the default order in Python/numpy
        meas = np.float32(file['meas'])
        mask = np.float32(file['mask'])
        orig = np.float32(file['orig'])
    else:  # MATLAB .mat v7.3
        file = h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
        order = 'F'  # [order] switch to MATLAB array order
        meas = np.float32(file['meas'], order=order).transpose()
        mask = np.float32(file['mask'], order=order).transpose()
        orig = np.float32(file['orig'], order=order).transpose()
    data = {'gt': orig / 255,
            'mask': mask,
            'meas': meas/255}
    return data

class SCITestDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.filelist = directory_filelist(dir)
        # self.full_filelist = [dir + single_file for single_file in filelist]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        mat_name = self.dir + self.filelist[item]

        data = load_test_data(mat_name)
        data['file'] = self.filelist[item]
        return data