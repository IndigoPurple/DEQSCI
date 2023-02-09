import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, re, random
from PIL import Image

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
