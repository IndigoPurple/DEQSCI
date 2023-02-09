import torch
import h5py
import random
import numpy as np
import os
from PIL import Image
from torchvision import transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True, mode='S'):
        super(Dataset, self).__init__()
        self.train = train
        self.mode = mode
        self.data_loc = '/share/data/vision-greg2/users/gilton/train.h5'
        self.val_loc = '/share/data/vision-greg2/users/gilton/val.h5'
        if self.train:
            if self.mode == 'S':
                h5f = h5py.File(self.data_loc, 'r')
            elif self.mode == 'B':
                h5f = h5py.File('train_B.h5', 'r')
        else:
            if self.mode == 'S':
                h5f = h5py.File(self.val_loc, 'r')
            elif self.mode == 'B':
                h5f = h5py.File('val_B.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            if self.mode == 'S':
                h5f = h5py.File(self.data_loc, 'r')
            elif self.mode == 'B':
                h5f = h5py.File('train_B.h5', 'r')
            # h5f = h5py.File('train.h5', 'r')
        else:
            if self.mode == 'S':
                h5f = h5py.File(self.val_loc, 'r')
            elif self.mode == 'B':
                h5f = h5py.File('val_B.h5', 'r')
            # h5f = h5py.File('val.h5', 'r')

        key = self.keys[index]
        #scale from -1 to 1
        data = 2*np.array(h5f[key]) - 1
        h5f.close()
        return torch.Tensor(data)

def directory_filelist(target_directory):
    file_list = [f for f in sorted(os.listdir(target_directory))
                 if os.path.isfile(os.path.join(target_directory, f))]
    file_list = list(file_list)
    file_list = [f for f in file_list if not f.startswith('.')]
    return file_list

def load_img(file_name):
    with open(file_name,'rb') as f:
        img = Image.open(f).convert("L")
    return img

class EquilibriumDataset(torch.utils.data.Dataset):
    def __init__(self, target_directory, init_directory, validation_data=False, transform=None):
        super(EquilibriumDataset, self).__init__()
        filelist = directory_filelist(target_directory)
        training_data = filelist

        self.full_filelist = [target_directory + single_file for single_file in training_data]
        self.init_directory = init_directory

        self.transform = transform
        self.options = ['_1.png','_2.png','_3.png','_4.png']

    def __len__(self):
        return len(self.full_filelist)

    def convert_to_2d(self, x):
        return torch.cat((x, torch.zeros_like(x)), dim=0)

    def __getitem__(self, item):
        image_name = self.full_filelist[item]
        # image_name = "/Users/dgilton/Documents/MATLAB/prDeep-master/train/test_001.png"
        data = load_img(image_name)
        if self.transform is not None:
            data = self.transform(data)
        data = 2.0*data - 1.0
        data = self.convert_to_2d(data)

        random_choice = random.choice(self.options)
        initial_point_filename = os.path.splitext(os.path.split(image_name)[1])[0] + random_choice
        initial_point = load_img(self.init_directory + initial_point_filename)
        if self.transform is not None:
            initial_point = self.transform(initial_point)
        initial_point = 2.0 * initial_point - 1.0
        initial_point = self.convert_to_2d(initial_point)

        return data, initial_point

if __name__=="__main__":
    dataset_folder = "/Users/dgilton/PycharmProjects/provableplaying/training/data/train/"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = EquilibriumDataset(dataset_folder, transform=transform)
    print(dataset[0].shape)