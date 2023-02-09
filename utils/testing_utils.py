from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import imageio
from PIL import Image

def save_tensor_as_color_img(img_tensor, filename):
    np_array = img_tensor.cpu().detach().numpy()
    imageio.save(filename, np_array)

def save_batch_as_color_imgs(tensor_batch, batch_size, ii, folder_name, names):
    # img_array = (np.transpose(tensor_batch.cpu().detach().numpy(),(0,2,3,1)) + 1.0) *  127.5
    img_array = (np.clip(np.transpose(tensor_batch.cpu().detach().numpy(),(0,2,3,1)),-1,1) + 1.0) *  127.5
    # img_array = tensor_batch.cpu().detach().numpy()
    # print(np.max(img_array[:]))
    # print(np.min(img_array[:]))

    img_array = img_array.astype(np.uint8)

    for kk in range(batch_size):
        desired_img = Image.fromarray(img_array[kk,...])
        desired_img = desired_img.resize((512,512), resample=Image.NEAREST)
        img_number = batch_size*ii + kk
        filename = folder_name + str(img_number) + "_" + str(names[kk]) + ".png"
        # print(np.shape(img_array))
        # print(filename)
        imageio.imwrite(filename, desired_img)

def save_mri_as_imgs(tensor_batch, batch_size, ii, folder_name, names):
    # img_array = (np.transpose(tensor_batch.cpu().detach().numpy(),(0,2,3,1)) + 1.0) *  127.5

    def rescale_to_01(input):
        batch_size = input.shape[0]
        for bb in range(batch_size):
            flattened_img = torch.flatten(input[bb, ...], start_dim=0)
            img_min = torch.min(flattened_img)
            img_max = torch.max(flattened_img - img_min)
            input[bb, ...] = (input[bb, ...] - img_min) / img_max
        return input
    tensor_batch = torch.norm(tensor_batch, dim=1)
    tensor_batch = rescale_to_01(tensor_batch)

    # img_array = torch.norm(tensor_batch, dim=1).cpu().detach().numpy()
    img_array = tensor_batch.cpu().detach().numpy()

    for kk in range(batch_size):
        img_number = batch_size*ii + kk
        target_img = img_array[kk,...] * 255.0
        target_img = target_img.astype(np.uint8)
        desired_img = Image.fromarray(target_img)
        desired_img = desired_img.resize((512, 512), resample=Image.NEAREST)
        filename = folder_name + str(img_number) + "_" + str(names[kk]) + ".png"
        # plt.imshow(np.sqrt(img_array[kk,0,:,:]**2 + img_array[kk,1,:,:]**2))
        # plt.gray()
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig(filename, bbox_inches='tight')
        imageio.imwrite(filename, desired_img, format="PNG-PIL")
