from PIL import Image
import numpy as np
from skimage.transform import resize
import nibabel as nib
import torch
from torchvision import datasets, transforms
import os
from os import listdir
from os.path import isfile, join

from utils import reshape_nifti
from dataset import HeartData

#       ct_2_x274y0z0.jpg

# add zeros to filenames
path_to_files = "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64_test/"

counter = 0

for f in listdir(path_to_files):
    counter += 1
    print(counter)
    if isfile(join(path_to_files, f)):
        image = Image.open(join(path_to_files, f))
        os.remove(join(path_to_files, f))

        if "ct_10" in f:
            while f[10] != "y":
                f = f.replace("x", "x0")

        else:
            while f[9] != "y":
                f = f.replace("x", "x0")

        image.save(join(path_to_files, f))



# calculate mean and std
# cts = {
#     "ct_2": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_2_la.nii",
#     "ct_3": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_3_la.nii",
#     "ct_4": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_4_la.nii",
#     "ct_5": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_5_la.nii",
#     "ct_6": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_6_la.nii",
#     "ct_7": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_7_la.nii",
#     "ct_8": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_8_la.nii",
#     "ct_9": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_9_la.nii",
#     "ct_1": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_10_la.nii"
# }
#
# hparams = {
#     "batch_size": 1,
#     "learning_rate": 0.001,
#     "epochs": 1
# }
#
# # load DRR filenames
# path_to_files = "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/"
#
# all_files = []
#
# for f in listdir(path_to_files):  # list all files in directory
#     if isfile(join(path_to_files, f)):
#         all_files.append(f)
#
# all_files = sorted(all_files)
#
# # create train and val dictionaries
# all_data = {
#     "path_files": path_to_files,
#     "input": all_files,
#     "target": cts
# }
#
# # create datasets
# all_data_set = HeartData(all_data)
#
# train_dataloader = torch.utils.data.DataLoader(all_data_set, batch_size=hparams["batch_size"], shuffle=True,
#                                                num_workers=12)
#
# full = torch.zeros(1)
# for img, _ in train_dataloader:
#     img = torch.reshape(img, (-1,))
#     full = torch.cat((full, img), 0)
#
# mean = torch.mean(full)
# std = torch.std(full)

# mean:  tensor(0.1278)
# std:  tensor(0.1814)

















# all_file = open('/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/all_files.txt', 'r')
# lines_all = all_file.readlines()
#
# names = []
# for line in lines_all:
#     names.append(line.strip())
#
# for i in range(len(names)):
#     # open and preprocess the input images
#     image = Image.open(names[i])
#     image = np.array(image)
#     image = image.squeeze()
#     image = image[0:480, 0:480]
#     image = resize(image, output_shape=(64, 64), preserve_range=True)
#     image = image.astype(np.int16)
#     im_save = Image.fromarray(image)
#     im_save = im_save.convert('L')
#     im_save.save(names[i])
#
#     if i % 10 == 0:
#         print(i)


# open and preprocess the target volume

# cts = ["/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_2_la.nii",
#        "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_3_la.nii",
#        "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_4_la.nii",
#        "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_5_la.nii",
#        "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_6_la.nii",
#        "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_7_la.nii",
#        "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_8_la.nii",
#        "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_9_la.nii",
#        "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_10_la.nii"]
#
# for i in range(9):
#     buffer = nib.load(cts[i])
#     target = np.array(buffer.dataobj)
#     target = target.astype('float32')
#     print("target1: ", target.shape)
#     # after loading there is a '1' dimension at the end, remove it
#     target = target.squeeze()
#     print("target2: ", target.shape)
#
#     # crop to a cube
#     target = reshape_nifti(target)
#     print("target3: ", target.shape)
#
#     target = nib.Nifti1Image(target, affine=np.eye(4))
#     nib.save(target, cts[i])
