import torch
import numpy as np
import nibabel as nib
from dataset import HeartData
from os import listdir
from os.path import isfile, join
from utils import dice_score
import pickle
from unet import UVnet
import torch.nn as nn


hparams = {
    "batch_size": 1,
    "learning_rate": 0.001,
    "epochs": 1
}

cts = {
    "ct_2": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_2_la.nii",
    "ct_3": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_3_la.nii",
    "ct_4": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_4_la.nii",
    "ct_5": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_5_la.nii",
    "ct_6": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_6_la.nii",
    "ct_7": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_7_la.nii",
    "ct_8": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_8_la.nii",
    "ct_9": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_9_la.nii",
    "ct_1": "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/cts/ct_10_la.nii"
}

# load filenames
# define here or load from file
path_to_files = "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64_zeros/"
files = ["ct_2_x20y0z0.jpg",
             "ct_4_x20y0z0.jpg",
             "ct_6_x20y0z0.jpg",
             "ct_8_x20y0z0.jpg",
             "ct_9_x20y0z0.jpg",
             "ct_10_x20y0z0.jpg"]

# load all ct_6 DRRs
ct_id = "ct_7"
files = [f for f in listdir(path_to_files) if isfile(join(path_to_files, f)) and ct_id in f]

# load all cts
files = [f for f in listdir(path_to_files) if isfile(join(path_to_files, f)) and "ct_6" not in f]
files = sorted(files)


data = {
    "path_files": path_to_files,
    "input": files,
    "target": cts
}

data_set = HeartData(data)

model_name = "full_dataset_lr0.00012_bs8_e12_6764128967"  # without .pt

# load full model
# model = torch.load("/home/fryderyk/Desktop/repository/models/" + model_name + ".pt")

# load only weights
model = UVnet(hparams=hparams)
model.load_state_dict(torch.load("/home/fryderyk/Desktop/repository/models/" + model_name + ".pt"))

model.eval()

dice_all = {}
bce_all = {}
val_dataloader = torch.utils.data.DataLoader(data_set, batch_size=hparams["batch_size"], num_workers=12)

for i, data in enumerate(val_dataloader):

    if i > -1:
        # inference
        print(files[i])
        x, y_true = data

        y_pred = model(x)
        y_pred_cpu = y_pred.cpu()
        y_true_cpu = y_true.cpu()

        y_pred_np = y_pred_cpu.detach().numpy()
        y_true_np = y_true_cpu.detach().numpy()
        y_pred_np = np.abs(y_pred_np)

        # for calculating dice per view
        threshold = 0.05
        y_pred_np[y_pred_np < threshold] = 0
        y_pred_np[y_pred_np >= threshold] = 1
        y_true_np[y_true_np < threshold] = 0
        y_true_np[y_true_np >= threshold] = 1
        dice_all[files[i]] = dice_score(y_true_np, y_pred_np)

        # for calculating BCE (before threshold, we need probabilities)
        im = y_true.type(torch.int)  # convert to int
        unique, counts = torch.unique(im, return_counts=True)  # get counts of how many 0s and 1s there are
        w0 = counts[0] / (counts[0] + counts[1])  # weight for background is no. of 1s divide by the no. of all

        # initialise loss function
        loss_func = nn.BCELoss(weight=w0)

        # calculate loss
        loss = loss_func(y_pred, y_true.unsqueeze(1))
        bce_all[files[i]] = np.asscalar(loss.cpu().detach().numpy())

        if i < -50:
        # for saving reconstructions
            pred_save = nib.Nifti1Image(y_pred_np, affine=np.eye(4))
            true_save = nib.Nifti1Image(y_true_np, affine=np.eye(4))

            nib.save(pred_save, "/home/fryderyk/Desktop/repository/compare/" + model_name + "/pred_" + files[i][:-4] + ".nii")
            nib.save(true_save, "/home/fryderyk/Desktop/repository/compare/" + model_name + "/true_" + files[i][:-4] + ".nii")


with open("/home/fryderyk/Desktop/repository/compare/" + model_name + "/dice_all.p", 'wb') as fp:
    pickle.dump(dice_all, fp, protocol=pickle.HIGHEST_PROTOCOL)
with open("/home/fryderyk/Desktop/repository/compare/" + model_name + "/bce_all.p", 'wb') as fp:
    pickle.dump(bce_all, fp, protocol=pickle.HIGHEST_PROTOCOL)


# with open("/home/fryderyk/Desktop/repository/compare/" + model_name + "/dice_" + ct_id + "_dict.p", 'wb') as fp:
#     pickle.dump(dice_all, fp, protocol=pickle.HIGHEST_PROTOCOL)
# with open("/home/fryderyk/Desktop/repository/compare/" + model_name + "/bce_" + ct_id + "_dict.p", 'wb') as fp:
#     pickle.dump(bce_all, fp, protocol=pickle.HIGHEST_PROTOCOL)

