import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
import numpy


def dice_score(prediction, ground_truth):
    i_flat = prediction.flatten()
    t_flat = ground_truth.flatten()

    i_flat = i_flat.astype(np.bool)
    t_flat = t_flat.astype(np.bool)

    if i_flat.shape != t_flat.shape:
        raise ValueError("Shape mismatch: prediction and ground truth must have same shape.")

    intersection = np.logical_and(i_flat, t_flat)

    return 2. * intersection.sum() / (i_flat.sum() + t_flat.sum())


def reshape_nifti(nifti):
    shape = nifti.shape
    if shape[2] > 512:
        nifti = nifti[:, :, 0:512]
    elif shape[2] < 512:
        nifti = np.concatenate((nifti, np.zeros((shape[0], shape[1], 512 - shape[2]))), axis=2)

    nifti = resize(nifti, output_shape=(64, 64, 64), preserve_range=True)

    return nifti


def volumetric_plot(image_pred, image_true, threshold=0.05):
    # make it positive
    image_pred = np.abs(np.asarray(image_pred).squeeze())
    image_true = np.abs(np.asarray(image_true).squeeze())

    # make binary
    image_pred[image_pred < threshold] = 0
    image_pred[image_pred >= threshold] = 1
    image_true[image_true < threshold] = 0
    image_true[image_true >= threshold] = 1

    # calculate the dice score to display on the axes
    dice = dice_score(image_true, image_pred)

    # volumetric visualisation
    prediction = image_pred.astype(bool)
    truth = image_true.astype(bool)
    voxels = prediction | truth
    colors = np.empty(voxels.shape, dtype=object)
    colors[prediction] = 'red'
    colors[truth] = 'cornflowerblue'

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='none')
    ax.set_xlabel("Dice score: %.2f" % dice)
    ax.set_ylabel("Dice score: %.2f" % dice)
    ax.set_zlabel("Dice score: %.2f" % dice)

    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)

    return fig


def volumetric_plot_torch(image_pred, image_true, dice, threshold=0.05):
    # make it positive
    image_pred = torch.abs(image_pred.squeeze())
    image_true = torch.abs(image_true.squeeze())

    # if prediction
    threshold = threshold
    image_pred[image_pred < threshold] = 0
    image_pred[image_pred >= threshold] = 1
    image_true[image_true < threshold] = 0
    image_true[image_true >= threshold] = 1

    # volumetric visualisation
    colors = torch.zeros(image_pred.shape + (3,))
    colors[..., 0][image_true == 1] = 37/255
    colors[..., 1][image_true == 1] = 150/255
    colors[..., 2][image_true == 1] = 190/255
    colors[..., 0][image_pred == 1] = 212/255
    colors[..., 1][image_pred == 1] = 36/255
    colors[..., 2][image_pred == 1] = 44/255

    # fuse the two arrays
    voxels = torch.zeros(image_pred.shape)
    voxels = voxels.bool()
    voxels[image_true == 1] = True
    voxels[image_pred == 1] = True
    voxels[image_true != 1] = False
    voxels[image_pred != 1] = False

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    voxels = voxels.numpy()
    colors = colors.numpy()
    ax.voxels(voxels, facecolors=colors, edgecolor='none')
    ax.set_xlabel("Dice score: %.2f" % dice)
    ax.set_ylabel("Dice score: %.2f" % dice)
    ax.set_zlabel("Dice score: %.2f" % dice)

    ax.set_xlim(10, 54)
    ax.set_ylim(10, 54)
    ax.set_zlim(10, 54)

    return fig
