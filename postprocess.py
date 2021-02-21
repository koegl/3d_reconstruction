import nibabel as nib
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import dice_score, volumetric_plot


# TODO:
#  Find biggest connected component of thresholded prediction
model_name = "full_dataset_lr0.00012_bs8_e12_6764128967"  # without .pt
ct_id = "ct_7"

'''
# visualise volumetric
# load image
image_pred_loaded = nib.load("/home/fryderyk/Desktop/repository/compare/" + model_name +
                             "/pred_" + ct_id + "_x000y0z0.nii")
image_true_loaded = nib.load("/home/fryderyk/Desktop/repository/compare/" + model_name +
                             "/true_" + ct_id + "_x000y0z0.nii")
image_pred = np.abs(np.array(image_pred_loaded.dataobj).squeeze())
image_true = np.abs(np.array(image_true_loaded.dataobj).squeeze())

# threshold a bit
threshold = 0.05
image_pred[image_pred < threshold] = 0
image_pred[image_pred >= threshold] = 1
image_true[image_true < threshold] = 0
image_true[image_true >= threshold] = 1

fig = volumetric_plot(image_pred, image_true)

plt.show()




'''
# plot dice and bce vs angle
with open("/home/fryderyk/Desktop/repository/compare/" + model_name + "/dice_" + ct_id + "_dict.p", 'rb') as fx:
    dice = pickle.load(fx)
# with open("/home/fryderyk/Desktop/repository/compare/" + model_name + "/bce_" + ct_id + "_dict.p", 'rb') as fxb:
#     bce = pickle.load(fxb)

# process DICE to array
x_dice = np.empty((0,), int)
y_dice = np.zeros((0,), float)
for key, value in dice.items():
    if "ct_1" in key:
        x_dice = np.append(x_dice, int(key[7:10]))
        y_dice = np.append(y_dice, value)
    else:
        x_dice = np.append(x_dice, int(key[6:9]))
        y_dice = np.append(y_dice, value)

# process BCE to array
# x_bce = np.empty((0,), int)
# y_bce = np.zeros((0,), float)
# for key, value in bce.items():
#     x_bce = np.append(x_bce, int(key[6:9]))
#     y_bce = np.append(y_bce, value)

# print("max ", np.argmax(y_dice))
# print("min1 ", np.argmin(y_dice[:150]))
# print("min2 ", np.argmin(y_dice[151:350]) + 150)

# plot both
plt.plot(x_dice, y_dice, c='b', label='DICE score')
#plt.plot(x_bce, y_bce, c='r', label='BCE')
plt.xlabel("Projection angle [$^\circ$]")
plt.ylabel("DICE score")
plt.title("Reconstruction DICE Score vs DRR Projection Angle")
plt.legend()
plt.show()

