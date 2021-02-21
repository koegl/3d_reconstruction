import numpy as np
import os
import torch
import torchvision.models as models
from torchvision import transforms
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from random import randrange
from os import listdir
from os.path import isfile, join

from dataset import HeartData
from unet import UVnet
import argparse


def main(params):

    # set params
    batch_size = params.batch_size
    learning_rate = params.learning_rate
    epochs = params.epochs
    gpu_id = params.gpu_id
    seed = params.seed
    if params.loss == 0:
        loss = "DICE"
    elif params.loss == 1:
        loss = "BCE"
    if params.optimiser == 0:
        optimiser = "Adam"
    elif params.optimiser == 1:
        optimiser = "SGD"

    # determinism
    # np.random.seed(seed)  # numpy
    # torch.random.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    ################################################################################################################
    # set up logging ###############################################################################################
    ################################################################################################################
    name_base = "full_dataset"
    name_lr = "_lr" + str(learning_rate)
    name_bs = "_bs" + str(batch_size)
    name_e = "_e" + str(epochs)
    rand_id = "_" + str(randrange(1111111111, 9999999999))
    name = name_base + name_lr + name_bs + name_e + rand_id
    wandb_logger = WandbLogger(name=name, project="sweep")

    ################################################################################################################
    # define hyper-parameters ######################################################################################
    ################################################################################################################

    hparams = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "training": ["ct_2", "ct_3", "ct_4", "ct_5", "ct_9"],  # 1 means 10
        "validation": ["ct_1", "ct_8"],
        "seed": seed,
        "loss": loss,
        "optimiser": optimiser,
        "normalisation": "no",
        "random": "yes"
    }

    ################################################################################################################
    # set up datasets and loading ##################################################################################
    ################################################################################################################
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

    # load DRR filenames
    path_to_files = "/home/fryderyk/Desktop/Datasets/STACOM_SLAWT/STACOM_3d_64/"

    train_files = []
    val_files = []

    for f in listdir(path_to_files):  # list all files in directory
        if isfile(join(path_to_files, f)):
            value = [True for key in hparams["training"] if key in f]  # check if a training ct is in the filename
            if value:  # if value is not empty
                train_files.append(f)

            value = [True for key in hparams["validation"] if key in f]  # check if a validation ct is in the filename
            if value:
                val_files.append(f)

    train_files = sorted(train_files)
    val_files = sorted(val_files)

    # create train and val dictionaries
    train_data = {
        "path_files": path_to_files,
        "input": train_files,
        "target": cts
    }
    val_data = {
        "path_files": path_to_files,
        "input": val_files,
        "target": cts
    }

    # create datasets
    train_data_set = HeartData(train_data)
    val_data_set = HeartData(val_data)

    train_dataloader = torch.utils.data.DataLoader(train_data_set, batch_size=hparams["batch_size"], shuffle=True,
                                                   num_workers=12)
    val_dataloader = torch.utils.data.DataLoader(val_data_set, batch_size=hparams["batch_size"], shuffle=False,
                                                 num_workers=12)

    ################################################################################################################
    # set up model####################################################################################################
    ################################################################################################################

    model = UVnet(hparams=hparams)

    ################################################################################################################
    # train the model ##############################################################################################
    ################################################################################################################
    # early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',  # what is monitored
        min_delta=0.0,  # an absolut change smaller than that is not an improvement
        patience=4,  # number of validation epochs with no improvement
        mode='min'  # when quantity stops decreasing
    )

    # set up trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=hparams["epochs"],
        callbacks=[early_stop_callback],
        gpus=[gpu_id] if torch.cuda.is_available() else None  # write -1 to use all GPUs
    )

    # train
    trainer.fit(model, train_dataloader, val_dataloader)

    ################################################################################################################
    # save model ###################################################################################################
    ################################################################################################################

    save_path = "/home/fryderyk/Desktop/repository/models/" + name + ".pt"
    model = model.cpu()

    # save entire model
    # torch.save(model, save_path)
    # save model weights for inference
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int, default=10,
                        help="Batch size (int). Default: 10")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.0001,
                        help="Learning rate (float). Default: 0.0001")
    parser.add_argument("-e", "--epochs", type=int, default=15,
                        help="Amount of epochs (int)")
    parser.add_argument("-g", "--gpu_id", type=int, default=0, choices=[0, 1],
                        help="GPU ID (int), either 0 or 1. Default: 0")
    parser.add_argument("-s", "--seed", type=int, default=1,
                        help="Setting the random seed. Default: 1")
    parser.add_argument("-ls", "--loss", type=int, default=0, choices=[0, 1],
                        help="Choose loss function. 0:DICE, 1:WeightedPixBCE. Default: 0")
    parser.add_argument("-op", "--optimiser", type=int, default=0, choices=[0, 1],
                        help="Choose optimiser. 0:Adam, 1:SGD. Default: 0")

    args = parser.parse_args()

    main(args)
