"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb


class UVnet(pl.LightningModule):

    def __init__(self, hparams=None):
        super().__init__()

        self.hparams = hparams

        self.accuracy = pl.metrics.Accuracy()
        self.save_hyperparameters()

        features = 64
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.depth_up = self.depth_up_sample()

        self.encoder1 = self.vnet_block(1, features)
        self.encoder2 = self.vnet_block(features, features*2)
        self.encoder3 = self.vnet_block(features*2, features*4)
        self.encoder4 = self.vnet_block(features*4, features*8)

        self.bottleneck = self.vnet_block(features*8, features*16)

        self.decoder4 = self.vnet_block(features*8 * 2, features*8)
        self.decoder3 = self.vnet_block(features*4 * 2, features*4)
        self.decoder2 = self.vnet_block(features*2 * 2, features*2)
        self.decoder1 = self.vnet_block(features * 2, features)

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(features*16, features*8, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.one_conv = nn.Conv3d(features, 1, kernel_size=1, stride=1)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        # preprocess
        x = x.float()  # otherwise there is a cda error because x is in double intiially
        x = x.unsqueeze(2)  # add 0 as first dim

        # up-sample to 3d
        x_3d = self.depth_up(x)

        # encode
        encode1 = self.encoder1(x_3d)
        encode1_pool = self.pool(encode1)

        encode2 = self.encoder2(encode1_pool)
        encode2_pool = self.pool(encode2)

        encode3 = self.encoder3(encode2_pool)
        encode3_pool = self.pool(encode3)

        encode4 = self.encoder4(encode3_pool)
        encode4_pool = self.pool(encode4)

        # bottleneck
        bottle = self.bottleneck(encode4_pool)

        # decode
        decode1 = self.upsample1(bottle)
        decode1 = torch.cat((decode1, encode4), dim=1)
        decode1 = self.decoder4(decode1)

        decode2 = self.upsample2(decode1)
        decode2 = torch.cat((decode2, encode3), dim=1)
        decode2 = self.decoder3(decode2)

        decode3 = self.upsample3(decode2)
        decode3 = torch.cat((decode3, encode2), dim=1)
        decode3 = self.decoder2(decode3)

        decode4 = self.upsample4(decode3)
        decode4 = torch.cat((decode4, encode1), dim=1)
        decode4 = self.decoder1(decode4)

        output = self.one_conv(decode4)

        output = self.sig(output)  # sigmoid so that the output is between 0 & 1

        return output

    @staticmethod
    def vnet_block(in_channels, features):
        return nn.Sequential(
                        nn.Conv3d(in_channels=in_channels, out_channels=features, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(num_features=features),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(in_channels=features, out_channels=features, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(num_features=features),
                        nn.ReLU(inplace=True))

    @staticmethod
    def depth_up_sample():
        return nn.Sequential(
                        nn.ConvTranspose3d(1, 1, kernel_size=(2, 1, 1), stride=[2, 1, 1]),
                        nn.ReLU(),
                        nn.ConvTranspose3d(1, 1, kernel_size=(4, 1, 1), stride=[4, 1, 1]),
                        nn.ReLU(),
                        nn.ConvTranspose3d(1, 1, kernel_size=(4, 1, 1), stride=[4, 1, 1]),
                        nn.ReLU(),
                        nn.ConvTranspose3d(1, 1, kernel_size=(2, 1, 1), stride=[2, 1, 1]),
                        nn.ReLU())

    def configure_optimizers(self):

        if self.hparams["optimiser"] == "Adam":
            return torch.optim.Adam(self.parameters(), self.hparams["learning_rate"])
        elif self.hparams["optimiser"] == "SGD":
            return torch.optim.SGD(self.parameters(), self.hparams["learning_rate"], momentum=0.9)

    def training_step(self, batch, batch_idx):
        # extract input and output from batch
        (inputs, targets) = batch

        # forward pass
        prediction = self.forward(inputs)

        # choose loss
        if self.hparams["loss"] == "DICE":

            # initialise loss function
            loss_func = DiceLoss()

            # calculate loss
            loss = loss_func(prediction, targets)

        elif self.hparams["loss"] == "BCE":
            # weighted pixel-wise BCE
            # calculate weights
            im = targets.type(torch.int)  # convert to int
            unique, counts = torch.unique(im, return_counts=True)  # get counts of how many 0s and 1s there are
            w0 = counts[0] / (counts[0] + counts[1])  # weight for background is no. of 1s divide by the no. of all

            # initialise loss function
            loss_func = nn.BCELoss(weight=w0)

            # calculate loss
            loss = loss_func(prediction, targets.unsqueeze(1))

        # log training loss
        wandb.log({'train_loss': loss, 'epoch': self.current_epoch})

        return loss

    def training_epoch_end(self, _) -> None:
        # log values
        wandb.log({"epochs": self.hparams["epochs"],
                   "batch_size": self.hparams["batch_size"],
                   "learning_rate": self.hparams["learning_rate"]})

    def validation_step(self, batch, batch_idx):
        # extract input and output from batch
        (inputs, targets) = batch

        # forward pass
        with torch.no_grad():  # don't compute the gradients, don't optimise
            prediction = self.forward(inputs)

            # choose loss
            if self.hparams["loss"] == "DICE":

                # initialise loss function
                loss_func = DiceLoss()

                # calculate loss
                loss = loss_func(prediction, targets)

            elif self.hparams["loss"] == "BCE":
                # weighted pixel-wise BCE
                # calculate weights
                im = targets.type(torch.int)  # convert to int
                unique, counts = torch.unique(im, return_counts=True)  # get counts of how many 0s and 1s there are
                w0 = counts[0] / (counts[0] + counts[1])  # weight for background is no. of 1s divide by the no. of all

                # initialise loss function
                loss_func = nn.BCELoss(weight=w0)

                # calculate loss
                loss = loss_func(prediction, targets.unsqueeze(1))

        # log training loss
        wandb.log({'val_loss': loss, 'epoch': self.current_epoch})
        self.log('val_loss', loss)

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class DiceLoss(nn.Module):

    def __init__(self, ):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, prediction, ground_truth):

        i_flat = prediction.view(-1)
        t_flat = ground_truth.view(-1)

        intersection = (i_flat * t_flat).sum()

        # t_flat doesn't have to be squared because it consists of 1s and 0s
        return 1 - ((2. * intersection + self.smooth) / (torch.sum(torch.square(i_flat) + t_flat) + self.smooth))