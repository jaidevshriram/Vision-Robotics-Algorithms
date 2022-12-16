import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import sampler
import torchvision.utils as vutils
import pytorch_lightning as pl

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class FoodClassifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1),
#            nn.BatchNorm2d(128),
#            nn.Dropout(0.2),
#            nn.LeakyReLU(0.2),
#            nn.ReLU(),
            nn.Tanh(),
#            nn.AvgPool2d(4, 2, 1),
 #           nn.Sigmoid(),
            nn.Conv2d(128, 64, 4, 2, 1),
#           nn.Conv2d(3, 128, 4, 2, 1),
#            nn.BatchNorm2d(128),
#            nn.Dropout(0.2),
#            nn.LeakyReLU(0.2),
#            nn.ReLU(),
            nn.Tanh(),
#            nn.AvgPool2d(4, 2, 1),
 #           nn.Sigmoid(),
            nn.Conv2d(64, 32, 4, 2, 1),
#            nn.Dropout(0.2),
#            nn.BatchNorm2d(64),
#            nn.LeakyReLU(0.2),
#            nn.ReLU(),
            nn.Tanh(),
#            nn.AvgPool2d(4, 2, 1), 
#            nn.Sigmoid(),
            nn.Conv2d(32, 16, 4, 2, 1),
#            nn.ReLU(),
            nn.Tanh(),
#            nn.AvgPool2d(4, 2, 1), 
#            nn.Dropout(0.2),
#            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(64 * 4, 61),
#            nn.Dropout(0.2),
#            nn.LeakyReLU(0.2),
#            nn.ReLU(),
#            nn.Linear(64 * 2, 61),
        )

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x, y = batch

        y = torch.tensor(y, dtype=torch.long).to(self.device)

        out = self(x)
#        print(out.shape)
#        exit()
        loss = nn.CrossEntropyLoss()(out, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(nn.Softmax(dim=1)(out), y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y = torch.tensor(y, dtype=torch.long).to(self.device)

        out = self(x)
        #print(out.shape)
        #exit()
        loss = nn.CrossEntropyLoss()(out, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(nn.Softmax(dim=1)(out), y))

        return loss

    def configure_optimizers(self):
        lr = 0.01
        optimizer = optim.SGD(self.parameters(), lr=lr)
        #optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        #optimizer = optim.Adagrad(self.parameters(), lr=lr)
#        optimizer = optim.Adam(self.parameters(), lr=0.0005, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }

        return optimizer

    def on_epoch_start(self):
        print('\n')
