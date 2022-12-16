import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import sampler
import torchvision.utils as vutils
import pytorch_lightning as pl

from efficientnet_pytorch import EfficientNet

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class FoodClassifierPreTrained(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b1')
#        self.model.train()
#        self.model.eval()

#        for param in self.model.parameters():
#            param.requires_grad = False

#        print(self.model)
        infeatures = self.model._fc.in_features
        self.clf = nn.Linear(in_features=1000, out_features=61, bias=True)

        self.model._fc = nn.Linear(in_features=infeatures, out_features=61, bias=True)

        self.accuracy = pl.metrics.Accuracy()
        self.f1 = pl.metrics.F1(num_classes=61)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
#        feature = self.model(x)
#        feature = nn.ReLU()(feature)
        return self.model(x)

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def training_step(self, batch, batch_idx):

        x, y = batch

        inputs, targets_a, targets_b, lam = self.mixup_data(x, y, 1, True)

        y = torch.tensor(y, dtype=torch.long).to(self.device)

        out = self(inputs)
        loss = self.mixup_criterion(out, targets_a, targets_b, lam)

        _, ind = torch.max(out.data, 1)

        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(ind, y), prog_bar=True)
        self.log('train_f1', self.f1(ind, y)) 

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y = torch.tensor(y, dtype=torch.long).to(self.device)

        out = self(x)
        loss = nn.CrossEntropyLoss()(out, y)

        val, ind = torch.max(nn.Softmax(dim=1)(out), 1)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(ind, y), prog_bar=True)
        self.log('val_f1', self.f1(ind, y), prog_bar=True)

        return loss

    def configure_optimizers(self):
    #    lr = 0.01
    #    optimizer = optim.SGD(model.parameters(), lr=lr)
    #    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #    optimizer = optim.Adagrad(model.parameters(), lr=lr)
        optimizer = optim.Adam(self.parameters(), lr=0.0005, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
        }

    def on_epoch_start(self):
        print('\n')
