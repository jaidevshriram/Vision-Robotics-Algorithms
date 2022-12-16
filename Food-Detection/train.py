# import wandb
# wandb.init(project='cv-5', entity='jaidev')

import argparse

parser = argparse.ArgumentParser(description="Train a Food detector!")
parser.add_argument('--img_dir', default="data/train_images",
                    help='Directory to images')
parser.add_argument('--batch_size', default=200, help='Batch Size')
parser.add_argument('--name', default='default', help='Name of model')
parser.add_argument('--num_workers', default=4, help='num_workers')

import os
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch.utils.data import DataLoader

from dataset import FoodDataset
from model import FoodClassifier

from tqdm.contrib import tenumerate

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

seed_everything(148)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    args = parser.parse_args()

    transform = T.Compose([
    #            T.RandomHorizontalFlip(),
    #            T.ColorJitter(0.1, 0.1, 0.1, 0.1),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    train_dataset = FoodDataset(image_dir=args.img_dir, train_csv="train.csv", transforms=transform)
    val_dataset = FoodDataset(image_dir=args.img_dir, train_csv="val.csv", transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    model = FoodClassifier().to(device)
    wandb_logger = WandbLogger(name=args.name, log_model=True)
    wandb_logger.watch(model, log='all')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        filename='food-{epoch:02d}-{val_acc:.2f}',
        save_top_k=3,
        mode='max',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(gpus=1, logger=wandb_logger, log_every_n_steps=1,
            callbacks=[checkpoint_callback, lr_monitor], check_val_every_n_epoch=3, max_epochs=30)
    trainer.fit(model, train_loader, val_loader)
