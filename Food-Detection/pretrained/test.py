# import wandb
# wandb.init(project='cv-5', entity='jaidev')

import argparse

parser = argparse.ArgumentParser(description="Train a Food detector!")
parser.add_argument('--img_dir', default="../data/test_images",
                    help='Directory to images')
parser.add_argument('--num_workers', default=10, help='num_workers')
parser.add_argument('--batch_size', default=32, help='Batch Size')
parser.add_argument('--name', default='default', help='Name of model')

import os
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from dataset import FoodDataset
from model import FoodClassifierPreTrained

from tqdm.contrib import tenumerate

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything, Trainer

seed_everything(148)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    args = parser.parse_args()

    train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2, 0.2),
                T.RandomPerspective(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                T.RandomErasing(),
            ])

    val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    train_dataset = FoodDataset(image_dir="../data/train_images", train_csv="../train.csv", transforms=test_transform)
    val_dataset = FoodDataset(image_dir="../data/train_images", train_csv="../val.csv", transforms=val_transform)
    test_dataset = FoodDataset(image_dir=args.img_dir, train_csv="../test.csv", transforms=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

#    model = FoodClassifierPreTrained.load_from_checkpoint("wandb/run-20210416_160509-245c4s5i/files/assignment-5-efficientnet/245c4s5i/checkpoints/food-epoch=24-val_acc=0.50.ckpt")

    model = FoodClassifierPreTrained.load_from_checkpoint("wandb/run-20210418_021313-3qdcvz4j/files/assignment-5-efficientnet/3qdcvz4j/checkpoints/food-epoch=04-val_acc=0.59.ckpt").to(device)

    trainer = Trainer(gpus=1, log_every_n_steps=1, check_val_every_n_epoch=5, max_epochs=1)
#    trainer.validate(val_loader)
#    trainer.fit(model, train_loader, train_loader)

    model.eval()
    model = model.to(device)

    pred = []
    with torch.no_grad():
        count = 0
        for img, _ in test_loader:
            img = img.to(device)
            out = model(img)
            _, pred_class = torch.max(nn.Softmax(dim=1)(out), 1)
#            print(pred_class)
            for i in pred_class.detach().cpu().numpy():
                pred.append(i)
            count += 1

#    print(train_dataset.le.inverse_transform(pred))
    df = pd.DataFrame(train_dataset.le.inverse_transform(pred), columns=['ClassName'])

    for i, row in df.iterrows():
        row['ClassName'] = row['ClassName'].strip()
    print(df)

    df.to_csv('submission.csv', index=False)


