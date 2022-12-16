import numpy as np
import pandas as pd

from dataset import FoodDataset

if __name__ == '__main__':

    base = pd.read_csv("base.csv")
    dataset = FoodDataset(image_dir="", train_csv="base.csv")

    train_csv = open("train.csv", "w")
    val_csv = open("val.csv", "w")

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    split = int(np.floor(0.2 * len(dataset)))
    train_idx, val_idx = indices[split:], indices[:split]

    for i, row in base.iterrows():
        if i in train_idx:
            train_csv.write(f"{row['ImageId']}, {row['ClassName']} \n")
        else:
            val_csv.write(f"{row['ImageId']}, {row['ClassName']} \n")