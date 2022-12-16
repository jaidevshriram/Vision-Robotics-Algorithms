import os
import numpy as np
import copy
import pandas as pd

from PIL import Image
from sklearn import preprocessing

import torch
import torch.utils.data as data
from torchvision import transforms

class FoodDataset(data.Dataset):
    
    def __init__(self, image_dir, train_csv, transforms=None):
        
        self.image_dir = image_dir
        self.files = []
        
        dataset_csv = pd.read_csv(train_csv)

        self.le = preprocessing.LabelEncoder()

        try:
            self.le.fit(dataset_csv['ClassName'])
        except:
            print("No classname")
                
        for i, file in dataset_csv.iterrows():
            self.files.append({
                'img': file['ImageId'],
                'class': self.le.transform([file['ClassName']])[0] if 'ClassName' in dataset_csv.columns else -1
            })

        self.transforms = transforms
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        
        out = copy.deepcopy(self.files[index])
        out['img'] = Image.open(os.path.join(self.image_dir, out['img'])).resize((240, 240))

        if self.transforms:
            out['img'] = self.transforms(out['img'])
            # out['class'] = transforms.ToTensor()(out['class']
        
        return out['img'], out['class']
            
