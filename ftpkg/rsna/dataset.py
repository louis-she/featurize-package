import os
import zipfile
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import (Compose, ElasticTransform, Flip, GaussNoise,
                            HorizontalFlip, IAAPiecewiseAffine, IAASharpen,
                            Normalize, OneOf, RandomBrightness, Resize,
                            ShiftScaleRotate, VerticalFlip)
from albumentations.imgaug.transforms import IAASharpen
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from featurize_jupyterlab.core import Dataset, Option


class RSNADataset(Dataset):
    
    def __init__(self, df, data_folder, transforms):
        self.df = df
        self.root = data_folder
        self.transforms = transforms
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        image_id = df_row['name']+'.png'
        image_path = os.path.join(self.root, image_id)
        img = cv2.imread(image_path)
        augmented = self.transforms(image=img)
        img = augmented['image']
        label = np.array(df_row[df_row.keys()[1:7]].tolist(),dtype=float)
        return img, torch.from_numpy(label).float()

    def __len__(self):
        return len(self.fnames)

#
def prepare_datasets(folder):
    if os.path.isfile(folder / 'train_processed.csv'):
        return

    os.environ['KAGGLE_USERNAME'] = 'pengbo0054'
    os.environ['KAGGLE_KEY'] = 'fcabd994e7e108759771ac7f3ad3aae1'
    import kaggle
    kaggle.api.competition_download_files('rsna-intracranial-hemorrhage-detection', folder, False, False)
    with zipfile.ZipFile(folder / 'rsna-intracranial-hemorrhage-detection.zip', 'r') as zip_ref:
        zip_ref.extractall(folder)
    for category in ('train', 'test'):
        images_dir = folder / category
        try:
            os.mkdir(images_dir)
        except:
            pass
        with zipfile.ZipFile(folder / f'stage_2_{category}_images.zip') as f:
            f.extractall(images_dir)

    gen_processed_csv(folder)


def gen_processed_csv(folder):
    a = pd.read_csv(folder / 'stage_2_train.csv')
    processed_csv_file = folder / 'train_processed.csv'
    if os.path.isfile(processed_csv_file):
        return
    x = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']
    tmp = []

    for i in tqdm(range(int(len(a)/6))):
        fn = a.loc[i*6]['ID'].split('_')[0]+'_'+a.loc[i*6]['ID'].split('_')[1]
        labels = list(a.loc[i*6:(i+1)*6-1]['Label'])
        labels.append(fn)
        labels = labels[-1:]+labels[0:len(labels)-1]
        tmp.append(labels)

    df = pd.DataFrame(tmp, columns = ['name', 'epidural','intraparenchymal','intraventricular','subarachnoid','subdural','any']) 
    df.to_csv(processed_csv_file, index=False)


class FeaturizeRSNADataset(Dataset):
    """Kaggle Severstal Steel Defect Detection Dataset
    """
    name = 'RSNA Dataset'

    folder = Option(default='/datasets')
    validation_percentage = Option(type='number')
    random_split_seed = Option(type='number')
    batch_size = Option(type='number', default=16)
    force_download = Option(type='boolean')

    train_dataloader = Option(type='hardcode', help='The `Train Dataloader` which should be already configured', required=False)
    val_dataloader = Option(type='hardcode', help='The `Val Dataloader` which should be already configured', required=False)

    def __call__(self):
        folder = Path(self.folder)
        prepare_datasets(folder)
        df = pd.read_csv(folder / 'train_processed.csv', index_col='ImageId')
        train_df, val_df = train_test_split(
            df,
            test_size=self.validation_percentage,
            random_state=self.random_split_seed
        )
        return (
            torch.utils.data.DataLoader(RSNADataset(train_df, folder, self.train_dataloader), self.batch_size),
            torch.utils.data.DataLoader(RSNADataset(val_df, folder, self.val_dataloader), self.batch_size)
        )
