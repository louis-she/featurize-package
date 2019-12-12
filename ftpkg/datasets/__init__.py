from featurize_jupyterlab.core import Dataset, Option, Task, BasicModule, DataflowModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import StratifiedKFold
from zipfile import ZipFile
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def rle2mask(label, shape):
    label = label.split(" ")
    positions = map(int, label[0::2])
    length = map(int, label[1::2])
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for pos, le in zip(positions, length):
        mask[pos:(pos + le)] = 1
    #masks[:, :, idx] = mask.reshape(shape[0], shape[1], order='F')
    return mask

def Kfold(df,n_splits=5):
    labels = []
    print('Spliting...')
    num_classes = len(df.columns) - 1
    for i,j in df.iterrows():
        tmp = []
        for k in range(num_classes):
            if j[df.columns[k]] is not np.nan:
                tmp.append(2**(k))

        labels.append(int(np.sum(tmp)))

    df['tmp'] = labels

    Spliter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=99)
    X, Y = df[df.columns[0]], df['tmp']

    Spliter.get_n_splits(X, Y)

    train_folds, val_folds = [], []
    for train_index, val_index in Spliter.split(X, Y):
        train_df = df.iloc[train_index].reset_index().drop(['index', 'labels'],axis=1)
        val_df = df.iloc[val_index].reset_index().drop(['index', 'labels'],axis=1)
        train_folds.append(train_df)
        val_folds.append(val_df)

    return train_folds, val_folds

class FeaturizeDataset(TorchDataset):
    
    def __init__(self, annotation, data_folder, transforms):
        self.df = annotation
        self.root = data_folder
        self.transforms = transforms
        self.fnames = self.df.columns[0]
        self.classes = self.df.columns[1:len(self.df.columns)+1]
        self.num_classes = len(self.classes)
        self.count = 0
        self.miss_count = 0

    def __len__(self):
        return len(self.df)
    
    def __count__(self):
        return self.count


class ClassificationDataset(FeaturizeDataset):

    def make_label(self, df_row):
        
        labels = df_row[1:self.num_classes + 1]
        classification_labels = []
        for idx, label in enumerate(labels.values):
            classification_labels.append(label)
            results = np.array(classification_labels)
        
        return results
    
    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        image_id = self.df.iloc[idx][self.fnames]
        image_path = os.path.join(self.root, image_id)
        try:
            img = cv2.imread(image_path)
        except:
            self.miss_count += 1
            return
        label = self.make_label(df_row)
        augmented = self.transforms(image=img)
        img = augmented['image']
        self.count += 1
        
        return img, label, image_id


class SegmentationDataset(FeaturizeDataset):
    
    def __init__(self, annotation, data_folder, transforms):
        self.df = annotation
        self.root = data_folder
        self.transforms = transforms
        self.fnames = self.df.columns[0]
        self.classes = self.df.columns[1:len(self.df.columns)+1]
        self.num_classes = len(self.classes)
        self.count = 0

    def make_mask(self, df_row, shape):
        labels = df_row[1:self.num_classes + 1]
        masks = np.zeros(shape, dtype=np.float32)
        for idx, mask in enumerate(labels.values):
            if mask is not np.nan:
                mask_array = rle2mask(mask, shape)
                masks[:, :, idx] = mask_array.reshape(shape[0], shape[1], order='F')
        results = masks
        return results
    
    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        image_id = self.df.iloc[idx][self.fnames]
        image_path = os.path.join(self.root, image_id)
        try:
            img = cv2.imread(image_path)
        except:
            self.miss_count += 1
            return
        mask = self.make_mask(df_row, img.shape)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        mask = mask[0].permute(2, 0, 1)
        self.count += 1
        
        return img, mask, image_id

    def __len__(self):
        return len(self.df)
    
    def __count__(self):
        return self.count


class TrainDataset(Dataset):
    """This is a segmentation datasettrain_test_split preparing data from annotations and data directory
    """
    #fold = Option(help='Absolute fold path to the dataset', required=True, default="~/.minetorch_dataset/torchvision_mnist")  
    annotations = Option(type='uploader', help='You may upload a csv file with columns=["image_names", "class_1_labels", "class_2_labels", ..., "class_n_labels"]')
    upload = Option(help='Upload your trainning images', type='uploader', required=True)
    batch_size = Option(name='Batch Size', type='number')
    split_ratio = Option(name='Split Ratio', type='number', default=0.2, help='Split your datasets into trainset and valset')
    #k_fold = Option(name='K folds', type='number', default=1, help='Number of folds to split from original datasets', required=False)
    dataset = BasicModule(name='Dataset', component_types=['Dataset'])
    train_augmentations = DataflowModule(name='Train Augmentations', component_types=['Dataflow'], multiple=True, required=False)
    val_augmentations = DataflowModule(name='Validation Augmentations', component_types=['Dataflow'], multiple=True, required=False)


    def __call__(self):
        #assert isinstance(k_fold, int) and kfold > 0, 'K fold must be an interger'
        #assert isinstance(batch_size, int), 'Batch Size must be an interger'
        #assert 0 <= split_ratio <= 1, 'Split Ratio must be between 0 to 1'
        print(self.upload)
        print(self.annotations)
        with ZipFile(self.upload[0], 'r') as zip_object:
            zip_object.extractall()
        fold = self.upload[0].split('.zip')[0]
        df = pd.read_csv('./inputs/annotations/satellite_sample.csv')
        #train_dfs, val_dfs = Kfold(df, n_splits=5)  # TO DO: kfold for datasets
        train_df, val_df = train_test_split(df, test_size=0.1)

        if self.__task__.task_type == 'classification':
                dataloader_train = (
                    DataLoader(dataset=ClassificationDataset(annotation=train_df, data_folder=fold, transforms=self.train_augmentations), batch_size=self.batch_size),
                    DataLoader(dataset=ClassificationDataset(annotation=val_df, data_folder=fold, transforms=self.val_augmentations), batch_size=self.batch_size)
                    )
                    
        elif self.__task__.task_type == 'segmentation':
                dataloader_train = (
                    DataLoader(dataset=SegmentationDataset(annotation=train_df, data_folder=fold, transforms=self.train_augmentations), batch_size=self.batch_size),
                    DataLoader(dataset=SegmentationDataset(annotation=val_df, data_folder=fold, transforms=self.val_augmentations), batch_size=self.batch_size)
                    )
        return dataloader_train
