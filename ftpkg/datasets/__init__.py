from featurize_jupyterlab.core import Dataset, Option, Task, BasicModule, DataflowModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import StratifiedKFold
from zipfile import Zipfile

def rle2mask(label):
    label = label.split(" ")
    positions = map(int, label[0::2])
    length = map(int, label[1::2])
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for pos, le in zip(positions, length):
        mask[pos:(pos + le)] = 1
    masks[:, :, idx] = mask.reshape(shape[0], shape[1], order='F')
    return masks

def Kfold(df,n_splits=5):
    labels = []
    print('Spliting...')
    num_classes = len(df.columns) - 1
    for i,j in df.iterrows():
        tmp = []
        for k in range(num_classes):
            if j['%s'%(k)] is not np.nan:
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

class TorchDatasetBoth(TorchDataset):
    
    def __init__(self, annotation, data_folder, transforms, mode='classifications'):
        self.df = annotation
        self.root = data_folder
        self.transforms = transforms
        self.mode = mode
        self.fnames = self.df.columns[0]
        self.classes = self.df.columns[1:len(df.columns)+1]
        self.num_classes = len(self.classes)
        self.count = 0

    def make_mask(self, df_row, shape):
        
        labels = df_row[1:self.num_classes + 1]
        if self.mode == 'classifications':
            classification_labels = []
            for idx, label in enumerate(labels.values):
                classification_labels.append(label)
                results = np.array(classification_labels)

        elif self.mode == 'segmentations':
            masks = np.zeros(shape, dtype=np.float32)
            for idx, label in enumerate(labels.values):
                if label is not np.nan:
                    masks = rle2mask(label)
            results = masks
        else:
            assert mode in ['classifications', 'segmentations']
        
        return results
    
    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        image_id = self.df.iloc[idx][self.fnames]
        image_path = os.path.join(self.root, image_id)
        try:
            img = cv2.imread(image_path)
        except:
            return
        mask = self.make_mask(df_row, img.shape)
        if self.mode == 'segmentations':
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            mask = mask[0].permute(2, 0, 1)
        else:
            augmented = self.transforms(image=img)
            img = augmented['image']
        self.count += 1
        
        return img, mask, image_id

    def __len__(self):
        return len(self.df)
    
    def __count__(self):
        return self.count


class TrainDataset(Dataset):
    """This is a segmentation dataset preparing data from annotations and data directory
    """
    #fold = Option(help='Absolute fold path to the dataset', required=True, default="~/.minetorch_dataset/torchvision_mnist")
    upload = Option(help='Upload your trainning images', type='uploader', required=True)
    annotations = Option(type='uploader', help='You may upload a csv file with columns=["image_names", "class_1_labels", "class_2_labels", ..., "class_n_labels"]')
    batch_size = Option(type='number')
    mode = Option(type='collection', default='Classification', collection=['Classification', 'Segmentation'])
    split_ratio = Option(type='number', default=0.2, help='Split your datasets into trainset and valset')
    k_fold = Option(type='number', default=1, help='Number of folds to split from original datasets', required=False)
    dataset = BasicModule(name='Dataset', component_types=['Dataset'])
    train_augmentations = DataflowModule(name='Train Augmentations', component_types=['Dataflow'], multiple=True, required=False)
    val_augmentations = DataflowModule(name='Validation Augmentations', component_types=['Dataflow'], multiple=True, required=False)

    def __call__(self):
        with ZipFile(upload, 'r') as zip_object:
            zip_object.extractall()
        fold = upload.split('.zip')[0]
        df = pd.read_csv(self.annotations)
        train_dfs, val_dfs = Kfold(df, n_splits=self.kfold)
        train_df = train_dfs[0]
        val_df = val_dfs[0]
        return (
            DataLoader(dataset=TorchDatasetBoth(annotation=train_df, data_folder=fold, transforms=self.train_augmentations, mode=self.mode), batch_size=self.batch_size),
            DataLoader(dataset=TorchDatasetBoth(annotation=val_df, data_folder=fold, transforms=self.val_augmentations, mode=self.mode), batch_size=self.batch_size)
        )
