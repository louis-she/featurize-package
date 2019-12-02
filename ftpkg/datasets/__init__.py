from featurize_jupyterlab.core import Dataset, Option, Task, BasicModule, DataflowModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset


class TorchSegDataset(TorchDataset):
    
    def __init__(self, annotation, data_folder, transforms):
        self.df = pd.read_csv(annotation)
        self.root = data_folder
        self.transforms = transforms
        self.fnames = self.df.columns[0]
        self.classes = self.df.columns[1:len(df.columns)+1]
        self.num_classes = len(self.classes)

    def make_mask(self, df_row, shape):
        
        labels = df_row[1:self.num_classes + 1]
        masks = np.zeros(shape, dtype=np.float32)

        for idx, label in enumerate(labels.values):
            if label is not np.nan:
                label = label.split(" ")
                positions = map(int, label[0::2])
                length = map(int, label[1::2])
                mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
                for pos, le in zip(positions, length):
                    mask[pos:(pos + le)] = 1
                masks[:, :, idx] = mask.reshape(shape[0], shape[1], order='F')
        return masks
    
    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        image_id = self.df.iloc[idx][self.fnames]
        image_path = os.path.join(self.root, image_id)
        img = cv2.imread(image_path)
        mask = self.make_mask(df_row, img.shape)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        mask = mask[0].permute(2, 0, 1)
        return img, mask, image_id

    def __len__(self):
        return len(self.df)


class SegmentationDataset(Dataset):
    """This is a segmentation dataset preparing data from annotations and data directory
    """
    fold = Option(help='Absolute fold path to the dataset', required=True, default="~/.minetorch_dataset/torchvision_mnist")
    annotations = Option(type='uploader', help='You may upload a csv file with columns=["image_names", "class_1", "class_2", ..., "class_n"]')
    batch_size = Option(type='number')
    dataset = BasicModule(name='Dataset', component_types=['Dataset'])
    train_augmentations = DataflowModule(name='Train Augmentations', component_types=['Dataflow'], multiple=True, required=False)
    val_augmentations = DataflowModule(name='Validation Augmentations', component_types=['Dataflow'], multiple=True, required=False)

    def __call__(self):
        return (
            DataLoader(dataset=TrochSegDataset(annotation=self.annotations, data_folder=self.fold, transforms=self.train_augmentations), batch_size=self.batch_size),
            DataLoader(dataset=TrochSegDataset(annotation=self.annotations, data_folder=self.fold, transforms=self.val_augmentations), batch_size=self.batch_size)
        )
