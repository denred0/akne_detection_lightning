import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from transforms.affine_transforms import RandomRotate

from dataset import DatasetWrapper


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path, train_file, test_file, batch_size):
        super().__init__()
        self.data_path = data_path
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size

        self.dataset_train, self.dataset_val, self.dataset_test = None, None, None

    @staticmethod
    def get_train_transforms():
        train_transforms = transforms.Compose([
            transforms.Scale((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RandomRotate(rotation_range=20),
            transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                 std=[0.2814769, 0.226306, 0.20132513]),
        ])

        return train_transforms

    @staticmethod
    def get_valid_transforms():
        valid_transforms = transforms.Compose([
            transforms.Scale((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                 std=[0.2814769, 0.226306, 0.20132513]),
        ])

        return valid_transforms

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.dataset_train = DatasetWrapper(
                self.data_path,
                self.train_file,
                self.get_train_transforms())

            self.dataset_val = DatasetWrapper(
                self.data_path,
                self.test_file,
                self.get_valid_transforms())

            self.dims = tuple(self.dataset_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.dataset_test = DatasetWrapper(
                self.data_path,
                self.test_file,
                self.get_valid_transforms())

            self.dims = tuple(self.dataset_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)
