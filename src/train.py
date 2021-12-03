from src.datamodule import DataModule
from src.model import Model

from pathlib import Path

# lightning related imports
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import datetime
import warnings

warnings.filterwarnings("ignore")


def train_test(data_path, fold_number, batch_size, init_lr, max_epochs, early_stop_patience, sigma, lam):
    train_file = 'data/acne4/VOCdevkit2007/VOC2007/ImageSets/Main/NNEW_trainval_' + str(fold) + '.txt'
    test_file = 'data/acne4/VOCdevkit2007/VOC2007/ImageSets/Main/NNEW_test_' + str(fold) + '.txt'

    dm = DataModule(data_path=data_path, train_file=train_file, test_file=test_file, batch_size=batch_size)
    dm.setup()

    model = Model(sigma=sigma, lam=lam, learning_rate=init_lr)

    experiment_name = "acne_fold" + str(fold_number)
    logger = TensorBoardLogger('tb_logs', name=experiment_name)

    checkpoint_name = experiment_name + '_{epoch}_{test_loss:.4f}_{test_acc:.3f}'

    checkpoint_callback_loss = ModelCheckpoint(monitor='test_loss', mode='min',
                                               filename=checkpoint_name,
                                               verbose=True, save_top_k=1,
                                               save_last=False)
    checkpoint_callback_acc = ModelCheckpoint(monitor='test_acc', mode='max',
                                              filename=checkpoint_name,
                                              verbose=True, save_top_k=1,
                                              save_last=False)

    early_stop_callback = EarlyStopping(
        monitor='test_loss',
        patience=early_stop_patience,
        verbose=True,
        mode='min'
    )

    checkpoints = [checkpoint_callback_acc, checkpoint_callback_loss, early_stop_callback]
    callbacks = checkpoints

    trainer = pl.Trainer(max_epochs=max_epochs,
                         progress_bar_refresh_rate=2,
                         gpus=1,
                         logger=logger,
                         # auto_lr_find=True,
                         callbacks=callbacks)

    # Train the model ⚡🚅⚡
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    trainer.test(model, dm.test_dataloader())


if __name__ == '__main__':
    folds = [0, 1, 2, 3, 4]

    sigma = 30 * 0.1
    lam = 6 * 0.1

    data_path = 'data/acne4/VOCdevkit2007/VOC2007/ImageSets/Main/JPEGImages'
    batch_size = 32
    init_lr = 0.0001
    max_epochs = 1
    early_stop_patience = 6

    for fold in folds:
        train_test(data_path,
                   fold,
                   batch_size,
                   init_lr,
                   max_epochs,
                   early_stop_patience,
                   sigma,
                   lam)
