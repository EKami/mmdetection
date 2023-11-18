import os
from pathlib import Path

import pytorch_lightning as pl
from mmcv import LoadImageFromFile, RandomResize
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from mmengine.registry import TRANSFORMS, MODELS
from mmdet.datasets.transforms import transforms
from mmdet.datasets.transforms import LoadAnnotations, RandomCrop, YOLOXHSVRandomAug, RandomFlip, \
    Pad, PackDetInputs
from mmdet.evaluation import CocoMetric
from mmdet.models import DetDataPreprocessor
from my_script.datasets import CustomCocoDataset
from my_script.download_data import download_pretrained_model
from my_script.model import _get_rmdet_model
from my_script.runner import BBoxPlModel

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = (cur_dir / "..").resolve()


def _get_datasets(data_root):
    train_dataset = CustomCocoDataset(data_root, step='train')
    val_dataset = CustomCocoDataset(data_root, step='val')
    return train_dataset, val_dataset


def _add_registry_modules():
    TRANSFORMS.register_module('RandomCrop', module=transforms.RandomCrop)
    TRANSFORMS.register_module('CachedMosaic', module=transforms.CachedMosaic)
    TRANSFORMS.register_module('YOLOXHSVRandomAug', module=transforms.YOLOXHSVRandomAug)
    TRANSFORMS.register_module('CachedMixUp', module=transforms.CachedMixUp)
    TRANSFORMS.register_module('PackDetInputs', module=PackDetInputs)

    MODELS.register_module('DetDataPreprocessor', module=DetDataPreprocessor)


def prepare_config(data_root):
    cpk_file = download_pretrained_model()
    _add_registry_modules()

    checkpoints_dir = root_dir / "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_batch_size_per_gpu = 4
    train_num_workers = 2

    max_epochs = 20
    stage2_num_epochs = 1
    base_lr = 0.00008

    # ----------------------------------------------

    train_ds, val_ds = _get_datasets(data_root)
    pytorch_model = _get_rmdet_model(cpk_file)
    train_loader = DataLoader(
        train_ds,
        train_batch_size_per_gpu,
        drop_last=False,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=True
    )
    # Doesn't use gradients, so we can double the batch size
    val_loader = DataLoader(
        val_ds,
        train_batch_size_per_gpu * 2,
        drop_last=False,
        shuffle=False,
        num_workers=train_num_workers,
        pin_memory=True,
    )

    # Switches the pipeline on the last epoch... okay
    train_pipeline_stage2 = [
        LoadImageFromFile(backend_args=None),
        LoadAnnotations(with_bbox=True),
        RandomResize(scale=(640, 640), ratio_range=(0.1, 2.0), keep_ratio=True),
        RandomCrop(crop_size=(640, 640)),
        YOLOXHSVRandomAug(),
        RandomFlip(prob=0.5),
        Pad(size=(640, 640), pad_val=dict(img=(114, 114, 114))),
        PackDetInputs()
    ]
    # TODO the pipeline switches at the last epoch?
    custom_hooks = [
        dict(
            type='PipelineSwitchHook',
            switch_epoch=max_epochs - stage2_num_epochs,
            switch_pipeline=train_pipeline_stage2
        )
    ]

    metrics = [
        CocoMetric(
            ann_file=data_root / 'val.json',
            metric='bbox',
            proposal_nums=(100, 1, 10),
        )
    ]
    wandb_logger = WandbLogger(project="mmdetection")
    trainer_model = BBoxPlModel(pytorch_model, base_lr, metrics)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        save_top_k=1,
        monitor="val_dice_coeff",
        mode="max",
        verbose=True,
        auto_insert_metric_name=True,
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        default_root_dir=checkpoints_dir,
        precision="16-mixed",
        log_every_n_steps=10,
        enable_checkpointing=True,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        num_sanity_val_steps=0
    )
    trainer.fit(model=trainer_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def main():
    print('Start training')
    data_root = Path("/mnt/ext_datasets/mmdetection/data/balloon")
    prepare_config(data_root)


if __name__ == '__main__':
    main()
