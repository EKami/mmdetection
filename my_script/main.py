import os
from pathlib import Path

import pytorch_lightning as pl
from mmcv import LoadImageFromFile, RandomResize
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from mmdet.datasets import CocoDataset
from mmdet.datasets.transforms import LoadAnnotations, RandomCrop, YOLOXHSVRandomAug, RandomFlip, \
    Pad, PackDetInputs
from mmdet.evaluation import CocoMetric
from mmdet.models import RTMDet
from my_script.download_data import download_balloon_ds, download_pretrained_model
from my_script.runner import BBoxPlModel

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = (cur_dir / "..").resolve()


def _get_model():
    base_checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
    backbone = {
        'type': 'CSPNeXt',
        'arch': 'P5',
        'expand_ratio': 0.5,
        'deepen_factor': 0.167,
        'widen_factor': 0.375,
        'channel_attention': True,
        'norm_cfg': {'type': 'SyncBN'},
        'act_cfg': {'type': 'SiLU', 'inplace': True},
        'init_cfg': {
            'type': 'Pretrained', 'prefix': 'backbone.',
            'checkpoint': base_checkpoint
        }
    }
    neck = {
        'type': 'CSPNeXtPAFPN',
        'in_channels': [96, 192, 384],
        'out_channels': 96,
        'num_csp_blocks': 1,
        'expand_ratio': 0.5,
        'norm_cfg': {'type': 'SyncBN'},
        'act_cfg': {'type': 'SiLU', 'inplace': True}
    }
    bbox_head = {
        'type': 'RTMDetSepBNHead',
        'num_classes': 1,
        'in_channels': 96,
        'stacked_convs': 2,
        'feat_channels': 96,
        'anchor_generator': {
            'type': 'MlvlPointGenerator',
            'offset': 0,
            'strides': [8, 16, 32]
        },
        'bbox_coder': {
            'type': 'DistancePointBBoxCoder'
        },
        'loss_cls': {
            'type': 'QualityFocalLoss',
            'use_sigmoid': True,
            'beta': 2.0,
            'loss_weight': 1.0
        },
        'loss_bbox': {'type': 'GIoULoss', 'loss_weight': 2.0},
        'with_objectness': False,
        'exp_on_reg': False,
        'share_conv': True,
        'pred_kernel_size': 1,
        'norm_cfg': {'type': 'SyncBN'},
        'act_cfg': {'type': 'SiLU', 'inplace': True}
    }
    train_cfg = {
        'assigner': {'type': 'DynamicSoftLabelAssigner', 'topk': 13},
        'allowed_border': -1,
        'pos_weight': -1,
        'debug': False
    }
    test_cfg = {
        'nms_pre': 30000,
        'min_bbox_size': 0,
        'score_thr': 0.001,
        'nms': {'type': 'nms', 'iou_threshold': 0.65},
        'max_per_img': 300
    }
    data_preprocessor = {
        'type': 'DetDataPreprocessor',
        'mean': [103.53, 116.28, 123.675],
        'std': [57.375, 57.12, 58.395],
        'bgr_to_rgb': False,
        'batch_augments': None
    }

    model = RTMDet(
        backbone=backbone,
        neck=neck,
        bbox_head=bbox_head,
        train_cfg=train_cfg,
        test_cfg=test_cfg,
        data_preprocessor=data_preprocessor
    )
    return model


def _get_datasets(data_root):
    metainfo = {
        'classes': ('balloon',),
        'palette': [
            (220, 20, 60),
        ]
    }
    train_dataset = CocoDataset(
        data_root=data_root,
        ann_file='train.json',
        data_prefix={'img': 'train/'},
        filter_cfg={
            'filter_empty_gt': True,
            'min_size': 32
        },
        # Same transforms as above
        pipeline=[
            {
                'type': 'LoadImageFromFile',
                'backend_args': None
            }, {
                'type': 'LoadAnnotations',
                'with_bbox': True
            }, {
                'type': 'CachedMosaic',
                'img_scale': (640, 640),
                'pad_val': 114.0,
                'max_cached_images': 20,
                'random_pop': False
            }, {
                'type': 'RandomResize',
                'scale': (1280, 1280),
                'ratio_range': (0.5, 2.0),
                'keep_ratio': True
            }, {
                'type': 'RandomCrop',
                'crop_size': (640, 640)
            }, {
                'type': 'YOLOXHSVRandomAug'
            }, {
                'type': 'RandomFlip',
                'prob': 0.5
            }, {
                'type': 'Pad',
                'size': (640, 640),
                'pad_val': {'img': (114, 114, 114)}
            }, {
                'type': 'CachedMixUp',
                'img_scale': (640, 640),
                'ratio_range': (1.0, 1.0),
                'max_cached_images': 10,
                'random_pop': False,
                'pad_val': (114, 114, 114),
                'prob': 0.5
            }, {
                'type': 'PackDetInputs'
            }
        ],
        metainfo=metainfo
    )
    val_dataset = CocoDataset(
        data_root=data_root,
        ann_file='val.json',
        data_prefix={'img': 'val/'},
        test_mode=True,
        pipeline=[
            {
                'type': 'LoadImageFromFile',
                'backend_args': None
            }, {
                'type': 'Resize',
                'scale': (640, 640),
                'keep_ratio': True
            }, {
                'type': 'Pad',
                'size': (640, 640),
                'pad_val': {
                    'img': (114, 114, 114)
                }
            }, {
                'type': 'LoadAnnotations',
                'with_bbox': True
            }, {
                'type': 'PackDetInputs',
                'meta_keys': (
                    'img_id', 'img_path',
                    'ori_shape', 'img_shape',
                    'scale_factor'
                )
            }
        ],
        metainfo=metainfo
    )
    return train_dataset, val_dataset


def _get_cfg(data_root, base_lr, max_epochs, train_batch_size_per_gpu, train_num_workers):
    # _base_ = './rtmdet_tiny_8xb32-300e_coco.py'

    metainfo = {
        'classes': ('balloon',),
        'palette': [
            (220, 20, 60),
        ]
    }

    train_dataloader = dict(
        batch_size=train_batch_size_per_gpu,
        num_workers=train_num_workers,
        dataset=dict(
            data_root=data_root,
            metainfo=metainfo,
            data_prefix=dict(img='train/'),
            ann_file='train.json')
    )

    val_dataloader = dict(
        dataset=dict(
            data_root=data_root,
            metainfo=metainfo,
            data_prefix=dict(img='val/'),
            ann_file='val.json')
    )

    test_dataloader = val_dataloader

    val_evaluator = dict(ann_file=data_root / 'val.json')

    test_evaluator = val_evaluator

    model = dict(bbox_head=dict(num_classes=1))

    # learning rate
    param_scheduler = [
        dict(
            type='LinearLR',
            start_factor=1.0e-5,
            by_epoch=False,
            begin=0,
            end=10),
        dict(
            # use cosine lr from 10 to 20 epoch
            type='CosineAnnealingLR',
            eta_min=base_lr * 0.05,
            begin=max_epochs // 2,
            end=max_epochs,
            T_max=max_epochs // 2,
            by_epoch=True,
            convert_to_iter_based=True),
    ]

    train_pipeline_stage2 = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='RandomResize',
            scale=(640, 640),
            ratio_range=(0.1, 2.0),
            keep_ratio=True),
        dict(type='RandomCrop', crop_size=(640, 640)),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', prob=0.5),
        dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
        dict(type='PackDetInputs')
    ]

    # optimizer
    # optim_wrapper = dict(
    #     _delete_=True,
    #     type='OptimWrapper',
    #     optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    #     paramwise_cfg=dict(
    #         norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
    # )

    # default_hooks = dict(
    #     checkpoint=dict(
    #         interval=5,
    #         max_keep_ckpts=2,  # only keep latest 2 checkpoints
    #         save_best='auto'
    #     ),
    #     logger=dict(type='LoggerHook', interval=5)
    # )

    # custom_hooks = [
    #     dict(
    #         type='PipelineSwitchHook',
    #         switch_epoch=max_epochs - stage2_num_epochs,
    #         switch_pipeline=train_pipeline_stage2)
    # ]

    # load COCO pre-trained weight
    load_from = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

    # train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
    # visualizer = dict(
    #     vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

    # runner = Runner(
    #     model=model,
    #     # work_dir=work_dir,
    #     train_dataloader=train_dataloader,
    #     val_dataloader=val_dataloader,
    #     test_dataloader=test_dataloader,
    #     train_cfg=train_cfg,
    #     # val_cfg=val_cfg,
    #     # test_cfg=test_cfg,
    #     # auto_scale_lr=auto_scale_lr,
    #     optim_wrapper=optim_wrapper,
    #     param_scheduler=param_scheduler,
    #     val_evaluator=val_evaluator,
    #     test_evaluator=test_evaluator,
    #     default_hooks=default_hooks,
    #     custom_hooks=custom_hooks,
    #     #data_preprocessor=data_preprocessor,
    #     load_from=load_from,
    #     resume=False,
    #     launcher='none',
    #     env_cfg=dict(dist_cfg=dict(backend='nccl')),
    #     # log_processor=log_processor,
    #     log_level='INFO',
    #     visualizer=visualizer,
    #     default_scope='mmengine',
    #     randomness=dict(seed=None),
    #     experiment_name="Baloon",
    #     cfg=cfg,
    # )


def prepare_config():
    cpk_file = download_pretrained_model()
    data_root = download_balloon_ds()
    checkpoints_dir = root_dir / "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_batch_size_per_gpu = 4
    train_num_workers = 2

    max_epochs = 20
    stage2_num_epochs = 1
    base_lr = 0.00008


    # ----------------------------------------------

    train_ds, val_ds = _get_datasets(data_root)
    pytorch_model = _get_model()
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


    optim_wrapper = dict(
        _delete_=True,
        type='OptimWrapper',
        optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
        paramwise_cfg=dict(
            norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
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
    custom_hooks = [
        dict(
            type='PipelineSwitchHook',
            switch_epoch=max_epochs - stage2_num_epochs,
            switch_pipeline=train_pipeline_stage2
        )
    ]

    metrics = [CocoMetric()]
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
    )
    trainer.fit(model=trainer_model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def main():
    print('Start training')
    prepare_config()


if __name__ == '__main__':
    main()
