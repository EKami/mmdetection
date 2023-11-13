import os
from pathlib import Path

from my_script.download_data import download_balloon_ds, download_pretrained_model
from my_script.runner import Runner

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = (cur_dir / "..").resolve()

def prepare_config(data_root, train_ann_file, val_ann_file):
    download_pretrained_model()
    download_balloon_ds()

    _base_ = './rtmdet_tiny_8xb32-300e_coco.py'

    train_batch_size_per_gpu = 4
    train_num_workers = 2

    max_epochs = 20
    stage2_num_epochs = 1
    base_lr = 0.00008

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
            ann_file='train.json'))

    val_dataloader = dict(
        dataset=dict(
            data_root=data_root,
            metainfo=metainfo,
            data_prefix=dict(img='val/'),
            ann_file='val.json'))

    test_dataloader = val_dataloader

    val_evaluator = dict(ann_file=data_root + 'val.json')

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
    optim_wrapper = dict(
        _delete_=True,
        type='OptimWrapper',
        optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
        paramwise_cfg=dict(
            norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

    default_hooks = dict(
        checkpoint=dict(
            interval=5,
            max_keep_ckpts=2,  # only keep latest 2 checkpoints
            save_best='auto'
        ),
        logger=dict(type='LoggerHook', interval=5))

    custom_hooks = [
        dict(
            type='PipelineSwitchHook',
            switch_epoch=max_epochs - stage2_num_epochs,
            switch_pipeline=train_pipeline_stage2)
    ]

    # load COCO pre-trained weight
    load_from = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

    train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
    visualizer = dict(
        vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

    runner = Runner(
        model=model,
        # work_dir=work_dir,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        train_cfg=train_cfg,
        # val_cfg=val_cfg,
        # test_cfg=test_cfg,
        # auto_scale_lr=auto_scale_lr,
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        val_evaluator=val_evaluator,
        test_evaluator=test_evaluator,
        default_hooks=default_hooks,
        custom_hooks=custom_hooks,
        #data_preprocessor=data_preprocessor,
        load_from=load_from,
        resume=False,
        launcher='none',
        env_cfg=dict(dist_cfg=dict(backend='nccl')),
        # log_processor=log_processor,
        log_level='INFO',
        visualizer=visualizer,
        default_scope='mmengine',
        randomness=dict(seed=None),
        experiment_name="Baloon",
        cfg=cfg,
    )


def main():
    print('Start training')
    data_root = cur_dir / "data" / "balloon"
    balloon_train = data_root / "train.json"
    balloon_val = data_root / "val.json"
    prepare_config(str(data_root), str(balloon_train), str(balloon_val))


if __name__ == '__main__':
    main()