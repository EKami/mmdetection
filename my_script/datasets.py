from mmdet.datasets import CocoDataset


class CustomCocoDataset(CocoDataset):
    def __init__(self, data_root, step):
        self.step = step
        metainfo = {
            'classes': ('balloon',),
            'palette': [
                (220, 20, 60),
            ]
        }

        if step == 'train':
            kwargs = dict(
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
                    },
                    {
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    },
                    {
                        'type': 'CachedMosaic',
                        'img_scale': (640, 640),
                        'pad_val': 114.0,
                        'max_cached_images': 20,
                        'random_pop': False
                    },
                    {
                        'type': 'RandomResize',
                        'scale': (1280, 1280),
                        'ratio_range': (0.5, 2.0),
                        'keep_ratio': True
                    },
                    {
                        'type': 'RandomCrop',
                        'crop_size': (640, 640)
                    },
                    {
                        'type': 'YOLOXHSVRandomAug'
                    },
                    {
                        'type': 'RandomFlip',
                        'prob': 0.5
                    },
                    {
                        'type': 'Pad',
                        'size': (640, 640),
                        'pad_val': {'img': (114, 114, 114)}
                    },
                    {
                        'type': 'CachedMixUp',
                        'img_scale': (640, 640),
                        'ratio_range': (1.0, 1.0),
                        'max_cached_images': 10,
                        'random_pop': False,
                        'pad_val': (114, 114, 114),
                        'prob': 0.5
                    },
                    {
                        'type': 'PackDetInputs'
                    }
                ],
                metainfo=metainfo
            )
        elif step == 'val':
            kwargs = dict(
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
        else:
            raise ValueError(f'Unknown step {step}')
        super().__init__(**kwargs)

    def __getitem__(self, idx: int):
        # Runs the transforms on the image
        items = super().__getitem__(idx)
        img = items['inputs']
        bboxes = items['data_samples']['gt_instances']['bboxes']
        labels = items['data_samples']['gt_instances']['labels']
        # TODO return tensors here
        return img, bboxes, labels