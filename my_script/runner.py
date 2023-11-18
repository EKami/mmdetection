import os

import tempfile
from mmdet.apis import DetInferencer
from pathlib import Path
import os.path as osp
import mmcv
from mmdet.utils import setup_cache_size_limit_of_dynamo
from tqdm import tqdm

from mmengine.fileio import dump, load
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from my_script.download_data import download_pretrained_model, download_balloon_ds

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = (cur_dir / "..").resolve()


def _inference_demo():
    print('Start inference demo...')
    tmp_dir = Path(tempfile.mkdtemp())
    # Choose to use a config
    model_name = 'rtmdet_tiny_8xb32-300e_coco'
    # Setup a checkpoint file to load
    checkpoint = download_pretrained_model(cpk_name=model_name)
    # Set the device to be used for evaluation
    device = 'cuda'

    # Initialize the DetInferencer
    inferencer = DetInferencer(model_name, str(checkpoint), device)

    # Use the detector to do inference
    img = str(root_dir / 'demo' / 'demo.jpg')
    result = inferencer(img, out_dir=str(tmp_dir))
    img_path = tmp_dir / 'vis' / 'demo.jpg'
    print(f"Resulting image can be found in {img_path}")
    return result


def convert_balloon_to_coco(ann_file, out_file, image_prefix):
    data_infos = load(ann_file)

    annotations = []
    images = []
    obj_count = 0
    for idx, v in enumerate(tqdm(data_infos.values())):
        filename = v['filename']
        img_path = osp.join(image_prefix, filename)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(
            dict(id=idx, file_name=filename, height=height, width=width))

        for _, obj in v['regions'].items():
            assert not obj['region_attributes']
            obj = obj['shape_attributes']
            px = obj['all_points_x']
            py = obj['all_points_y']
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=0,
                bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                area=(x_max - x_min) * (y_max - y_min),
                segmentation=[poly],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{
            'id': 0,
            'name': 'balloon'
        }])
    dump(coco_format_json, out_file)


def _train(config_file, work_dir=None):
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(config_file)
    cfg.launcher = 'none'

    if work_dir is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_file))[0])
    else:
        cfg.work_dir = str(work_dir)

    runner = Runner.from_cfg(cfg)
    # start training
    model = runner.train()
    print(f"Finished training!")


def _custom_training_demo():
    # Now train a custom model
    print('Start training...')
    data_dir = download_balloon_ds()
    convert_balloon_to_coco(
        ann_file=data_dir / 'train' / 'via_region_data.json',
        out_file=data_dir / 'train.json',
        image_prefix=data_dir / 'train'
    )
    convert_balloon_to_coco(
        ann_file=data_dir / 'val' / 'via_region_data.json',
        out_file=data_dir / 'val.json',
        image_prefix=data_dir / 'val'
    )
    config_file = cur_dir / 'configs' / 'rtmdet_tiny_1xb4-20e_balloon.py'
    with tempfile.TemporaryDirectory() as tmp_dirname:
        _train(config_file, work_dir=tmp_dirname)

def prepare_config(data_root):
    #_inference_demo()
    _custom_training_demo()

def main():
    data_root = Path("/tmp/balloon")
    os.makedirs(data_root, exist_ok=True)
    prepare_config(data_root)


if __name__ == '__main__':
    main()