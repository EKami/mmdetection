import glob
import os
from pathlib import Path

from mim.commands import download as mim_download
from tools.misc.download_dataset import download as mm_download

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = (cur_dir / "..").resolve()

def download_pretrained_model():
    cpk_name = "rtmdet_tiny_8xb32-300e_coco"
    cpk_dir = root_dir / "checkpoints"
    os.makedirs(cpk_dir, exist_ok=True)
    package = "mmdet"
    mim_download(package, [cpk_name], str(cpk_dir))
    cpk_file = Path(glob.glob(str(cpk_dir / f"{cpk_name}*.pth"))[0])
    return cpk_file

def download_balloon_ds():
    # !python tools/misc/download_dataset.py --dataset-name balloon --save-dir data --unzip
    url = "https://download.openmmlab.com/mmyolo/data/balloon_dataset.zip"
    to_dir = root_dir / "data"
    os.makedirs(to_dir, exist_ok=True)
    mm_download(url, to_dir, unzip=True, delete=False, threads=1)
    ret_dir = to_dir / "balloon"
    return ret_dir