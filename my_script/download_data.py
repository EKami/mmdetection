import os
import re
import sys
from pathlib import Path

from mim.commands import download as mim_download
from tools.misc.download_dataset import download

cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
root_dir = (cur_dir / "..").resolve()

def download_pretrained_model():
    cpk_dir = root_dir / "checkpoints"
    os.makedirs(cpk_dir, exist_ok=True)
    package = "mmdet"
    configs = ["rtmdet_tiny_8xb32-300e_coco"]
    mim_download(package, configs, str(cpk_dir))

def download_balloon_ds():
    pass