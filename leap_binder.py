from typing import List

import numpy as np
import torch
from typing import Tuple

import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.io import read_video
from casablanca.config import CONFIG
import os
import glob
import subprocess

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import Metric, DatasetMetadataType
from code_loader.contract.visualizer_classes import LeapHorizontalBar

from casablanca.data.preprocess import load_data
from casablanca.utils.gcs_utils import download


# Preprocess Function
def preprocess_func() -> List[PreprocessResponse]:
    train_videos = load_data()
    train_size = min(len(train_videos), CONFIG['train_size'])

    train = PreprocessResponse(length=train_size, data={'videos': train_videos})
    response = [train]
    return response


def convert(video):
    # Converting files from AAC to WAV
    # for fname in tqdm(video):
    outfile = video.replace('.m4a', '.wav')
    out = subprocess.call(
        'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' % (video, outfile),
        shell=True)
    if out != 0:
        raise ValueError('Conversion failed %s.' % video)


def input_encoder_image(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    filename = preprocess.data['samples'][idx]['file_name']
    img = Image.open(filename).convert('RGB')
    img = img.resize((256, 256))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1)) / 255.0
    img = (img - 0.5) * 2.0
    img = img[np.newaxis, ...]
    return torch.tensor(img, dtype=torch.float32)


# def input_encoder_video(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    # vid_path = preprocess.data['videos'][idx]
    # fpath = download(str(vid_path), CONFIG['BUCKET_NAME'])
def input_encoder_video(idx, preprocess) -> np.ndarray:
    fpath = '/Users/chenrothschild/Downloads/vox2_dev_mp4_partab'
    # convert_video = convert(fpath)
    vid_dict = read_video(fpath, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2)
    if vid.shape[2] != 256:
        vid = nn.functional.interpolate(vid, size=(256, 256), mode='bilinear', align_corners=False)
    vid = vid.unsqueeze(0)
    vid_norm = (vid / 255.0 - 0.5) * 2.0

    return vid_norm, vid_dict[2]['video_fps']


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return preprocessing.data['labels'][idx].astype('float32')


# Dataset binding functions to bind the functions above to the `Dataset Instance`.
leap_binder.set_preprocess(function=preprocess_func)
leap_binder.set_input(function=input_encoder_image, name='image')
leap_binder.set_input(function=input_encoder_video, name='video')

leap_binder.set_ground_truth(function=gt_encoder, name='classes')
# leap_binder.add_prediction(name='classes', labels=LABELS)
