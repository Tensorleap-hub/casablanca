
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.io import read_video


def make_video(imgs, out_path, fps=10):
    h, w, c = imgs[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for img in imgs:
        video.write(img)
    video.release()



class Preprocessor:
    def __init__(self, size=256):
        super(Preprocessor, self).__init__()
        self.size = size

    def image2torch(self, filename) -> torch.Tensor: # 1CHW, [-1, 1]
        img = Image.open(filename).convert('RGB')
        img = img.resize((self.size, self.size))
        img = np.asarray(img)
        img = np.transpose(img, (2, 0, 1))/255.0
        img = (img - 0.5) * 2.0
        img = img[np.newaxis, ...]
        return torch.tensor(img, dtype=torch.float32).cuda()

    def vid2torch(self, vid_path) -> Tuple[torch.Tensor, float]:
        vid_dict = read_video(vid_path, pts_unit='sec')
        vid = vid_dict[0].permute(0, 3, 1, 2)
        if vid.shape[2] != self.size:
            vid = nn.functional.interpolate(vid, size=(self.size, self.size), mode='bilinear', align_corners=False)
        vid = vid.unsqueeze(0)
        vid_norm = (vid / 255.0 - 0.5) * 2.0

        return vid_norm.cuda(), vid_dict[2]['video_fps']
