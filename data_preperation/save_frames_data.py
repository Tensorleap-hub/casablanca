import pandas as pd
from tqdm import tqdm
from functools import lru_cache
from os import environ
import json
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud.storage import Bucket
from typing import Optional
import os
import cv2
from casablanca.config import CONFIG
from casablanca.data.preprocess import _download, list_indices
from casablanca.utils.gcs_utils import download
from casablanca.utils.general_utils import input_video
from code_loader.utils import rescale_min_max




if __name__ == "__main__":
    config = CONFIG['dataset_creation']
    files_df = pd.read_csv(_download(CONFIG['data_filepath']), sep='\t')
    frames_indices = [0] + list_indices()
    for i, row in tqdm(files_df.iterrows()):
        for frame_number in frames_indices:
            path = row['path']
            root, extension = os.path.splitext(path)
            fpath = _download(path)
            save_path = fpath.rsplit('.', 1)[0] + CONFIG['frame_separator'] + str(frame_number) + '.png'
            save_path = save_path.replace('mp4', 'frames')
            if os.path.exists(save_path):
                continue
            # if fpath.rsplit('.', 1)[-1] == 'mp4':
            try:
                frame = input_video(fpath, frame_number)
            except Exception as e:
                print(e, i, path)
                continue
            frame = rescale_min_max(frame.numpy()).transpose((1, 2, 0))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            cv2.imwrite(save_path, frame)

    print(f"Done")
