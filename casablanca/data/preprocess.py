from pathlib import Path

import pandas as pd

# Tensorleap imports
from casablanca.config import CONFIG
from casablanca.utils.gcs_utils import _connect_to_gcs_and_return_bucket, _download
import os
import json
import torch.nn as nn
from torchvision.io import read_video
import numpy as np


def get_ids(file_names_all) -> list:
    ids = [(file.split('/')[-3])[2:] for file in file_names_all]
    ids_set = {*ids}
    ids = list(ids_set)
    ids = ['id' + number for number in ids]
    return ids


def load_data():
    files_df = pd.read_csv(_download(CONFIG['data_filepath']), sep='\t')
    selected_ids = CONFIG['dataset_creation']['ids']
    frames = CONFIG['dataset_creation']['frames']
    frames_indices = list(range(frames['initial'], frames['final'] + 1, frames['step']))
    files_df = files_df[files_df['id'].isin(selected_ids)]
    files_df = files_df.groupby(['id', 'vid_name']).head(CONFIG['dataset_creation']['n_clips_per_video']).groupby(
        'id').head(CONFIG['dataset_creation']['n_videos_per_id'])
    repeated_df = pd.DataFrame(np.repeat(files_df.values, len(frames_indices), axis=0), columns=files_df.columns)
    repeated_df['frame_id'] = np.tile(frames_indices, len(files_df))
    return repeated_df


def count_frames(video_path):
    vid_dict = read_video(video_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2)
    return vid.shape[0]


def count_videos(directory, state):
    frame_counts = {}
    for subdir in directory:
        for folder in os.listdir(subdir):
            folder_path = os.path.join(subdir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.mp4'):
                        video_path = os.path.join(subdir, folder, file)
                        num_frames = count_frames(video_path)
                        frame_counts[subdir.split('/')[-1] + '/' + folder + '/' + file] = num_frames
        save_frame_counts(frame_counts, f'{state}.json')

    return frame_counts


def save_frame_counts(frame_counts, output_file):
    with open(output_file, 'w') as f:
        json.dump(frame_counts, f, indent=4)


def extract_id(directory):
    return int((directory.split('/')[-1])[2:])


if __name__ == '__main__':
    state = 'dev'
    video_directory = f'/Users/chenrothschild/Tensorleap/data/casablanca/{state}/mp4'
    dirs = [os.path.join(video_directory, d) for d in os.listdir(video_directory) if d != '.DS_Store']
    sorted_dirs = sorted(dirs, key=extract_id)
    frame_counts = count_videos(dirs, state)
    save_frame_counts(frame_counts, f'{state}.json')
