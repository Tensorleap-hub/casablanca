from pathlib import Path

# Tensorleap imports
from casablanca.config import CONFIG
from casablanca.utils.gcs_utils import _connect_to_gcs_and_return_bucket
import os
import json
import torch.nn as nn
from torchvision.io import read_video
import numpy as np


def get_ids(file_names_all) -> list:
    ids = [(file.split('/')[-3])[2:] for file in file_names_all]
    ids_set = {*ids}
    ids = list(ids_set)
    ids = ['id'+number for number in ids]
    return ids


def load_data(set):
    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    file_names_all = [blob.name for blob in
                      bucket.list_blobs(prefix=str(Path('data') / set / 'mp4'))]

    file_names_all = [file for file in file_names_all if file.split('/')[-1] != '.DS_Store']
    if CONFIG[f'{set}_size'] is not None:
            if CONFIG[f'{set}_size'] < len(file_names_all):
                np.random.seed(42)
                np.random.shuffle(file_names_all)
                file_names_all = file_names_all[:CONFIG[f'{set}_size']]

    selected_ids = get_ids(file_names_all)

    #TODO: maybe filter just 10 each
    return file_names_all, selected_ids


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
