from pathlib import Path

# Tensorleap imports
from casablanca.config import CONFIG
from casablanca.utils.gcs_utils import _connect_to_gcs_and_return_bucket
import os
import json
import torch.nn as nn
from torchvision.io import read_video


def load_data(set):
    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    file_names_all = [blob.name for blob in
                      bucket.list_blobs(prefix=str(Path('data') / set / 'mp4'))]

    file_names_all = [file for file in file_names_all if file.split('/')[-1] != '.DS_Store']
    return file_names_all


def count_frames(video_path):
    vid_dict = read_video(video_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2)
    if vid.shape[2] != 256:
        vid = nn.functional.interpolate(vid, size=(256, 256), mode='bilinear', align_corners=False)
    vid = vid.unsqueeze(0)
    vid_norm = (vid / 255.0 - 0.5) * 2.0

    return vid_norm.shape[1]


def count_videos(directory):
    frame_counts = {}
    for subdir in directory:
        file_name = subdir.split('/')[-1]
        for folder in os.listdir(subdir):
            folder_path = os.path.join(subdir, folder)
            file_name = file_name + '/' + folder
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith('.mp4'):
                        file_name = file_name + '/' + file
                        video_path = os.path.join(subdir, folder, file)
                        num_frames = count_frames(video_path)
                        frame_counts[file_name] = num_frames

    return frame_counts


def save_frame_counts(frame_counts, output_file):
    with open(output_file, 'w') as f:
        json.dump(frame_counts, f, indent=4)


def extract_id(directory):
    return int((directory.split('/')[-1])[2:])


if __name__ == '__main__':
    state = 'test'
    video_directory = f'/Users/chenrothschild/Tensorleap/data/casablanca/{state}/mp4'
    dirs = [os.path.join(video_directory, d) for d in os.listdir(video_directory) if d != '.DS_Store']
    sorted_dirs = sorted(dirs, key=extract_id)
    frame_counts = count_videos(dirs)
    # Path to the output JSON file
    output_json_file = f'{state}.json'
    # Save the frame counts dictionary as a JSON file
    save_frame_counts(frame_counts, output_json_file)
