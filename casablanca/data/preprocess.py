import os.path

import numpy as np
from itertools import product
import pandas as pd

# Tensorleap imports
from casablanca.config import CONFIG
from casablanca.utils.gcs_utils import _download


def _load_ids_metadata():
    fpath = CONFIG['metadata_filepath']
    df = pd.read_json(_download(fpath), orient='records')
    return df


def list_indices():
    config = CONFIG['dataset_creation']
    # get frames from config
    frames = config['frames']
    # generate list of frames indices
    frames_indices = list(range(frames['initial'], frames['final'] + 1, frames['step']))
    return frames_indices


def load_data():
    config = CONFIG['dataset_creation']
    files_df = pd.read_csv(_download(CONFIG['data_filepath'], overwrite=True), sep='\t')
    # get selected ids from config
    selected_ids = set(config['ids'])
    frames_indices = list_indices()
    # filter files_df by selected ids
    files_df = files_df[files_df['id'].isin(selected_ids)]
    # select n_clips_per_video clips per video and n_videos_per_id videos per id
    files_df = files_df.groupby(['id', 'vid_name']).head(config['n_clips_per_video']).groupby(
        'id').head(config['n_videos_per_id'])
    # create mapping from person id to path
    id_to_path = files_df[['id', 'path']].drop_duplicates().groupby('id').head(1).set_index('id')
    path_to_id = files_df[['id', 'path']].set_index('path')
    # create mapping from video id to path
    vid_to_path = files_df[['vid_name', 'path']].drop_duplicates().set_index('vid_name')
    # create mapping from person id to video id
    person_to_videos = files_df[['id', 'vid_name']].drop_duplicates()
    # create all possible combinations of source_id, vid_name and frame_id
    all_combinations = pd.DataFrame(
        [x for x in product(person_to_videos['id'].unique(), person_to_videos['vid_name'].unique())],
        columns=['source_id', 'vid_name']
    )
    # repeat all_combinations for each frame
    repeated_df = pd.DataFrame(np.repeat(all_combinations.values, len(frames_indices), axis=0),
                               columns=all_combinations.columns)
    # add frame_id, source_path and vid_path to repeated_df
    repeated_df['frame_id'] = np.tile(frames_indices, len(all_combinations))
    repeated_df['source_path'] = repeated_df['source_id'].map(id_to_path['path'])
    repeated_df['vid_path'] = repeated_df['vid_name'].map(vid_to_path['path'])
    repeated_df['vid_id'] = repeated_df['vid_path'].map(path_to_id['id'])

    # add frame path column
    repeated_df['frame_path'] = repeated_df.apply(
        lambda row: f"{os.path.splitext(row['vid_path'])[0]}{CONFIG['frame_separator']}{row['frame_id']}.png", axis=1)
    # add gender metadata
    ids_metadata = _load_ids_metadata().set_index("id")
    repeated_df['source_gender'] = repeated_df['source_id'].map(ids_metadata['gender'])
    repeated_df['vid_gender'] = repeated_df['vid_id'].map(ids_metadata['gender'])

    # assertions to make sure the data is as expected
    n_occurrences_per_id_expected = config['n_videos_per_id'] * config['n_clips_per_video'] * len(selected_ids) * len(
        frames_indices)
    n_occurrences_per_id_actual = repeated_df['source_id'].value_counts().unique()
    n_samples_expected = len(selected_ids) ** 2 * len(frames_indices) * config['n_clips_per_video'] * config[
        'n_videos_per_id']
    if n_occurrences_per_id_actual[0] != n_occurrences_per_id_expected:
        raise ValueError('The number of occurrences per id is not as expected'
                         f'Expected: {n_occurrences_per_id_expected} got {n_occurrences_per_id_actual[0]}')
    if len(n_occurrences_per_id_actual) != 1:
        raise ValueError('The number of occurrences per id is not the same for all ids')
    if repeated_df.shape[0] != n_samples_expected:
        raise ValueError('The number of samples is not as expected'
                         f'Expected: {n_samples_expected} got {repeated_df.shape[0]}')

    return repeated_df


def load_data_all():
    files_df = pd.read_csv(_download(CONFIG['data_filepath'], overwrite=True), sep='\t')
    # frames_indices = list_indices()
    frames_indices = CONFIG['dataset_settings']['more_frames']['frames_indices']
    repeated_df = files_df.loc[files_df.index.repeat(len(frames_indices))].reset_index(drop=True)
    repeated_frames = np.tile(frames_indices, len(files_df))
    repeated_df['frame_id'] = repeated_frames
    repeated_df['frame_path'] = repeated_df.apply(
        lambda row: f"{os.path.splitext(row['path'])[0]}{CONFIG['frame_separator']}{row['frame_id']}.png", axis=1)

    return repeated_df

