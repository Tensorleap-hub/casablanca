import numpy as np
from itertools import product
import pandas as pd

# Tensorleap imports
from casablanca.config import CONFIG
from casablanca.utils.gcs_utils import _download


def load_data():
    config = CONFIG['dataset_creation']
    files_df = pd.read_csv(_download(CONFIG['data_filepath']), sep='\t')
    # get selected ids from config
    selected_ids = set(config['ids'])
    # get frames from config
    frames = config['frames']
    # generate list of frames indices
    frames_indices = list(range(frames['initial'], frames['final'] + 1, frames['step']))
    # filter files_df by selected ids
    files_df = files_df[files_df['id'].isin(selected_ids)]
    # select n_clips_per_video clips per video and n_videos_per_id videos per id
    files_df = files_df.groupby(['id', 'vid_name']).head(config['n_clips_per_video']).groupby(
        'id').head(config['n_videos_per_id'])
    # create mapping from person id to path
    id_to_path = files_df[['id', 'path']].drop_duplicates().groupby('id').head(1).set_index('id')
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
