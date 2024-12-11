from casablanca.config import CONFIG
from casablanca.utils.gcs_utils import _connect_to_gcs_and_return_bucket
from pathlib import Path
import pandas as pd
import random
from collections import defaultdict


if __name__ == '__main__':

    sets = ['dev', 'test']
    dfs = []
    for set_name in sets:
        bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
        file_names_all = [blob.name for blob in
                          bucket.list_blobs(prefix=str(Path('data') / set_name / 'mp4'))]
        file_names_all = [file for file in file_names_all if file.split('/')[-1] != '.DS_Store']
        file_names_all = [path for path in file_names_all if path.endswith('.mp4')]

        # Dictionary to group paths by their common prefix
        random.seed(42)
        prefix_dict = defaultdict(list)
        # Group paths by their common prefix (excluding the file name)
        for path in file_names_all:
            prefix = '/'.join(path.split('/')[:-1])  # Get the prefix (excluding the file name)
            prefix_dict[prefix].append(path)

        selected_paths = []
        for prefix, grouped_paths in prefix_dict.items():
            selected_path = random.choice(grouped_paths)
            selected_paths.append(selected_path)

        data = {'subset': [], 'id': [], 'vid_name': [], 'vid_clip_name': [], 'path': []}

        # Process each file path
        for path in selected_paths:
            parts = path.split('/')
            data['subset'].append(parts[1])
            data['id'].append(parts[3])
            data['vid_name'].append(parts[4])
            data['vid_clip_name'].append(parts[5].split('.')[0])
            data['path'].append(path)

        # Create DataFrame
        dfs.append(pd.DataFrame(data))
        print('done with set:', set_name)
    pd.concat(dfs, axis=0).to_csv('files_mapping_all.tsv', index=False, sep='\t')

