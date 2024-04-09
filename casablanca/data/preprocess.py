from typing import Tuple
from pathlib import Path

# Tensorleap imports
from code_loader.contract.datasetclasses import PreprocessResponse
from casablanca.config import CONFIG
from casablanca.utils.gcs_utils import _connect_to_gcs_and_return_bucket
import json


def load_data():
    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    file_names_all = [blob.name for blob in bucket.list_blobs(prefix=str(Path(CONFIG['all_images_path']) / 'data'))]

    return file_names_all
