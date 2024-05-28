from functools import lru_cache
from os import environ
import json
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud.storage import Bucket
from typing import Optional, Any
import os
from concurrent.futures import ThreadPoolExecutor

from casablanca.config import CONFIG


@lru_cache()
def _connect_to_gcs_and_return_bucket(bucket_name: str) -> Bucket:
    auth_secret_string = environ['AUTH_SECRET']
    auth_secret = json.loads(auth_secret_string)
    if type(auth_secret) is dict:
        # getting credentials from dictionary account info
        credentials = service_account.Credentials.from_service_account_info(auth_secret)
    else:
        # getting credentials from path
        credentials = service_account.Credentials.from_service_account_file(auth_secret)
    project = credentials.project_id
    gcs_client = storage.Client(project=project, credentials=credentials)
    return gcs_client.bucket(bucket_name)


def _download(cloud_file_path: str, local_file_path: Optional[str] = None, overwrite: bool = False) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "tensorleap", "data", CONFIG['BUCKET_NAME'], cloud_file_path)

    # check if file is already exists
    if os.path.exists(local_file_path) and not overwrite:
        return local_file_path

    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path


def _upload(cloud_file_path: str, file: Optional[Any] = None, local_file_path: Optional[str] = None) -> None:
    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    blob = bucket.blob(cloud_file_path)
    if file is not None:
        blob.upload_from_file(file)
    elif local_file_path is not None:
        blob.upload_from_filename(local_file_path)
    else:
        raise ValueError("Either file or local_file_path must be provided")


def download(cloud_file_path: str, local_file_path: Optional[str] = None, overwrite: bool = False) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "tensorleap", "data", CONFIG['BUCKET_NAME'], cloud_file_path)

    # check if file is already exists
    if os.path.exists(local_file_path) and not overwrite:
        return local_file_path

    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    root, extension = os.path.splitext(cloud_file_path)
    cloud_file_path = root.split(CONFIG['frame_separator'])[0] + '.mp4'
    blob = bucket.blob(cloud_file_path)
    root, extension = os.path.splitext(local_file_path)
    local_file_path = root.split(CONFIG['frame_separator'])[0] + '.mp4'
    blob.download_to_filename(local_file_path)
    return local_file_path


def check_gcs_files_existence(paths):
    """Check if each path in the list exists in the specified GCS bucket.

    Args:
        paths (list): A list of paths to check in the GCS bucket.

    Returns:
        pd.DataFrame: A DataFrame with columns 'path' and 'exists' indicating the existence of each path.
    """

    def check_existence(path):
        """Helper function to check existence of a single file."""
        blob = bucket.blob(path)
        return path, blob.exists()

    # Use ThreadPoolExecutor to parallelize the existence checks
    results = []
    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(check_existence, path) for path in paths]
        for future in futures:
            results.append(future.result())
    return results

