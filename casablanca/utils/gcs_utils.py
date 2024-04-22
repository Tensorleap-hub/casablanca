from functools import lru_cache
from os import environ
import json
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud.storage import Bucket
from typing import Optional
import os

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


def _download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "tensorleap", "data", CONFIG['BUCKET_NAME'], cloud_file_path)

    # check if file is already exists
    if os.path.exists(local_file_path):
        return local_file_path

    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    dir_path = os.path.dirname(local_file_path)
    os.makedirs(dir_path, exist_ok=True)
    blob = bucket.blob(cloud_file_path)
    blob.download_to_filename(local_file_path)
    return local_file_path


def download(cloud_file_path: str, local_file_path: Optional[str] = None) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        home_dir = os.getenv("HOME")
        local_file_path = os.path.join(home_dir, "tensorleap", "data", CONFIG['BUCKET_NAME'], cloud_file_path)

    # check if file is already exists
    if os.path.exists(local_file_path):
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
