import time

import pandas as pd
import os
import cv2
from code_loader.utils import rescale_min_max
from concurrent.futures import ThreadPoolExecutor

from casablanca.config import CONFIG
from casablanca.data.preprocess import load_data, load_data_all
from casablanca.utils.gcs_utils import _download, _connect_to_gcs_and_return_bucket, check_gcs_files_existence
from casablanca.utils.general_utils import input_video


def download_video(path, frame_index):
    local_video_path = _download(path)
    return local_video_path, frame_index


# Function to process video frames and upload them
def process_frame(frame_index, local_video_path, bucket):
    frame = input_video(local_video_path, frame_index)
    frame = rescale_min_max(frame.numpy()).transpose((1, 2, 0))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_filename = f"{'/'.join(local_video_path.split('/')[-6:])[:-4]}{CONFIG['frame_separator']}{frame_index}.png"
    local_image_path = os.path.join("/tmp", os.path.basename(image_filename))
    cv2.imwrite(local_image_path, frame)

    # Check if image already exists on GCS
    blob = bucket.blob(image_filename)
    blob.upload_from_filename(local_image_path)
    # if not blob.exists():  # Only upload if the blob does not exist
    #     blob.upload_from_filename(local_image_path)

    # Clean up local files
    if os.path.exists(local_image_path):
        os.remove(local_image_path)


def save_frames_to_gcs(data) -> pd.DataFrame:
    # data = load_data_all()
    # data = load_data()
    frame_paths = set(data['frame_path'])
    frame_zero_paths = set(data['frame_path'].apply(lambda x: x.split(CONFIG['frame_separator'])[0] + CONFIG['frame_separator'] + '0.png'))
    all_frames_paths = list(frame_paths.union(frame_zero_paths))
    t0 = time.time()
    results = check_gcs_files_existence(list(all_frames_paths))
    t1 = time.time()
    print(f"Checked existence of {len(frame_paths)} frames in {t1 - t0} seconds")
    res_df = pd.DataFrame(results, columns=['path', 'exists'])
    missing_frames = res_df[~res_df['exists']]
    vid_paths = [(x.split(CONFIG['frame_separator'])[0] + '.mp4',
                  int(x.split(CONFIG['frame_separator'])[-1].split('.')[0]))
                 for x in missing_frames['path']]
    # Initialize connection to GCS
    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])
    t0 = time.time()
    downloaded_videos = []
    with ThreadPoolExecutor() as executor:
        # Parallel download of all videos
        futures = [executor.submit(download_video, path, frame_index) for path, frame_index in vid_paths]
        for future in futures:
            result = future.result()
            if result:
                downloaded_videos.append(result)
    t1 = time.time()
    print(f"Downloaded {len(downloaded_videos)} videos in {t1 - t0} seconds. Processing frames...")
    # time.sleep(5)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, frame_index, local_video_path, bucket) for local_video_path, frame_index in
                   downloaded_videos]
    tf = time.time()
    print(f"Processed {len(downloaded_videos)} videos in {tf - t1} seconds. Total time: {tf - t0}")
    print(f"Time taken: {tf - t0}")
    print('Done')
    return data
