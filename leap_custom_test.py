import os
import time
from os import environ
import onnxruntime
import numpy as np
import tensorflow as tf

from casablanca.config import CONFIG
from casablanca.data.preprocess import load_data_all
from casablanca.data.save_frames_to_gcs import process_frame, save_frames_to_gcs
from casablanca.utils.gcs_utils import _connect_to_gcs_and_return_bucket
from casablanca.utils.loss import lpip_loss_alex, lpip_loss_vgg, dummy_loss
from casablanca.utils.metrics import lpip_alex_metric, lpip_vgg_metric, l1
from casablanca.utils.visuelizers import image_change_last, grid_frames, grid_all
from leap_binder import input_encoder_first_frame, preprocess_func, input_encoder_current_frame, \
    input_encoder_first_frame, calc_metadata_vals, calc_metadata_stats_func


def check_custom_test():
    if environ.get('AUTH_SECRET') is None:
        print("The AUTH_SECRET system variable must be initialized with the relevant secret to run this test")
        exit(-1)
    print("started custom tests")

    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/lia_encoder_decoder.onnx'
    # sess = onnxruntime.InferenceSession(os.path.join(dir_path, model_path))
    #
    # input_name_1 = sess.get_inputs()[0].name
    # input_name_2 = sess.get_inputs()[1].name
    # input_name_3 = sess.get_inputs()[2].name
    # output_names = [output.name for output in sess.get_outputs()]

    # t = process_frame(0,
    #                   '/Users/chenrothschild/tensorleap/data/casablanca-datasets/data/dev/mp4/id00012/O2jVZ--4vCk/00067.mp4',
    #                   _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME']))

    subsets = preprocess_func()
    for set in subsets:
        for idx in range(set.length):
            source_image = first_frame = input_encoder_first_frame(idx, set)
            current_frame = input_encoder_current_frame(idx, set)

            # pred = sess.run(output_names, {input_name_1: np.expand_dims(source_image, 0),
            #                                input_name_2: np.expand_dims(first_frame, 0),
            #                                input_name_3: np.expand_dims(current_frame, 0)})[0]
            #

            # metrics
            batch_input = tf.convert_to_tensor(np.expand_dims(current_frame, 0))
            t0 = time.time()
            l1_res = l1(batch_input, batch_input)
            t1 = time.time()
            print(f'l1 time: {round(t1 - t0, 3)}')
            t0 = time.time()
            loss_alex = lpip_alex_metric(np.expand_dims(current_frame, 0), np.expand_dims(current_frame, 0))
            t1 = time.time()
            print(f'lpip_alex_metric time: {round(t1 - t0, 3)}')
            t0 = time.time()
            loss_vgg = lpip_vgg_metric(np.expand_dims(current_frame, 0), np.expand_dims(current_frame, 0))
            t1 = time.time()
            print(f'lpip_vgg_metric time: {round(t1 - t0, 3)}')
            dummy_loss_ = dummy_loss(np.expand_dims(current_frame, 0), np.expand_dims(current_frame, 0))

            # vis
            pred_image = current_frame
            grid_all_ = grid_all(first_frame, current_frame, source_image, pred_image)
            grid_frames_ = grid_frames(first_frame, current_frame)
            source_image_vis = image_change_last(source_image)
            current_frame_vis = image_change_last(current_frame)
            first_frame_vis = image_change_last(first_frame)

            source_res = calc_metadata_stats_func('source_image')(idx, set)
            current_res = calc_metadata_stats_func('current_frame')(idx, set)
            metadata_vals = calc_metadata_vals(idx, set)

    print("successfully!")


if __name__ == "__main__":
    data = load_data_all()
    dev = data[data['subset'] == 'dev']
    train_df = save_frames_to_gcs(dev)

    # check_custom_test()
