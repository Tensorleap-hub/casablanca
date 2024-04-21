import os
from os import environ
import onnxruntime
import numpy as np

from casablanca.utils.loss import lpip_loss_alex, lpip_loss_vgg, dummy_loss
from casablanca.utils.metrics import lpip_alex_metric, lpip_vgg_metric
from casablanca.utils.visuelizers import Image_change_last, grid_frames
from leap_binder import input_encoder_source_image, preprocess_func, input_encoder_current_frame, \
    input_encoder_first_frame, source_image_color_brightness_mean, \
    source_image_color_brightness_std, source_image_hsv, source_image_lab, get_idx, get_id_of_source_image, \
    get_video_path_of_source_image


def check_custom_test():
    if environ.get('AUTH_SECRET') is None:
        print("The AUTH_SECRET system variable must be initialized with the relevant secret to run this test")
        exit(-1)
    print("started custom tests")

    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = 'model/lia_encoder_decoder.onnx'
    sess = onnxruntime.InferenceSession(os.path.join(dir_path, model_path))

    input_name_1 = sess.get_inputs()[0].name
    input_name_2 = sess.get_inputs()[1].name
    input_name_3 = sess.get_inputs()[2].name
    output_names = [output.name for output in sess.get_outputs()]

    subsets = preprocess_func()
    for set in subsets:
        print(f'set: {set}')
        for idx in range(set.length):
            print(f'start idx: {idx}')
            first_frame = input_encoder_first_frame(idx, set)
            current_frame = input_encoder_current_frame(idx, set)
            source_image = input_encoder_source_image(idx, set)

            # pred = sess.run(output_names, {input_name_1: np.expand_dims(source_image, 0),
            #                                input_name_2: np.expand_dims(first_frame, 0),
            #                                input_name_3: np.expand_dims(current_frame, 0)})[0]
            #
            loss_alex = lpip_alex_metric(np.expand_dims(source_image, 0), np.expand_dims(source_image, 0))
            loss_vgg = lpip_vgg_metric(np.expand_dims(source_image, 0), np.expand_dims(source_image, 0))
            dummy_loss_ = dummy_loss(np.expand_dims(source_image, 0), np.expand_dims(source_image, 0))

            # grid_frames_ = grid_frames(first_frame, current_frame)
            # source_image_vis = Image_change_last(source_image)
            # current_frame_vis = Image_change_last(current_frame)
            # first_frame_vis = Image_change_last(first_frame)

            video_path_of_source_image = get_video_path_of_source_image(idx, set)
            id_of_source_image = get_id_of_source_image(idx, set)
            print(f'id_of_source_image: {id_of_source_image}')
            print(f'video_path_of_source_image: {video_path_of_source_image}')

            idx_ = get_idx(idx, set)
            source_image_color_brightness_mean_ = source_image_color_brightness_mean(idx, set)
            source_image_color_brightness_std_ = source_image_color_brightness_std(idx, set)
            source_image_hsv_ = source_image_hsv(idx, set)
            source_image_lab_ = source_image_lab(idx, set)

    print("successfully!")


if __name__ == "__main__":
    check_custom_test()
