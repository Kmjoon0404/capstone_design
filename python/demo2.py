import glob
import os
import cv2
import numpy as np

from pipeline import run_pipeline_v2
from pipeline_utils import get_visible_raw_image, get_metadata

params = {
    'save_as': 'png',
    'demosaic_type': 'EA',
    'save_dtype': np.uint8
}

# processing a directory
images_dir = '../data/'
image_paths = glob.glob(os.path.join(images_dir, '*.dng'))
#image_paths = '../data/test.DNG'
for image_path in image_paths:
    # raw image data
    raw_image = get_visible_raw_image(image_path)

    # metadata
    metadata = get_metadata(image_path)

    # modify WB here
    metadata['as_shot_neutral'] = [1., 1., 1.]

    # render
    output_image, original_image = run_pipeline_v2(image_path, params)
    print('final output shape : {}'.format(output_image.shape))

    # save
    output_image_path1 = image_path.replace('.dng', '_{}.'.format('final_output_with_rdunet_32_full') + params['save_as'])
    output_image_path2 = image_path.replace('.dng', '_{}.'.format('original_image_without_rdunet') + params['save_as'])
    max_val = 255
    output_image = (output_image[..., ::-1] * max_val).astype(params['save_dtype'])
    original_image = (original_image[..., ::-1] * max_val).astype(params['save_dtype'])
    cv2.imwrite(output_image_path1, output_image)
    cv2.imwrite(output_image_path2, original_image)
