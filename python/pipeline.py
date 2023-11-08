import numpy as np
from pipeline_utils import get_visible_raw_image, get_metadata, normalize, white_balance, demosaic, \
    apply_color_space_transform, transform_xyz_to_srgb, apply_gamma, apply_tone_map, fix_orientation, \
    lens_shading_correction
import cv2
from model import RDUNet
from inference import predict
import yaml
import torch.nn as nn
from PIL import Image
import glob
import os
import torch
from PIL import Image
import skimage.io


def run_pipeline_v2(image_or_path, params=None, metadata=None, fix_orient=True):
    params_ = params.copy()
    if type(image_or_path) == str:
        image_path = image_or_path
        # raw image data
        raw_image = get_visible_raw_image(image_path)
        # metadata
        metadata = get_metadata(image_path)
    else:
        raw_image = image_or_path.copy()
        # must provide metadata
        if metadata is None:
            raise ValueError("Must provide metadata when providing image data in first argument.")

    current_image = raw_image
    #reshape input image
    current_image = current_image[0:2500, 0:2500,...]
    
    # linearization
    linearization_table = metadata['linearization_table']
    if linearization_table is not None:
        print('Linearization table found. Not handled.')

    #normalizing
    current_image = normalize(current_image, metadata['black_level'], metadata['white_level'])
    
    #lens shading correction
    gain_map_opcode = None
    if 'opcode_lists' in metadata:
        if 51009 in metadata['opcode_lists']:
            opcode_list_2 = metadata['opcode_lists'][51009]
            gain_map_opcode = opcode_list_2[9]
    if gain_map_opcode is not None:
        current_image = lens_shading_correction(current_image, gain_map_opcode=gain_map_opcode, bayer_pattern=metadata['cfa_pattern'])

    #white balancing
    current_image = white_balance(current_image, metadata['as_shot_neutral'], metadata['cfa_pattern'])

    #demosaicing
    current_image = demosaic(current_image, metadata['cfa_pattern'], output_channel_order='RGB', alg_type=params_['demosaic_type'])

    #color space transform
    current_image = apply_color_space_transform(current_image, metadata['color_matrix_1'], metadata['color_matrix_2'])

    #transform xyz to srgb
    current_image = transform_xyz_to_srgb(current_image)
    
    if fix_orient:
        # fix image orientation, if needed (after srgb stage, ok?)
        current_image = fix_orientation(current_image, metadata['orientation'])
    
    ###rdu denoising 시작###
    #model & test configuration load
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    model_params = config['model']
    test_params = config['test']
    n_channels = model_params['channels']
    
    #model parameters check
    print(model_params)
    
    #model generate
    model_path = os.path.join(test_params['pretrained models path'], 'small_model.pth')
    model = RDUNet(**model_params)
    #load model in multiple GPU
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print('Using multiple GPUs')
    model = model.to(device)
    #load model in single GPU
    #device = torch.device(test_params['device'])
    #print("Using device: {}".format(device))

    #weight load
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    #model = model.to(device) #single GPU 사용 case
    model.eval()
    ###end of model generating###
    
    #save path setting
    if test_params['save images']:
        save_path = os.path.join(test_params['results path'])
    else:
        save_path = None
        
    #forward pass
    #y_hat : original image
    #y_hat_ens : ensamble result
    y_hat, y_hat_ens = predict(model, current_image, device, test_params['padding'],  n_channels, save_path)
    ###end of rdu denoising###
    
    #gamma correction with rdunet
    y_hat_ens = apply_gamma(y_hat_ens)
    #gamma correction without rdunet
    original_image = apply_gamma(current_image)   


    #tone mapping
    #y_hat_ens = apply_tone_map(y_hat_ens)
    
    print('Image Pipeline with RDUNet is done.')
    
    return y_hat_ens, original_image
