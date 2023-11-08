import os
import yaml
import torch
import numpy as np
import scipy.io as sio
from os.path import join
from model import RDUNet
from skimage import io
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from utils import build_ensemble, ensemble_forward_pass, mod_pad, mod_crop


def predict(model, input_array, device, padding, n_channels, results_path):
    multi_channel = True if n_channels == 3 else False

    x, size = mod_pad(input_array, 8)
    x = build_ensemble(x, normalize=False)

    with torch.no_grad():
        y_hat_ens, y_hat = ensemble_forward_pass(model, x, device, return_single=True)

        if padding:
            y_hat = y_hat[:size[0], :size[1], ...]
            y_hat_ens = y_hat_ens[:size[0], :size[1], ...]


    if results_path is not None:
        y_hat = np.squeeze(y_hat)
        y_hat_ens = np.squeeze(y_hat_ens)

        os.makedirs(results_path, exist_ok=True)
        
    print("RDUNet : Prediction Finished")
    
    return y_hat, y_hat_ens
