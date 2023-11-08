import random
import torch
import numpy as np
from skimage import io, color, img_as_ubyte
import os


def load_image(image_path, channels):
    """
    Load image and change it color space from RGB to Grayscale if necessary.
    :param image_path: str
        Path of the image.
    :param channels: int
        Number of channels (3 for RGB, 1 for Grayscale)
    :return: numpy array
        Image loaded.
    """
    image = io.imread(image_path)

    if image.ndim == 3 and channels == 1:       # Convert from RGB to Grayscale and expand dims.
        image = img_as_ubyte(color.rgb2gray(image))
        return np.expand_dims(image, axis=-1)
    elif image.ndim == 2 and channels == 1:     # Handling grayscale images if needed.
        if image.dtype != 'uint8':
            image = img_as_ubyte(image)
        return np.expand_dims(image, axis=-1)

    return image


def mod_crop(image, mod):
    """
    Crops image according to mod to restore spatial dimensions
    adequately in the decoding sections of the model.
    :param image: numpy array
        Image to crop.
    :param mod: int
        Module for padding allowed by the number of
        encoding/decoding sections in the model.
    :return: numpy array
        Copped image
    """
    size = image.shape[:2] #height와 width 받기
    size = size - np.mod(size, mod) #구한 size(height, width)를 mode로 나눈 나머지를 size에서 빼준다 -> size가 mod의 배수가 된다 
    image = image[:size[0], :size[1], ...] #channel의 수는 유지, size와 width 변경

    return image #crop한 image return


def mod_pad(image, mod):
    """
    Pads image according to mod to restore spatial dimensions
    adequately in the decoding sections of the model.
    :param image: numpy array
        Image to pad.
    :param mod: int
        Module for padding allowed by the number of
        encoding/decoding sections in the model.
    :return: numpy  array, tuple
        Padded image, original image size.
    """
    size = image.shape[:2] #height와 width 받기
    print(size)
    h, w = np.mod(size, mod) # h : height를 mod로 나눈 나머지/ w : width를 mod로 나눈 나머지 
    h, w = mod - h, mod - w # 각각 padding에 필요한 양 계산
    if h != mod or w != mod: #padding이 필요한 경우
        if image.ndim == 3: #3 channel인 경우
            image = np.pad(image, ((0, h), (0, w), (0, 0)), mode='reflect') #reflect mode로 padding
        else:
            image = np.pad(image, ((0, h), (0, w)), mode='reflect')

    return image, size #padding한 image와 원래 이미지의 size를 tuple로 return


def set_seed(seed=1): #random seed 설정
    """
    Sets all random seeds.
    :param seed: int
        Seed value.
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_ensemble(image, normalize=True):
    """
    Create image ensemble to estimate denoised image.
    :param image: numpy array
        Noisy image.
    :param normalize: bool
        Normalize image to range [0., 1.].
    :return: list
        Ensemble of noisy image transformed.
    """
    img_rot = np.rot90(image) #이미지 90도 회전
    ensemble_list = [
        image, np.fliplr(image), np.flipud(image), np.flipud(np.fliplr(image)),
        img_rot, np.fliplr(img_rot), np.flipud(img_rot), np.flipud(np.fliplr(img_rot))
    ]#원본 이미지와 여러가지 변환(뒤집기,회전)한 이미지의 리스트 생성

    ensemble_transformed = []#앙상블을 저장할 리스트 생성
    for img in ensemble_list:
        if img.ndim == 2:    #흑백인 경우                                       # Expand dims for channel dimension in gray scale.
            img = np.expand_dims(img.copy(), 0)                     # Use copy to avoid problems with reverse indexing.
        else: #3channel인 경우
            img = np.transpose(img.copy(), (2, 0, 1))               # Channels-first transposition.
        if normalize:
            img = img / 255.

        img_t = torch.from_numpy(np.expand_dims(img, 0)).float()    # 배치 차원 추가
        ensemble_transformed.append(img_t) #앙상블 리스트에 추가

    return ensemble_transformed #만들어진 앙상블 리스트 return


def ensemble_forward_pass(model, ensemble, device, return_single=False):
    """
    Apply inverse transforms to predicted image ensemble and average them.
    :param ensemble: list
        Predicted images, ensemble[0] is the original image,
        and ensemble[i] is a transformed version of ensemble[i].
    :param return_single: bool
        Return also ensemble[0] to evaluate single prediction
    :return: numpy array or tuple of numpy arrays
        Average of the predicted images, original image denoised.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4"
    ensemble_np = [] #앙상블 역 변환한 이미지를 저장할 리스트
    print('ensemble_forward_pass device : {}'.format(device))

    for x in ensemble:
        #data GPU로 이동시키기
        x = x.to(device)

        with torch.no_grad():
            y_hat = model(x) #forward pass
            y_hat = y_hat.cpu().detach().numpy().astype('float32') #cpu로 데이터 가져오기
            
        y_hat = y_hat.squeeze()#추가한 차원 제거
        if y_hat.ndim == 3:                       # Transpose if necessary.
            y_hat = np.transpose(y_hat, (1, 2, 0)) #channel last 방식으로 변경
            
        ensemble_np.append(y_hat)

    # Apply inverse transforms to vertical and horizontal flips.
    img = ensemble_np[0] + np.fliplr(ensemble_np[1]) + np.flipud(ensemble_np[2]) + np.fliplr(np.flipud(ensemble_np[3]))

    # Apply inverse transforms to 90º rotation, vertical and horizontal flips
    img = img + np.rot90(ensemble_np[4], k=3) + np.rot90(np.fliplr(ensemble_np[5]), k=3)
    img = img + np.rot90(np.flipud(ensemble_np[6]), k=3) + np.rot90(np.fliplr(np.flipud(ensemble_np[7])), k=3)

    # Average and clip final predicted image.
    img = img / 8.
    img = np.clip(img, 0., 1.)

    if return_single:
        return img, ensemble_np[0] #예측한 이미지와 원본 이미지 return
    else:
        return img #예측한 이미지만 return

