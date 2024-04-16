import os
from mmseg.apis import init_model, inference_model
import torch
from PIL import Image
import numpy as np
#from tqdm import tqdm
# Backgroud = [255,255,255]
# GBM = [255,0,0]
# FP = [0,0,255]


def main(inputimage):
    #inputImages = './test_data/images' ## input
    #predoutFolder = './test_data/pred' ## output
    #maskedoutFolder = './test_data/masksonimage'
    isCPUonly = True

    #extensions=('.jpg', '.jpeg', '.png', '.tif')

    if isCPUonly:
        device = torch.device('cpu')
        model = init_model(
            './models/convnext-base_upernet_8xb2-amp-160k_tem-512x512_cpu.py',
            './models/iter_64000.pth', device=device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("CUDA is not available. Using CPU.")
        model = init_model(
            './models/convnext-base_upernet_8xb2-amp-160k_tem-512x512.py',
            './models/iter_64000.pth')

    model.to(device)

    image = Image.open(inputimage).convert('L')
    result = inference_model(model, np.asarray(image))
    pred = result.pred_sem_seg.data.cpu().numpy().squeeze()

    # Colored mask:
    bin_colormap = np.array([[255,255,255],[255,0,0],[0,0,255],[0,255,0]]).astype(np.uint8)
    mask = pred.astype(np.uint8)
    mask_p = Image.fromarray(mask, mode='P')
    mask_p.putpalette(bin_colormap)
    mask_rgb = mask_p.convert("RGB") # ZLC convert back to RGB but don't save

    # Colored mask on image:
    colored_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    colored_mask[pred == 1] = [255, 0, 0]  # 红色
    colored_mask[pred == 2] = [0, 0, 255]  # 蓝色
    image_rgb = np.stack([image] * 3, axis=-1)
    alpha = 0.5
    overlayed_image = (image_rgb * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    overlayed_image = Image.fromarray(overlayed_image)

    return mask_rgb, overlayed_image
