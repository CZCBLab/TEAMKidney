import os
from mmseg.apis import init_model, inference_model
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
# Backgroud = [255,255,255]
# GBM = [255,0,0]
# FP = [0,0,255]

def save_colored_mask(mask, save_path):
    bin_colormap = np.array([[255,255,255],[255,0,0],[0,0,255],[0,255,0]]).astype(np.uint8)
    mask = mask.astype(np.uint8)
    mask_p = Image.fromarray(mask, mode='P')
    mask_p.putpalette(bin_colormap)
    mask_p.save(save_path)
def save_maskon_image(image, mask, save_path):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == 1] = [255, 0, 0]  # 红色
    colored_mask[mask == 2] = [0, 0, 255]  # 蓝色
    image_rgb = np.stack([image] * 3, axis=-1)
    alpha = 0.5
    overlayed_image = (image_rgb * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    Image.fromarray(overlayed_image).save(save_path)

def main():
    inputImages = './test_data/images' ## input
    predoutFolder = './test_data/pred' ## output
    maskedoutFolder = './test_data/masksonimage'
    isCPUonly = True

    extensions=('.jpg', '.jpeg', '.png', '.tif')

    if isCPUonly:
        device = torch.device('cpu')
    # model = init_model(
    #     '../result/mmseg_result/deeplabv3/20230204-scc/deeplabv3_r50-d8_4xb4_kidney_512x512.py',
    #     '../result/mmseg_result/deeplabv3/20230204-scc/iter_3000.pth')

        model = init_model(
            './models/convnext-base_upernet_8xb2-amp-160k_tem-512x512_cpu.py',
            './models/iter_64000.pth', device=device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("CUDA is not available. Using CPU.")
        # model = init_model(
        #     '../result/mmseg_result/deeplabv3/20230204-scc/deeplabv3_r50-d8_4xb4_kidney_512x512.py',
        #     '../result/mmseg_result/deeplabv3/20230204-scc/iter_3000.pth')
        model = init_model(
            './models/convnext-base_upernet_8xb2-amp-160k_tem-512x512.py',
            './models/iter_64000.pth')



    # device = torch.device('cpu')
    print(device)
    model.to(device)

    if os.path.isdir(inputImages):
        imFolder = inputImages
        files = os.listdir(inputImages)
    if os.path.isfile(inputImages):
        (imFolder, files) = os.path.split(inputImages)
        files = [files]
    if not os.path.exists(predoutFolder):
        os.makedirs(predoutFolder)
    if not os.path.exists(maskedoutFolder):
        os.makedirs(maskedoutFolder)

    for file in tqdm(files, desc = 'files'):

        if file.endswith(extensions):
            imfile = os.path.join(imFolder, file)
            try:
                image = Image.open(imfile).convert('L')
                result = inference_model(model, imfile)
                pred = result.pred_sem_seg.data.cpu().numpy().squeeze()
                save_colored_mask(pred, os.path.join(predoutFolder, file.split('.')[0]+'.png'))
                save_maskon_image(image, pred, os.path.join(maskedoutFolder, file.split('.')[0]+'.png'))
            except Exception as error:
                print(f'{imfile} error:', error)


if __name__ == '__main__':
    main()