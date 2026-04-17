import argparse
import json
import os
import pickle
import random

import cv2
import labelme.utils
import numpy as np
from PIL import Image
from skimage import measure
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector


GBM_closing_kernel_size = 5
FP_openning_kernel_size = 4
GBM_point_interval = 5
FP_point_interval = 3


def mask2shape(results):
    group_id = 0
    shapes = []

    instances = results.pred_instances
    instance_labels = instances.labels.to('cpu').numpy()
    instance_masks = instances.masks
    instance_masks = [instance_masks[i].to('cpu').numpy() for i in range(instance_masks.size(0))]
    seg_mask = results.pred_panoptic_seg.sem_seg.to('cpu').numpy().squeeze()
    im_shape = seg_mask.shape

    FP_mask = np.where(seg_mask > 2, seg_mask, 0)
    unique_values = np.unique(FP_mask[FP_mask > 0])
    new_FP_mask = np.zeros_like(FP_mask)
    for idx, value in enumerate(unique_values, start=1):
        new_FP_mask[FP_mask == value] = idx

    GBM_mask = np.where(seg_mask == 1, 1, 0)
    GBM_instance = measure.label(GBM_mask, connectivity=2)
    FP_instance = measure.label(new_FP_mask, connectivity=2)
    GBM_n = np.max(GBM_instance)
    FP_n = np.max(FP_instance)

    GBM_ins = np.zeros(im_shape)
    FP_ins = np.zeros(im_shape)

    GBM_contours = []
    FP_contours = []

    for i in range(GBM_n):
        GBM_ins[np.where(GBM_instance == i + 1)] = True
        GBM_ins[np.where(GBM_instance != i + 1)] = False
        GBM_ins = GBM_ins.astype(np.uint8)

        if np.sum(GBM_ins) < 1000:
            continue

        GBM_ins_countours, h = cv2.findContours(
            GBM_ins, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for j in range(len(GBM_ins_countours)):
            GBM_ins_countour = np.squeeze(GBM_ins_countours[j])
            GBM_ins_countour = GBM_ins_countour[::GBM_point_interval, :]
            GBM_contours.append(GBM_ins_countour)

            shape = {
                "label": "GBM",
                "points": GBM_ins_countour.tolist(),
                "group_id": group_id if len(GBM_ins_countours) > 1 else None,
                "description": "",
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)

        if len(GBM_ins_countours) > 1:
            group_id += 1

    for i in range(FP_n):
        FP_ins[np.where(FP_instance == i + 1)] = True
        FP_ins[np.where(FP_instance != i + 1)] = False
        FP_ins = FP_ins.astype(np.uint8)

        if np.sum(FP_ins) < 100:
            continue

        FP_ins_countour, h = cv2.findContours(
            FP_ins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        FP_ins_countour = np.squeeze(FP_ins_countour[0])
        FP_ins_countour = FP_ins_countour[::FP_point_interval, :]

        FP_contours.append(FP_ins_countour)

        shape = {
            "label": "FP",
            "points": FP_ins_countour.tolist(),
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {}
        }
        shapes.append(shape)

    return shapes


def mask2labelme(results, imfile):
    img = cv2.imread(imfile)

    data_labelme = {}
    data_labelme["version"] = "5.3.0"
    data_labelme["flags"] = {}

    shape_my = mask2shape(results)
    data_labelme["shapes"] = shape_my

    data_labelme["imagePath"] = imfile

    im_64 = labelme.utils.img_arr_to_b64(img)
    im_64 = im_64.decode("utf-8")
    data_labelme["imageData"] = im_64

    data_labelme["imageHeight"] = img.shape[0]
    data_labelme["imageWidth"] = img.shape[1]

    return data_labelme


def show_panoptic(result, outfile):
    seg_mask = result.pred_panoptic_seg.sem_seg.to('cpu').numpy().squeeze()
    im_shape = seg_mask.shape

    GBM_mask = np.where(seg_mask == 1, 1, 0)
    FP_mask = np.where(seg_mask > 2, seg_mask, 0)

    unique_labels = np.unique(FP_mask)
    label_to_color = {
        label: [random.randint(50, 255), 0, 0]
        for label in unique_labels if label > 0
    }

    FP_mask_rgb = np.zeros((*im_shape, 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        FP_mask_rgb[FP_mask == label] = color

    result_image = np.zeros((*im_shape, 3), dtype=np.uint8)
    result_image[:, :, 2] = GBM_mask * 255
    result_image += FP_mask_rgb

    result_image_pil = Image.fromarray(result_image)
    result_image_pil.save(outfile, format='JPEG', optimize=True, quality=85)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run panoptic inference and export JSON / PKL / visualization outputs."
    )

    parser.add_argument(
        "--config-file",
        type=str,
        default="./myconfigs/mask2former_hr/mask2former_r50_8xb2-lsj-50e_kidney-panoptic_1024_hm_st.py",
        help="Path to model config file."
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="./models/checkpoint.pth",
        help="Path to model checkpoint."
    )
    parser.add_argument(
        "--imgpath",
        type=str,
        default="./data/images",
        help="Root directory of input images."
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="./results/infer_panoptic_json",
        help="Output directory for labelme JSON files."
    )
    parser.add_argument(
        "--pkl-path",
        type=str,
        default="./results/infer_panoptic_pkl",
        help="Output directory for pickle result files."
    )
    parser.add_argument(
        "--outim-path",
        type=str,
        default="./results/infer_panoptic_im",
        help="Output directory for visualization images."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference, e.g. cuda or cpu."
    )

    parser.add_argument(
        "--savepkl",
        action="store_true",
        help="Save raw inference results as PKL."
    )
    parser.add_argument(
        "--savejson",
        action="store_true",
        help="Save labelme JSON outputs."
    )
    parser.add_argument(
        "--ifshow",
        action="store_true",
        help="Save panoptic visualization images."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.json_path, exist_ok=True)
    os.makedirs(args.pkl_path, exist_ok=True)
    os.makedirs(args.outim_path, exist_ok=True)

    model = init_detector(args.config_file, args.checkpoint_file, device=args.device)
    extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.TIF')

    for root, dirs, files in tqdm(os.walk(args.imgpath)):
        relative_path = os.path.relpath(root, args.imgpath)

        for file in tqdm(files):
            if not file.endswith(extensions):
                continue

            stem = os.path.splitext(file)[0]

            if relative_path in ('', '.', './'):
                jsonfile = os.path.join(args.json_path, stem + '.json')
                outimfile = os.path.join(args.outim_path, stem + '.jpg')
                pklfile = os.path.join(args.pkl_path, stem + '.pkl')
            else:
                os.makedirs(os.path.join(args.json_path, relative_path), exist_ok=True)
                os.makedirs(os.path.join(args.pkl_path, relative_path), exist_ok=True)
                os.makedirs(os.path.join(args.outim_path, relative_path), exist_ok=True)

                jsonfile = os.path.join(args.json_path, relative_path, stem + '.json')
                outimfile = os.path.join(args.outim_path, relative_path, stem + '.jpg')
                pklfile = os.path.join(args.pkl_path, relative_path, stem + '.pkl')

            # Keep the original skip logic unchanged
            if os.path.exists(jsonfile) and os.path.exists(outimfile):
                continue

            imfile = os.path.join(root, file)
            result = inference_detector(model, imfile)

            if args.savepkl:
                with open(pklfile, 'wb') as f:
                    pickle.dump(result, f)

            if args.savejson:
                data_labelme = mask2labelme(result, imfile)
                with open(jsonfile, 'w') as f:
                    json.dump(data_labelme, f, indent=4)

            if args.ifshow:
                show_panoptic(result, outimfile)


if __name__ == "__main__":
    main()