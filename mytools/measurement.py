import argparse
import json
import pandas as pd
from skimage import measure, draw
from tqdm import tqdm
import cv2
import numpy as np
import os
from collections import defaultdict

import mmdet.utils.m4m_functions as m4m


def process_image_file(imfile, jsonfilename, outSegfile, pixelsize, SDheight):
    image_id = os.path.basename(imfile).split('.')[0]
    image = cv2.imread(imfile, cv2.IMREAD_GRAYSCALE)

    height, width = image.shape
    GBM_ins = np.zeros((height, width), dtype=np.uint16)
    FP_ins = np.zeros((height, width), dtype=np.uint16)

    with open(jsonfilename, 'r') as f:
        segimage = json.load(f)

    fp_instance_id = 1
    gbm_instance_id = 1
    gbm_groups = defaultdict(lambda: np.zeros((height, width), dtype=bool))

    for shape in segimage['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)

        rr, cc = draw.polygon(points[:, 1], points[:, 0], (height, width))

        if 'GBM' in label:
            group_id = shape.get('group_id', None)
            if group_id is not None:
                gbm_groups[group_id][rr, cc] = np.logical_xor(gbm_groups[group_id][rr, cc], True)
            else:
                GBM_ins[rr, cc] = gbm_instance_id
                gbm_instance_id += 1

        elif 'FP' in label:
            FP_ins[rr, cc] = fp_instance_id
            fp_instance_id += 1

    for group_mask in gbm_groups.values():
        GBM_ins[group_mask] = gbm_instance_id
        gbm_instance_id += 1

    if GBM_ins.sum() < 1:
        return {"image_id": image_id, 'skeleton_length': "None", 'area': "None",
                'GBM_mean_width': "None", 'FP_num': "None", 'FP_mean_width': "None"}

    GBM_im_new, skeleton_img, GBM_info, FP_width_img, FP_info, FP_widths, FP_im_new = \
        m4m.calculate_measurements_fp_split(GBM_ins, FP_ins, pixelsize, SDheight)

    if FP_widths == 0:
        return {"image_id": image_id, 'skeleton_length': "None", 'area': "None",
                'GBM_mean_width': "None", 'FP_num': "None", 'FP_mean_width': "None"}

    m4m.save_insmaskon_image(image, GBM_im_new, FP_ins, skeleton_img, FP_width_img, outSegfile)

    return {**GBM_info, **FP_info, "image_id": image_id, "FP_widths": FP_widths}


def parse_args():
    parser = argparse.ArgumentParser(description="Post-processing measurement pipeline")

    parser.add_argument("--image-path", type=str, 
                        default="./data/images",
                        required=True,
                        help="Input image root directory")
    parser.add_argument("--seg-path", type=str, 
                        default="./results/infer_panoptic_json",
                        required=True,
                        help="Path to segmentation JSON results")
    parser.add_argument("--out-csv", type=str, 
                        default="./results/measuremnts.csv",
                        required=True,
                        help="Output CSV file path")
    parser.add_argument("--out-seg-path", type=str, 
                        default="./results/measurements_im",
                        required=True,
                        help="Output directory for visualization images")
    parser.add_argument("--out-fp-json", type=str, 
                        default="./results/FPs.json",
                        required=True,
                        help="Output JSON file for FP widths")
    parser.add_argument("--pixelsize", type=float, default=1,
                        help="Pixel size (default=1)")
    parser.add_argument("--sdheight", type=int, default=20,
                        help="SD height (default=20)")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_seg_path, exist_ok=True)

    width_info_list = []
    FP_widths_json = []
    results = []

    extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.TIF')

    for root, dirs, files in tqdm(os.walk(args.image_path)):
        relative_path = os.path.relpath(root, args.image_path)

        os.makedirs(os.path.join(args.out_seg_path, relative_path), exist_ok=True)

        for file in tqdm(files):
            if not file.endswith(extensions):
                continue

            imfile = os.path.join(root, file)
            jsonfilename = os.path.join(
                args.seg_path,
                relative_path,
                os.path.splitext(file)[0] + '.json'
            )
            outSegfile = os.path.join(
                args.out_seg_path,
                relative_path,
                os.path.splitext(file)[0] + '.png'
            )

            result = process_image_file(
                imfile, jsonfilename, outSegfile,
                args.pixelsize, args.sdheight
            )

            results.append(result)

    for result in results:
        if result:
            width_info_list.append(result)
            if "FP_widths" in result:
                FP_widths_json.append({
                    "image_id": result["image_id"],
                    "FP_widths": result["FP_widths"]
                })

    with open(args.out_fp_json, 'w') as file:
        json.dump(FP_widths_json, file, indent=4)

    df = pd.DataFrame(width_info_list)
    df.to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    main()