# functions for the mask4measurements file

import json
import os
import pandas as pd
from skimage.morphology import skeletonize, medial_axis
import networkx as nx
from PIL import Image, ImageDraw
from skimage import measure
import random

def calculate_measurements(GBM_ins, FP_ins, pixelsize, SDheight):
    im_shape = GBM_ins.shape[0:2]
    GBM_n = np.max(GBM_ins)
    if GBM_n == 0:
        print("There is no GBM in this image!")
        return 1

    # SDheight = round(min(int(MINGBMWIDTH / pixelsize / 2), min(im_shape) / 20)) ## 4 for dkd 2 for fabry
    # if SDheight <= 3:
    #     SDheight = 2

    # Declaring variables
    FP_seg_width = []
    FP_seg_dic = {}
    FP_seg_dic['FP_num'] = 0
    FP_seg_dic['FP_mean_width'] = 0
    FP_width_img = np.zeros(im_shape)
    FP_widths = []
    # new FP declaration 6/10
    FP_im_new = np.zeros(im_shape, dtype=np.uint8)

    skeleton_img = np.zeros(im_shape, dtype=np.uint8)
    GBM_im_new = np.zeros(im_shape, dtype=np.uint8)
    GBM_seg_dic = {}
    GBM_seg_dic['skeleton_length'] = 0
    GBM_seg_dic['area'] = 0

    # iterates over every GBM instance
    for i in range(GBM_n):

        GBM_in = np.zeros(im_shape)
        GBM_in[np.where(GBM_ins == i + 1)] = True
        GBM_in = GBM_in.astype(np.uint8)
        if np.sum(GBM_in) < 5000:  # (MINGBMWIDTH / pixelsize) ** 2 * 3
            # print(np.sum(GBM_in), (MINGBMWIDTH / pixelsize) ** 2)
            continue
        skeleton = skeletonize(GBM_in)
        if np.sum(skeleton) < 500:  # MINGBMWIDTH / pixelsize * 10
            continue

        kernel = np.ones((SDheight, SDheight), dtype=np.uint8)
        GBM_in_dilated = cv2.dilate(GBM_in, kernel, 1)  # dilate the GBM to calculate the FP
        GBM_in_dilated = GBM_in_dilated.astype(np.uint8)
        GBM_contour_dilated, h = cv2.findContours(GBM_in_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        GBM_contour_dilated_list = [np.squeeze(arr) for arr in GBM_contour_dilated]
        GBM_contour_dilated_array = np.concatenate(GBM_contour_dilated_list, axis=0)
        # GBM_contour_dilated = np.squeeze(GBM_contour_dilated[0]) # GBM contour after dilation
        GBM_contour_dilated_img = np.zeros(im_shape).astype(np.uint8)
        GBM_contour_dilated_img[GBM_contour_dilated_array[:, 1], GBM_contour_dilated_array[:,
                                                                 0]] = 1  # image of the GBM contour after dilation
        FP_width_pos = GBM_contour_dilated_img * FP_ins  # width image of each FP along the i th GBM
        #

        FP_seg_ids = np.unique(FP_width_pos)  # the ids of the FP along the i th GBM
        FP_seg_ids = FP_seg_ids.astype(np.int64)

        FP_ratio = []  # initialize list for sum of FP_widths

        FP_seg_ids = FP_seg_ids[FP_seg_ids != 0]
        FP_seg_dic['FP_num'] = FP_seg_dic['FP_num'] + len(FP_seg_ids)

        for j in range(len(FP_seg_ids)):
            FP_seg_pos = np.where(FP_width_pos == FP_seg_ids[j])
            FP_seg_dic['FP_mean_width'] = FP_seg_dic['FP_mean_width'] + len(FP_seg_pos[0])
            FP_ratio.append(len(FP_seg_pos[0]))  # added to use as a comparison to the GBM skel_len
            if pixelsize:
                FP_widths.append(pixelsize * len(FP_seg_pos[0]))  # pixelsize *
            else:
                FP_widths.append(len(FP_seg_pos[0]))

        FP_width_sum = sum(FP_ratio)


        GBM_contours = cv2.findContours(GBM_in, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(GBM_contours[0]) >= 2:

            contours = GBM_contours[0]
            contours_sorted = sorted(contours, key=lambda x: x.shape[0], reverse=True)
            ctLen1 = contours_sorted[0].shape[0]
            ctLen2 = contours_sorted[1].shape[0]

            if 0.5 < ctLen1 / ctLen2 < 2:

                for i in range(2, len(contours_sorted)):
                    cv2.drawContours(GBM_in, [contours_sorted[i]], -1, 1, -1)

                skeleton = skeletonize(GBM_in)

                # Threshold that will delete a small GBM
                if np.sum(skeleton) < 500:
                    print("MIN GBM width overtaken by skeleton")  # none made by my subsample
                    print("np.sum(skeleton) ", np.sum(skeleton))
                    FP_widths = FP_widths[:-len(FP_seg_ids)]
                    continue

                skeleton, skel_len, endpoints = find_longest_path_ring(skeleton)
            else:
                for i in range(1, len(contours_sorted)):
                    cv2.drawContours(GBM_in, [contours_sorted[i]], -1, 1, -1)

                skeleton = skeletonize(GBM_in)

                # Threshold that will delete a small GBM
                if np.sum(skeleton) < 500:
                    print("MIN GBM width overtaken by skeleton")  # none made by my subsample
                    print("np.sum(skeleton) ", np.sum(skeleton))
                    FP_widths = FP_widths[:-len(FP_seg_ids)]
                    continue

                skeleton, skel_len, endpoints = find_longest_path_band(skeleton)
        else:

            skeleton, skel_len, endpoints = find_longest_path_band(skeleton)

        # Gets rid of Fake GBM with comparing FP skel length (width) to GBM length
        if (skel_len * 0.15) > FP_width_sum:
            print(FP_width_sum)
            print("Not enough FP width, GBM will not be considered")
            continue

        # FP_in creation 6/10
        FP_in = np.zeros(im_shape)
        for i in FP_seg_ids:
            FP_in[np.where(FP_ins == i)] = True
        FP_in = FP_in.astype(np.uint8)

        # adds FP mask to the image
        FP_im_new = np.maximum(FP_im_new, FP_in)
        # adds FP skeleton to image
        FP_width_img = np.maximum(FP_width_img, FP_width_pos)  # should stop skeleton of FP if fails if statement
        # add GBM_n mask to image
        GBM_im_new = np.maximum(GBM_im_new, GBM_in)
        # adds GBM skeleton
        skeleton_img = np.maximum(skeleton_img, skeleton.astype(np.uint8))

        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length'] + skel_len
        GBM_seg_dic['area'] = GBM_seg_dic['area'] + np.sum(GBM_in)

    # FP after for loop
    if FP_seg_dic['FP_num'] == 0:
        FP_seg_dic['FP_mean_width'] = 0
    else:
        if pixelsize:
            FP_seg_dic['FP_mean_width'] = pixelsize * FP_seg_dic['FP_mean_width'] / FP_seg_dic['FP_num']
        else:
            FP_seg_dic['FP_mean_width'] = FP_seg_dic['FP_mean_width'] / FP_seg_dic['FP_num']

    # GBM after for loop
    if GBM_seg_dic['skeleton_length']:
        GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['area'] / GBM_seg_dic['skeleton_length']
    else:
        GBM_seg_dic['GBM_mean_width'] = 0

    if pixelsize:
        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length'] * pixelsize
        GBM_seg_dic['area'] = GBM_seg_dic['area'] * pixelsize * pixelsize
        GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['GBM_mean_width'] * pixelsize

    else:
        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length']  # * pixelsize
        GBM_seg_dic['area'] = GBM_seg_dic['area']  # * pixelsize * pixelsize
        GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['GBM_mean_width']  # * pixelsize

    return GBM_im_new, skeleton_img, GBM_seg_dic, FP_width_img, FP_seg_dic, FP_widths, FP_im_new


def calculate_measurements_fp_split(GBM_ins, FP_ins, pixelsize, SDheight):
    split_FP = True
    im_shape = GBM_ins.shape[0:2]
    GBM_n = np.max(GBM_ins)
    if GBM_n == 0:
        print("There is no GBM in this image!")
        return 1

    # Declaring variables
    FP_seg_width = []
    FP_seg_dic = {}
    FP_seg_dic['FP_num'] = 0
    FP_seg_dic['FP_mean_width'] = 0
    FP_width_img = np.zeros(im_shape)
    FP_widths = []
    # new FP declaration 6/10
    FP_im_new = np.zeros(im_shape, dtype=np.uint8)

    skeleton_img = np.zeros(im_shape, dtype=np.uint8)
    GBM_im_new = np.zeros(im_shape, dtype=np.uint8)
    GBM_seg_dic = {}
    GBM_seg_dic['skeleton_length'] = 0
    GBM_seg_dic['area'] = 0

    # iterates over every GBM instance
    for i in range(GBM_n):

        GBM_in = np.zeros(im_shape)
        GBM_in[np.where(GBM_ins == i + 1)] = True
        GBM_in = GBM_in.astype(np.uint8)
        GBM_in[:2, :] = False  # 顶部
        GBM_in[-2:, :] = False  # 底部
        GBM_in[:, :2] = False  # 左边
        GBM_in[:, -2:] = False  # 右边

        # kernel = np.ones((5, 5), dtype=np.uint8)
        # GBM_in = cv2.erode(GBM_in, kernel, 1)  # dilate the GBM to calculate the FP

        # 连通域分析
        labeled_gbm = measure.label(GBM_in, connectivity=2)
        props = measure.regionprops(labeled_gbm)

        if len(props) > 1:
            # 找到最大面积的区域
            largest_region = max(props, key=lambda x: x.area)
            largest_label = largest_region.label

            # 保留最大连通域
            GBM_in = (labeled_gbm == largest_label).astype(np.uint8)

        if np.sum(GBM_in) < 5000:  # (MINGBMWIDTH / pixelsize) ** 2 * 3
            # print(np.sum(GBM_in), (MINGBMWIDTH / pixelsize) ** 2)
            continue
        skeleton = skeletonize(GBM_in)
        if np.sum(skeleton) < 500:  # MINGBMWIDTH / pixelsize * 10
            continue

        kernel = np.ones((SDheight, SDheight), dtype=np.uint8)
        GBM_in_dilated = cv2.dilate(GBM_in, kernel, 1)  # dilate the GBM to calculate the FP
        GBM_in_dilated = GBM_in_dilated.astype(np.uint8)
        GBM_contour_dilated, h = cv2.findContours(GBM_in_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        GBM_contour_dilated_list = [np.squeeze(arr) for arr in GBM_contour_dilated]
        GBM_contour_dilated_array = np.concatenate(GBM_contour_dilated_list, axis=0)
        # GBM_contour_dilated = np.squeeze(GBM_contour_dilated[0]) # GBM contour after dilation
        GBM_contour_dilated_img = np.zeros(im_shape).astype(np.uint8)
        GBM_contour_dilated_img[GBM_contour_dilated_array[:, 1], GBM_contour_dilated_array[:,
                                                                 0]] = 1  # image of the GBM contour after dilation
        FP_width_pos = GBM_contour_dilated_img * FP_ins  # width image of each FP along the i th GBM
        #

        FP_seg_ids = np.unique(FP_width_pos)  # the ids of the FP along the i th GBM
        FP_seg_ids = FP_seg_ids.astype(np.int64)
        FP_ratio = []  # initialize list for sum of FP_widths
        FP_seg_ids = FP_seg_ids[FP_seg_ids != 0]

        if split_FP == True:
            FP_width_pos_split = measure.label(FP_width_pos, connectivity=2)
            regions = measure.regionprops(FP_width_pos_split)
            valid_region_count = sum(1 for region in regions if region.area > 2)
            FP_seg_dic['FP_num'] = FP_seg_dic['FP_num'] + valid_region_count

            # FP_width_pos_split = measure.label(FP_width_pos, connectivity=2)
            # FP_seg_split_ids = np.unique(FP_width_pos_split)  # the ids of the FP along the i th GBM
            # FP_seg_split_ids = FP_seg_split_ids.astype(np.int64)
            # FP_seg_split_ids = FP_seg_split_ids[FP_seg_split_ids != 0]
            # FP_seg_dic['FP_num'] = FP_seg_dic['FP_num'] + len(FP_seg_split_ids)
        else:
            FP_seg_dic['FP_num'] = FP_seg_dic['FP_num'] + len(FP_seg_ids)

        for j in range(len(FP_seg_ids)):
            FP_seg_pos = np.where(FP_width_pos == FP_seg_ids[j])
            FP_seg_dic['FP_mean_width'] = FP_seg_dic['FP_mean_width'] + len(FP_seg_pos[0])
            FP_ratio.append(len(FP_seg_pos[0]))  # added to use as a comparison to the GBM skel_len
            if pixelsize:
                FP_widths.append(pixelsize * len(FP_seg_pos[0]))  # pixelsize *
            else:
                FP_widths.append(len(FP_seg_pos[0]))

        FP_width_sum = sum(FP_ratio)

        GBM_contours = cv2.findContours(GBM_in, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(GBM_contours[0]) >= 2:

            contours = GBM_contours[0]
            contours_sorted = sorted(contours, key=lambda x: x.shape[0], reverse=True)
            ctLen1 = contours_sorted[0].shape[0]
            ctLen2 = contours_sorted[1].shape[0]

            if 0.5 < ctLen1 / ctLen2 < 2:

                for i in range(2, len(contours_sorted)):
                    cv2.drawContours(GBM_in, [contours_sorted[i]], -1, 1, -1)

                skeleton = skeletonize(GBM_in)

                # Threshold that will delete a small GBM
                if np.sum(skeleton) < 500:
                    print("MIN GBM width overtaken by skeleton")  # none made by my subsample
                    print("np.sum(skeleton) ", np.sum(skeleton))
                    FP_widths = FP_widths[:-len(FP_seg_ids)]
                    continue

                skeleton, skel_len, endpoints = find_longest_path_ring(skeleton)
            else:
                for i in range(1, len(contours_sorted)):
                    cv2.drawContours(GBM_in, [contours_sorted[i]], -1, 1, -1)

                skeleton = skeletonize(GBM_in)

                # Threshold that will delete a small GBM
                if np.sum(skeleton) < 500:
                    print("MIN GBM width overtaken by skeleton")  # none made by my subsample
                    print("np.sum(skeleton) ", np.sum(skeleton))
                    FP_widths = FP_widths[:-len(FP_seg_ids)]
                    continue

                skeleton, skel_len, endpoints = find_longest_path_band(skeleton)
        else:

            skeleton, skel_len, endpoints = find_longest_path_band(skeleton)

        # Gets rid of Fake GBM with comparing FP skel length (width) to GBM length
        if (skel_len * 0.01) > FP_width_sum:
            print(FP_width_sum)
            print("Not enough FP width, GBM will not be considered")
            continue

        # FP_in creation 6/10
        FP_in = np.zeros(im_shape)
        for i in FP_seg_ids:
            FP_in[np.where(FP_ins == i)] = True
        FP_in = FP_in.astype(np.uint8)

        # adds FP mask to the image
        FP_im_new = np.maximum(FP_im_new, FP_in)
        # adds FP skeleton to image
        FP_width_img = np.maximum(FP_width_img, FP_width_pos)  # should stop skeleton of FP if fails if statement
        # add GBM_n mask to image
        GBM_im_new = np.maximum(GBM_im_new, GBM_in)
        # adds GBM skeleton
        skeleton_img = np.maximum(skeleton_img, skeleton.astype(np.uint8))

        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length'] + skel_len
        GBM_seg_dic['area'] = GBM_seg_dic['area'] + np.sum(GBM_in)

    # FP after for loop
    if FP_seg_dic['FP_num'] == 0:
        FP_seg_dic['FP_mean_width'] = 0
    else:
        if pixelsize:
            FP_seg_dic['FP_mean_width'] = pixelsize * FP_seg_dic['FP_mean_width'] / FP_seg_dic['FP_num']
        else:
            FP_seg_dic['FP_mean_width'] = FP_seg_dic['FP_mean_width'] / FP_seg_dic['FP_num']

    # GBM after for loop
    if GBM_seg_dic['skeleton_length']:
        GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['area'] / GBM_seg_dic['skeleton_length']
    else:
        GBM_seg_dic['GBM_mean_width'] = 0

    if pixelsize:
        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length'] * pixelsize
        GBM_seg_dic['area'] = GBM_seg_dic['area'] * pixelsize * pixelsize
        GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['GBM_mean_width'] * pixelsize

    else:
        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length']  # * pixelsize
        GBM_seg_dic['area'] = GBM_seg_dic['area']  # * pixelsize * pixelsize
        GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['GBM_mean_width']  # * pixelsize

    return GBM_im_new, skeleton_img, GBM_seg_dic, FP_width_img, FP_seg_dic, FP_widths, FP_im_new


def calculate_measurements_before(GBM_ins, FP_ins, pixelsize, SDheight):
    split_FP = True
    im_shape = GBM_ins.shape[0:2]
    GBM_n = np.max(GBM_ins)
    if GBM_n == 0:
        print("There is no GBM in this image!")
        return 1

    # Declaring variables
    FP_seg_width = []
    FP_seg_dic = {}
    FP_seg_dic['FP_num'] = 0
    FP_seg_dic['FP_mean_width'] = 0
    FP_width_img = np.zeros(im_shape)
    FP_widths = []
    # new FP declaration 6/10
    FP_im_new = np.zeros(im_shape, dtype=np.uint8)

    skeleton_img = np.zeros(im_shape, dtype=np.uint8)
    GBM_im_new = np.zeros(im_shape, dtype=np.uint8)
    GBM_seg_dic = {}
    GBM_seg_dic['skeleton_length'] = 0
    GBM_seg_dic['area'] = 0

    # iterates over every GBM instance
    for i in range(GBM_n):

        GBM_in = np.zeros(im_shape)
        GBM_in[np.where(GBM_ins == i + 1)] = True
        GBM_in = GBM_in.astype(np.uint8)
        GBM_in[:2, :] = False  # 顶部
        GBM_in[-2:, :] = False  # 底部
        GBM_in[:, :2] = False  # 左边
        GBM_in[:, -2:] = False  # 右边

        # 连通域分析
        labeled_gbm = measure.label(GBM_in, connectivity=2)
        props = measure.regionprops(labeled_gbm)

        if len(props) > 1:
            # 找到最大面积的区域
            largest_region = max(props, key=lambda x: x.area)
            largest_label = largest_region.label

            # 保留最大连通域
            GBM_in = (labeled_gbm == largest_label).astype(np.uint8)

        if np.sum(GBM_in) < 5000:  # (MINGBMWIDTH / pixelsize) ** 2 * 3
            # print(np.sum(GBM_in), (MINGBMWIDTH / pixelsize) ** 2)
            continue
        skeleton = skeletonize(GBM_in)
        skel_len = np.sum(skeleton)
        if skel_len < 500:  # MINGBMWIDTH / pixelsize * 10
            continue

        kernel = np.ones((SDheight, SDheight), dtype=np.uint8)
        GBM_in_dilated = cv2.dilate(GBM_in, kernel, 1)  # dilate the GBM to calculate the FP
        GBM_in_dilated = GBM_in_dilated.astype(np.uint8)
        GBM_contour_dilated, h = cv2.findContours(GBM_in_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        GBM_contour_dilated_list = [np.squeeze(arr) for arr in GBM_contour_dilated]
        GBM_contour_dilated_array = np.concatenate(GBM_contour_dilated_list, axis=0)
        # GBM_contour_dilated = np.squeeze(GBM_contour_dilated[0]) # GBM contour after dilation
        GBM_contour_dilated_img = np.zeros(im_shape).astype(np.uint8)
        GBM_contour_dilated_img[GBM_contour_dilated_array[:, 1], GBM_contour_dilated_array[:,
                                                                 0]] = 1  # image of the GBM contour after dilation
        FP_width_pos = GBM_contour_dilated_img * FP_ins  # width image of each FP along the i th GBM
        #

        FP_seg_ids = np.unique(FP_width_pos)  # the ids of the FP along the i th GBM
        FP_seg_ids = FP_seg_ids.astype(np.int64)
        FP_ratio = []  # initialize list for sum of FP_widths
        FP_seg_ids = FP_seg_ids[FP_seg_ids != 0]

        if split_FP == True:
            FP_width_pos_split = measure.label(FP_width_pos, connectivity=2)
            regions = measure.regionprops(FP_width_pos_split)
            valid_region_count = sum(1 for region in regions if region.area > 2)
            FP_seg_dic['FP_num'] = FP_seg_dic['FP_num'] + valid_region_count

            # FP_width_pos_split = measure.label(FP_width_pos, connectivity=2)
            # FP_seg_split_ids = np.unique(FP_width_pos_split)  # the ids of the FP along the i th GBM
            # FP_seg_split_ids = FP_seg_split_ids.astype(np.int64)
            # FP_seg_split_ids = FP_seg_split_ids[FP_seg_split_ids != 0]
            # FP_seg_dic['FP_num'] = FP_seg_dic['FP_num'] + len(FP_seg_split_ids)
        else:
            FP_seg_dic['FP_num'] = FP_seg_dic['FP_num'] + len(FP_seg_ids)

        for j in range(len(FP_seg_ids)):
            FP_seg_pos = np.where(FP_width_pos == FP_seg_ids[j])
            FP_seg_dic['FP_mean_width'] = FP_seg_dic['FP_mean_width'] + len(FP_seg_pos[0])
            FP_ratio.append(len(FP_seg_pos[0]))  # added to use as a comparison to the GBM skel_len
            if pixelsize:
                FP_widths.append(pixelsize * len(FP_seg_pos[0]))  # pixelsize *
            else:
                FP_widths.append(len(FP_seg_pos[0]))

        FP_width_sum = sum(FP_ratio)

        # GBM_contours = cv2.findContours(GBM_in, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # if len(GBM_contours[0]) >= 2:

        #     contours = GBM_contours[0]
        #     contours_sorted = sorted(contours, key=lambda x: x.shape[0], reverse=True)
        #     ctLen1 = contours_sorted[0].shape[0]
        #     ctLen2 = contours_sorted[1].shape[0]

        #     if 0.5 < ctLen1 / ctLen2 < 2:

        #         for i in range(2, len(contours_sorted)):
        #             cv2.drawContours(GBM_in, [contours_sorted[i]], -1, 1, -1)

        #         skeleton = skeletonize(GBM_in)

        #         # Threshold that will delete a small GBM
        #         if np.sum(skeleton) < 500:
        #             print("MIN GBM width overtaken by skeleton")  # none made by my subsample
        #             print("np.sum(skeleton) ", np.sum(skeleton))
        #             FP_widths = FP_widths[:-len(FP_seg_ids)]
        #             continue

        #         skeleton, skel_len, endpoints = find_longest_path_ring(skeleton)
        #     else:
        #         for i in range(1, len(contours_sorted)):
        #             cv2.drawContours(GBM_in, [contours_sorted[i]], -1, 1, -1)

        #         skeleton = skeletonize(GBM_in)

        #         # Threshold that will delete a small GBM
        #         if np.sum(skeleton) < 500:
        #             print("MIN GBM width overtaken by skeleton")  # none made by my subsample
        #             print("np.sum(skeleton) ", np.sum(skeleton))
        #             FP_widths = FP_widths[:-len(FP_seg_ids)]
        #             continue

        #         skeleton, skel_len, endpoints = find_longest_path_band(skeleton)
        # else:

            # skeleton, skel_len, endpoints = find_longest_path_band(skeleton)

        # Gets rid of Fake GBM with comparing FP skel length (width) to GBM length
        if (skel_len * 0.01) > FP_width_sum:
            print(FP_width_sum)
            print("Not enough FP width, GBM will not be considered")
            continue

        # FP_in creation 6/10
        FP_in = np.zeros(im_shape)
        for i in FP_seg_ids:
            FP_in[np.where(FP_ins == i)] = True
        FP_in = FP_in.astype(np.uint8)

        # adds FP mask to the image
        FP_im_new = np.maximum(FP_im_new, FP_in)
        # adds FP skeleton to image
        FP_width_img = np.maximum(FP_width_img, FP_width_pos)  # should stop skeleton of FP if fails if statement
        # add GBM_n mask to image
        GBM_im_new = np.maximum(GBM_im_new, GBM_in)
        # adds GBM skeleton
        skeleton_img = np.maximum(skeleton_img, skeleton.astype(np.uint8))

        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length'] + skel_len
        GBM_seg_dic['area'] = GBM_seg_dic['area'] + np.sum(GBM_in)

    # FP after for loop
    if FP_seg_dic['FP_num'] == 0:
        FP_seg_dic['FP_mean_width'] = 0
    else:
        if pixelsize:
            FP_seg_dic['FP_mean_width'] = pixelsize * FP_seg_dic['FP_mean_width'] / FP_seg_dic['FP_num']
        else:
            FP_seg_dic['FP_mean_width'] = FP_seg_dic['FP_mean_width'] / FP_seg_dic['FP_num']

    # GBM after for loop
    if GBM_seg_dic['skeleton_length']:
        GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['area'] / GBM_seg_dic['skeleton_length']
    else:
        GBM_seg_dic['GBM_mean_width'] = 0

    if pixelsize:
        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length'] * pixelsize
        GBM_seg_dic['area'] = GBM_seg_dic['area'] * pixelsize * pixelsize
        GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['GBM_mean_width'] * pixelsize

    else:
        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length']  # * pixelsize
        GBM_seg_dic['area'] = GBM_seg_dic['area']  # * pixelsize * pixelsize
        GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['GBM_mean_width']  # * pixelsize

    return GBM_im_new, skeleton_img, GBM_seg_dic, FP_width_img, FP_seg_dic, FP_widths, FP_im_new
# leftover information from the single file version

# GBM_closing_kernel_size = 50

MINGBMWIDTH = 300

cl1 = (255, 0, 0)  # color for FP
cl2 = (0, 0, 255)  # color for GBM
cl3 = (255, 255, 255)  # color for FP width
cl4 = (255, 255, 255)  # color for GBM skeleton
cl5 = (0, 0, 255)  # color for GBM width

GBM_closing_kernel_size = 10
GBM_opening_kernel_size = 10
GBM_dilation_kernel_size = 20  ## decided the distance from GBM to calculate the FP width
# GBM = 1, FP = 2
directions1 = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]

import cv2
import numpy as np


def get_neighbors(skel, point, directions, im_height, im_width):
    x, y = point
    neighbors = []
    # directions is one (or two for dir2) step change from the point
    for dx, dy in directions:
        # this if statement is to make sure the neighbors are on screen to not throw an error
        if x + dx < im_height and y + dy < im_width:
            # Pretty sure this is if the skeleton contains a point on the point plus the direction
            if skel[x + dx, y + dy]:
                neighbors.append((x + dx, y + dy))
    return neighbors


def find_endpoints(skel):
    newskel = np.pad(skel, 5, 'constant', constant_values=0)
    endpoint_kernel = np.array([[1, 1, 1],
                                [1, 10, 1],
                                [1, 1, 1]], dtype=np.uint8)
    convolution_result = cv2.filter2D(newskel.astype(np.uint8), -1, endpoint_kernel)
    convolution_result = convolution_result[5:-5, 5:-5]
    endpoints = (convolution_result == 11)
    return endpoints


def find_bifurcation_points(skel):
    bifurcation_kernel = np.array([[1, 1, 1],
                                   [1, 10, 1],
                                   [1, 1, 1]], dtype=np.uint8)
    convolution_result = cv2.filter2D(skel.astype(np.uint8), -1, bifurcation_kernel)
    bifurcation_points = (convolution_result >= 13)

    return bifurcation_points


def find_longest_path(skel, bifurcation_coords, endpoint_coords):
    directions1 = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
    directions2 = [[-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2],
                   [-2, 1], [2, 1],
                   [-2, 0], [2, 0],
                   [-2, -1], [2, -1],
                   [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2]]

    im_height, im_width = skel.shape[0:2]

    def track_paths(skel, vertices, start, second, neighbors1):

        path = []
        # neighbors = get_neighbors(skel, start, directions1, im_height, im_width)

        visited = {start}

        # ZACH CHANGE- got rid of neighbors1
        # for neighbor in neighbors1:
        #    visited.add(neighbor)
        visited.add(second)

        path.append(start)
        path.append(second)
        endpoint = second

        while path:
            x, y = path[-1]
            visited.add((x, y))
            neighbors = get_neighbors(skel, (x, y), directions1, im_height, im_width)
            n_visited = 0
            for neighbor in neighbors:
                if neighbor in visited:
                    n_visited = n_visited + 1
                if neighbor not in visited:
                    path.append(neighbor)
                    if neighbor in vertices:
                        endpoint = neighbor
                        return path, endpoint
            if n_visited == len(neighbors):
                return path, endpoint

    ## build the graph using the simplified bifurcation and end point
    vertices = bifurcation_coords + endpoint_coords
    vertice2index = {}
    index2vertice = {}
    for i, v in enumerate(vertices):
        index2vertice[i] = v
        vertice2index[v] = i
    G = nx.Graph()
    for i in range(len(vertices)):
        G.add_node(i)
    ## build the weighted edges of the graph
    pathdict = {}
    edge_num = 0  # z add

    for bif_point in bifurcation_coords:

        neighbors1 = get_neighbors(skel, bif_point, directions1, im_height, im_width)
        # neighbors2 = get_neighbors(bif_point, directions2, im_height, im_width)
        for neighbor in neighbors1:

            path, endpoint = track_paths(skel, vertices, bif_point, neighbor, neighbors1)

            # print("Path Length", len(path))

            if not endpoint in vertices:
                print("not endpoint in vertices")

                print(vertices)
                print(neighbor)

                continue

            # startindex will be the bifur int
            startindex = vertice2index[bif_point]
            endpindex = vertice2index[endpoint]
            # weight is the value of the edge (should be used in which path is the longest?)
            weight = len(path)
            if (startindex, endpindex) in pathdict.keys():
                oldpath = pathdict[(startindex, endpindex)]

                ####Gets activated when I changed from neighbors1 to only second, probably because it has one less point#####
                if len(oldpath) > len(path):
                    # print("nested", path)
                    pathdict[(startindex, endpindex)] = path
            else:
                pathdict[(startindex, endpindex)] = path
                pathdict[(endpindex, startindex)] = path
            if endpoint_coords in path:
                print(path)
            G.add_edge(startindex, endpindex, weight=weight)
            edge_num += 1

            # print("bifurc weight", weight)
        # all of the bifurcs are included
    # print("EDGE NUM", edge_num)
    ####end of bifurcation loop ######

    # this should be where the problem is
    max_distance = -1
    node_pair = (None, None)

    # maybe could start with just endpoints to make it more effiecient
    for startpoint in G.nodes():
        for endpoint in G.nodes():
            if startpoint != endpoint:
                distance = nx.dijkstra_path_length(G, source=startpoint, target=endpoint, weight='weight')
                # distance, path = nx.bidirectional_dijkstra(G, source=startpoint, target=endpoint, weight='weight')

                # print("Distance of one", distance)
                if distance > max_distance:
                    max_distance = distance
                    node_pair = (startpoint, endpoint)
    # print("Max Distance", max_distance)

    pathV = nx.shortest_path(G, source=node_pair[0], target=node_pair[1])
    endpoints = (index2vertice[node_pair[0]], index2vertice[node_pair[1]])

    newskel = []
    for i in range(len(pathV) - 1):
        newskel = newskel + pathdict[(pathV[i], pathV[i + 1])]
    newskelimage = np.zeros(skel.shape)
    rows, cols = zip(*newskel)
    newskelimage[rows, cols] = True

    return newskelimage, max_distance, endpoints


def find_longest_path_ring(skel):
    directions1 = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
    im_height, im_width = skel.shape[0:2]

    def get_8_neighborhood(matrix, x, y):
        rows, cols = matrix.shape
        neighbors = []
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if matrix[nx, ny] == 1:
                    neighbors.append((nx, ny))
        return neighbors

    def track_paths_ring(skel_bool, bifurcation, start):
        # track_paths(skel, vertices, neighbor, neighbors1, neighbors2)

        path = []
        # neighbors = get_neighbors(start, directions1, im_height, im_width)
        visited = {start}
        path.append(start)
        while path:
            x, y = path[-1]
            visited.add((x, y))
            neighbors = get_neighbors(skel_bool, (x, y), directions1, im_height, im_width)
            n_visited = 0
            for neighbor in neighbors:
                if neighbor in visited:
                    n_visited = n_visited + 1
                if neighbor not in visited:
                    if neighbor in bifurcation:
                        neighbor8 = get_8_neighborhood(skel_bool, neighbor[0], neighbor[1])
                        neighbor8_list = list(neighbor8)
                        neighbor8_list.remove((x, y))
                        neighbor8 = tuple(neighbor8_list)
                        if len(neighbor8) >= 2:
                            if abs(neighbor8[0][0] - neighbor8[1][0]) <= 1 and abs(
                                    neighbor8[0][1] - neighbor8[1][1]) <= 1:
                                path.append(neighbor)
                        return path
                    else:
                        path.append(neighbor)
            if n_visited == len(neighbors):
                return path

    def remove_branch(skeleton_bool, endpoint_coords, bifurcation_coords):

        for endpoint in endpoint_coords:
            branch = track_paths_ring(skeleton_bool, bifurcation_coords, endpoint)
            row_indices, col_indices = zip(*branch)
            skeleton_bool[row_indices, col_indices] = False

        skel_len = np.sum(skeleton_bool)
        skel_points = np.where(skeleton_bool > 0)
        endpoint = (skel_points[0][0], skel_points[1][0])
        endpoints = [endpoint, endpoint]

        return skeleton_bool, skel_len, endpoints
    
    def remove_branch_min(skeleton_bool, endpoint_coords, bifurcation_coords, min_branch_length=50):
        branch_removed = False
        for endpoint in endpoint_coords:
            branch = track_paths_ring(skeleton_bool, bifurcation_coords, endpoint)
            if branch is None or len(branch) == 0:
                continue
            if len(branch) < min_branch_length:
                row_indices, col_indices = zip(*branch)
                skeleton_bool[row_indices, col_indices] = False
                branch_removed = True

        skel_len = np.sum(skeleton_bool)
        skel_points = np.where(skeleton_bool > 0)
        endpoint = (skel_points[0][0], skel_points[1][0])
        endpoints = [endpoint, endpoint]

        return skeleton_bool, skel_len, endpoints, branch_removed

    skeleton_bool = skel.astype(bool)
    endpoints = find_endpoints(skel)
    bifurcation_points = find_bifurcation_points(skel)

    endpoint_coords = list(zip(*np.where(endpoints)))
    bifurcation_coords = list(zip(*np.where(bifurcation_points)))
    if len(endpoint_coords) == 0:
        skel_len = np.sum(skeleton_bool)
        skel_points = np.where(skeleton_bool > 0)
        endpoint = (skel_points[0][0], skel_points[1][0])
        endpoints = [endpoint, endpoint]

    while len(endpoint_coords) > 0:
        # skeleton_bool, skel_len, endpoints = remove_branch(skeleton_bool, endpoint_coords, bifurcation_coords)
        skeleton_bool, skel_len, endpoints, branch_removed = remove_branch_min(skeleton_bool, endpoint_coords, bifurcation_coords, min_branch_length=300)
        if not branch_removed:
            break
        endpoints = find_endpoints(skeleton_bool.astype(np.uint8))
        bifurcation_points = find_bifurcation_points(skeleton_bool.astype(np.uint8))
        endpoint_coords = list(zip(*np.where(endpoints)))
        bifurcation_coords = list(zip(*np.where(bifurcation_points)))

    return skeleton_bool, skel_len, endpoints


def find_longest_path_band(skeleton):
    endpoints = find_endpoints(skeleton)
    bifurcation_points = find_bifurcation_points(skeleton)

    skeleton_bool = skeleton.astype(bool)
    endpoint_coords = list(zip(*np.where(endpoints)))
    bifurcation_coords = list(zip(*np.where(bifurcation_points)))
    # skeleton = find_shortest_path_to_endpoint(skeleton_bool, bifurcation_coords, endpoint_coords)
    if len(endpoint_coords) == 0:
        skeleton = np.zeros(skeleton.shape)
        skel_len = 0
        twoendpoints = ()
    elif len(bifurcation_coords) == 0:
        skeleton = skeleton_bool
        skel_len = np.sum(skeleton)
        twoendpoints = endpoint_coords
        # twoendpoints = tuple(zip(*endpoint_coords))
    else:

        skeleton, skel_len, twoendpoints = find_longest_path(skeleton_bool, bifurcation_coords, endpoint_coords)
    return skeleton, skel_len, twoendpoints


def save_maskon_image(image, GBM_img, FP_img, skeleton_img, FP_width_img, outfile):
    image_rgb = np.stack([image] * 3, axis=-1)
    image_rgba = np.concatenate([image_rgb, np.full((*image.shape, 1), 255, dtype=np.uint8)], axis=-1)
    colored_mask = np.zeros_like(image_rgba, dtype=np.uint8)  # 创建彩色掩码数组

    red_mask = (GBM_img == 1)
    blue_mask = (FP_img == 1)

    white_mask = np.where(skeleton_img == 1)
    width_mask = np.where(FP_width_img != 0)

    colored_mask[..., :3][red_mask] = [255, 0, 0]  # red
    colored_mask[..., :3][blue_mask] = [0, 0, 255]  # blue
    colored_mask[..., :3][white_mask] = [255, 255, 255]
    colored_mask[..., :3][width_mask] = [255, 255, 255]
    colored_mask[..., 3][red_mask | blue_mask] = 127 * 2

    mask_indices = red_mask | blue_mask

    image_rgba[mask_indices] = image_rgba[mask_indices] * 0.5 + colored_mask[mask_indices] * 0.5
    image_rgba[white_mask] = colored_mask[white_mask]
    image_rgba[width_mask] = colored_mask[width_mask]
    image_rgba = image_rgba.astype(np.uint8)

    final_image = Image.fromarray(image_rgba, 'RGBA')
    final_image.save(outfile)

def save_insmaskon_image(image, GBM_img, FP_ins, skeleton_img, FP_width_img, outfile):
    image_rgb = np.stack([image] * 3, axis=-1)
    image_rgba = np.concatenate([image_rgb, np.full((*image.shape, 1), 255, dtype=np.uint8)], axis=-1)
    colored_mask = np.zeros_like(image_rgba, dtype=np.uint8)  # 彩色掩码层

    blue_mask = (GBM_img == 1)

    # 为每个 FP 实例分配不同的红色（可变亮度）
    unique_fp_ids = np.unique(FP_ins)
    unique_fp_ids = unique_fp_ids[unique_fp_ids != 0]  # 排除背景

    for fp_id in unique_fp_ids:
        mask = (FP_ins == fp_id)
        # 随机红色（亮度变化），R 固定为 255，G/B 随机变动
        red_variant = [random.randint(150, 255), random.randint(0, 80), random.randint(0, 80)]
        colored_mask[..., :3][mask] = red_variant
        colored_mask[..., 3][mask] = 255  # 设置 alpha 通道可见

    kernel = np.ones((5, 5), np.uint8)  # 3x3卷积核，可根据需要变大
    skeleton_dilated = cv2.dilate((skeleton_img == 1).astype(np.uint8), kernel)
    width_dilated = cv2.dilate((FP_width_img != 0).astype(np.uint8), kernel)

    white_mask = np.where(skeleton_dilated == 1)
    width_mask = np.where(width_dilated != 0)

    colored_mask[..., :3][blue_mask] = [0, 0, 255]  # GBM - 固定红色
    colored_mask[..., 3][blue_mask] = 255

    colored_mask[..., :3][white_mask] = [255, 255, 255]
    colored_mask[..., :3][width_mask] = [255, 255, 255]
    colored_mask[..., 3][white_mask] = 255
    colored_mask[..., 3][width_mask] = 255

    mask_indices = colored_mask[..., 3] > 0

    image_rgba[mask_indices] = (
        image_rgba[mask_indices].astype(np.float32) * 0.5 +
        colored_mask[mask_indices].astype(np.float32) * 0.5
    )
    image_rgba = image_rgba.astype(np.uint8)

    final_image = Image.fromarray(image_rgba, 'RGBA')
    final_image.save(outfile)
def save_GBM_maskon_image(image, GBM_img, skeleton_img, rect_list, outfile):
    im_shape = GBM_img.shape[0:2]

    GBM_img = np.int64(GBM_img > 0)
    GBM_img_c = np.zeros((im_shape[0], im_shape[1], 3))
    for i in range(3):
        GBM_img_c[:, :, i] = GBM_img * cl1[i]

    colored_mask = GBM_img_c

    GBM_ske_index = np.where(skeleton_img == 1)
    colored_mask[GBM_ske_index[0], GBM_ske_index[1], 0] = cl4[0]
    colored_mask[GBM_ske_index[0], GBM_ske_index[1], 1] = cl4[1]
    colored_mask[GBM_ske_index[0], GBM_ske_index[1], 2] = cl4[2]

    colored_mask = colored_mask.astype(np.int64)

    image_rgb = np.stack([image] * 3, axis=-1)
    alpha = 0.5
    overlayed_image = (image_rgb * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    overlayed_image = Image.fromarray(overlayed_image)
    draw = ImageDraw.Draw(overlayed_image)

    for rects in rect_list:
        for rect in rects:
            rect = [tuple(re[::-1]) for re in rect]
            draw.polygon(rect, outline='white')
    # overlayed_image.show()
    overlayed_image.save(outfile)
    # print('done!')

