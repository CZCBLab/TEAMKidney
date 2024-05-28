## calculate the GBM mean width from the DL model prediction mask image
## the mean with is calculated by area / skeleton length
## the skeleton length is pruned by graph algorith, dijkstra for example
## can process the skeleton extraction with rings
## calculate the FP mean width from the DL model prediction mask image
## solving some bugs with ring GBM
import json
import os
import pandas as pd
from skimage.morphology import skeletonize, medial_axis
from skimage import measure, morphology
from tqdm import tqdm
import networkx as nx
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

GBM_closing_kernel_size = 50
FP_openning_kernel_size = 15

MINGBMWIDTH = 200

cl1 = (255, 0, 0) # color for FP
cl2 = (0, 0, 255) # color for GBM
cl3 = (255, 255, 255)     # color for FP width
cl4 = (255, 255, 255)     # color for GBM skeleton
cl5 = (0, 0, 255)     # color for GBM width

GBM_closing_kernel_size = 10
GBM_opening_kernel_size = 10
GBM_dilation_kernel_size = 20 ## decided the distance from GBM to calculate the FP width
# GBM = 1, FP = 2
directions1 = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]

import cv2
import numpy as np


def get_neighbors(skel, point, directions, im_height, im_width):
    x, y = point
    neighbors = []
    for dx, dy in directions:
        if x + dx < im_height and y + dy < im_width:
            if skel[x + dx, y + dy]:
                neighbors.append((x + dx, y + dy))
    return neighbors
def find_endpoints(skel):
    endpoint_kernel = np.array([[1, 1, 1],
                                [1, 10, 1],
                                [1, 1, 1]], dtype=np.uint8)
    convolution_result = cv2.filter2D(skel.astype(np.uint8), -1, endpoint_kernel)
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

    #

    # def get_neighbors(skel, point, directions, im_height, im_width):
    #     x, y = point
    #     neighbors = []
    #     for dx, dy in directions:
    #         if x + dx < im_height and y + dy < im_width:
    #             if skel[x + dx, y + dy]:
    #                 neighbors.append((x + dx, y + dy))
    #     return neighbors

    def track_paths(skel, vertices, start, second, neighbors1):

        path = []
        # neighbors = get_neighbors(skel, start, directions1, im_height, im_width)
        visited = {start}
        for neighbor in neighbors1:
            visited.add(neighbor)

        path.append(start)
        path.append(second)
        endpoint = second
        while path:
            x, y = path[-1]
            visited.add((x, y))
            neighbors = get_neighbors(skel,(x, y), directions1, im_height, im_width)
            n_visited = 0
            for neighbor in neighbors:
                if neighbor in visited:
                    n_visited = n_visited+1
                if neighbor not in visited:
                    path.append(neighbor)
                    if neighbor in vertices:
                        endpoint = neighbor
                        return path, endpoint
            if n_visited == len(neighbors):
                return path, endpoint

    ## build the graph using the simplified bifurcation and end point
    vertices = bifurcation_coords + endpoint_coords
    # vertices = bifurcation_coords + endpoint_coords
    vertice2index = {}
    index2vertice = {}
    for i,v in enumerate(vertices):
        index2vertice[i] = v
        vertice2index[v] = i
    G = nx.Graph()
    for i in range(len(vertices)):
        G.add_node(i)
    ## build the weighted edges of the graph
    pathdict = {}
    for bif_point in bifurcation_coords:
        neighbors1 = get_neighbors(skel, bif_point, directions1, im_height, im_width)
        # neighbors2 = get_neighbors(bif_point, directions2, im_height, im_width)
        for neighbor in neighbors1:
            path, endpoint = track_paths(skel, vertices, bif_point, neighbor, neighbors1)

            if not endpoint in vertices:
                continue
            startindex = vertice2index[bif_point]
            endpindex = vertice2index[endpoint]
            weight = len(path)
            if (startindex, endpindex) in pathdict.keys():
                oldpath = pathdict[(startindex, endpindex)]
                if len(oldpath)>len(path):
                    pathdict[(startindex, endpindex)] = path
            else:
                pathdict[(startindex, endpindex)] = path
                pathdict[(endpindex, startindex)] = path
            G.add_edge(startindex, endpindex, weight=weight)

    max_distance = -1
    node_pair = (None, None)
    for startpoint in G.nodes():
        for endpoint in G.nodes():
            if startpoint != endpoint:
                distance = nx.dijkstra_path_length(G, source=startpoint, target=endpoint, weight='weight')
                if distance>max_distance:
                    max_distance = distance
                    node_pair = (startpoint, endpoint)
    pathV = nx.shortest_path(G, source=node_pair[0], target=node_pair[1])
    endpoints = (index2vertice[node_pair[0]], index2vertice[node_pair[1]])

    newskel = []
    for i in range(len(pathV)-1):
        newskel = newskel + pathdict[(pathV[i], pathV[i+1])]
    newskelimage = np.zeros(skel.shape)
    rows, cols = zip(*newskel)
    newskelimage[rows, cols] = True

    return newskelimage, max_distance, endpoints

def find_longest_path_ring(skel):
    directions1 = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
    im_height, im_width = skel.shape[0:2]

    # def get_neighbors(point, directions, im_height, im_width):
    #     x, y = point
    #     neighbors = []
    #     for dx, dy in directions:
    #         if x + dx < im_height and y + dy < im_width:
    #             if skel[x + dx, y + dy]:
    #                 neighbors.append((x + dx, y + dy))
    #     return neighbors
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

    def track_paths(skel_bool, bifurcation, start):
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
                    n_visited = n_visited+1
                if neighbor not in visited:
                    if neighbor in bifurcation:
                        neighbor8 = get_8_neighborhood(skel_bool, neighbor[0], neighbor[1])
                        neighbor8_list = list(neighbor8)
                        neighbor8_list.remove((x, y))
                        neighbor8 = tuple(neighbor8_list)
                        if len(neighbor8)>=2:
                            if abs(neighbor8[0][0] - neighbor8[1][0]) <= 1 and abs(neighbor8[0][1] - neighbor8[1][1]) <= 1:
                                path.append(neighbor)
                        return path
                    else:
                        path.append(neighbor)
            if n_visited == len(neighbors):
                return path

    def remove_branch(skeleton_bool, endpoint_coords, bifurcation_coords):
        for endpoint in endpoint_coords:
            branch = track_paths(skeleton_bool, bifurcation_coords, endpoint)
            row_indices, col_indices = zip(*branch)
            skeleton_bool[row_indices, col_indices] = False

        skel_len = np.sum(skeleton_bool)
        skel_points = np.where(skeleton_bool > 0)
        endpoint = (skel_points[0][0], skel_points[1][0])
        endpoints = [endpoint, endpoint]

        return skeleton_bool, skel_len, endpoints

    skeleton_bool = skel.astype(bool)
    endpoints = find_endpoints(skel)
    bifurcation_points = find_bifurcation_points(skel)

    endpoint_coords = list(zip(*np.where(endpoints)))
    bifurcation_coords = list(zip(*np.where(bifurcation_points)))
    if len(endpoint_coords) ==0:
        skel_len = np.sum(skeleton_bool)
        skel_points = np.where(skeleton_bool > 0)
        endpoint = (skel_points[0][0], skel_points[1][0])
        endpoints = [endpoint, endpoint]

    while len(endpoint_coords)>0:
        skeleton_bool, skel_len, endpoints = remove_branch(skeleton_bool, endpoint_coords, bifurcation_coords)
        endpoints = find_endpoints(skeleton_bool.astype(np.uint8))
        bifurcation_points = find_bifurcation_points(skeleton_bool.astype(np.uint8))
        endpoint_coords = list(zip(*np.where(endpoints)))
        bifurcation_coords = list(zip(*np.where(bifurcation_points)))

    return skeleton_bool, skel_len, endpoints
def find_longest_path_band(skeleton):
    endpoints = find_endpoints(skeleton)
    bifurcation_points = find_bifurcation_points(skeleton)

    # 获取端点和分叉点的坐标
    # y_endpoints, x_endpoints = np.where(endpoints)
    # y_bifurcation, x_bifurcation = np.where(bifurcation_points)

    # plt.imshow(skeleton, cmap='gray')
    # plt.scatter(x_endpoints, y_endpoints, color='r', s=10)  # 端点用红色标记
    # plt.scatter(x_bifurcation, y_bifurcation, color='g', s=10)  # 分叉点用绿色标记
    # plt.title('Skeleton with Endpoints and Bifurcation Points')
    # plt.axis('off')
    # plt.show()

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
# def seg_width_cal_rect(skeleton, GBM_im_new, startpoint, windowsize, recwidth):
#     seg_width = []
#     (im_height, im_width) = GBM_im_new.shape
#     skel_list = [startpoint]
#     curentpoint = startpoint
#     neighbors = get_neighbors(skeleton, curentpoint, directions1, im_height, im_width)
#     curentpoint = neighbors[0]
#     skel_list.append(curentpoint)
#     i = 1
#     # generate skel_list
#     maxiter = 200000
#     rect_list = []
#     while maxiter>0:
#         maxiter = maxiter-1
#         neighbors = get_neighbors(skeleton, curentpoint, directions1, im_height, im_width)
#         not_visited_n = [item for item in neighbors if item not in skel_list[max(i-2, 0):i+1]+skel_list[0:3]]
#         if len(not_visited_n)==1:
#             skel_list.append(not_visited_n[0])
#             curentpoint = not_visited_n[0]
#             i = i+1
#         elif len(not_visited_n)==2:
#             neighbor0 = not_visited_n[0]
#             neighbor1 = not_visited_n[1]
#             if neighbor0[0]==curentpoint[0] or neighbor0[1]==curentpoint[1]:
#                 skel_list.append(neighbor0)
#                 skel_list.append(neighbor1)
#                 curentpoint = neighbor1
#             else:
#                 skel_list.append(neighbor1)
#                 skel_list.append(neighbor0)
#                 curentpoint = neighbor0
#             i = i + 2
#         else:
#             break
#     if maxiter<=2:
#         print('time out')
#
#     RecImage = np.zeros(skeleton.shape, dtype=np.uint8)
#     for i in range(int((len(skel_list)-5)/windowsize)):
#         window_start = np.array(skel_list[i*windowsize+2])
#         start_pre = skel_list[i * windowsize]
#         start_pos = skel_list[i*windowsize+4]
#         window_end = np.array(skel_list[(i+1) * windowsize+2])
#         end_pre = skel_list[(i+1) * windowsize]
#         end_pos = skel_list[(i+1) * windowsize+4]
#
#         direction = np.array([start_pos[0]-start_pre[0], start_pos[1]-start_pre[1]])
#         normal = np.array([-direction[1], direction[0]])
#         normal = normal / np.linalg.norm(normal)
#         point1 = window_start+ normal*recwidth
#         point2 = window_start- normal*recwidth
#
#         direction = np.array([end_pos[0] - end_pre[0], end_pos[1] - end_pre[1]])
#         normal = np.array([-direction[1], direction[0]])
#         normal = normal / np.linalg.norm(normal)
#         point4 = window_end + normal * recwidth
#         point3 = window_end - normal * recwidth
#         points =[point1, point2, point3, point4]
#         points = np.expand_dims(points, axis=0).astype(np.int32)
#         points = points[:,:,::-1]
#         cv2.fillPoly(RecImage, points, i+1)
#         rect_list.append([point1, point2, point3, point4])
#     segGBM = GBM_im_new * RecImage
#     seg_width = []
#     for i in range(int((len(skel_list)-2)/windowsize)):
#         area = np.where(segGBM==i+1)
#         area = len(area[0])
#         seg_width.append(area/windowsize)
#
#     return seg_width, rect_list

def seg_width_cal(skeleton, GBM_im_new, startpoint, windowsize, recwidth):
    seg_width = []
    (im_height, im_width) = GBM_im_new.shape
    skel_list = [startpoint]
    curentpoint = startpoint
    neighbors = get_neighbors(skeleton, curentpoint, directions1, im_height, im_width)
    curentpoint = neighbors[0]
    skel_list.append(curentpoint)
    i = 1
    # generate skel_list
    maxiter = 200000
    rect_list = []
    while maxiter>0:
        maxiter = maxiter-1
        neighbors = get_neighbors(skeleton, curentpoint, directions1, im_height, im_width)
        not_visited_n = [item for item in neighbors if item not in skel_list[max(i-2, 0):i+1]+skel_list[0:3]]
        if len(not_visited_n)==1:
            skel_list.append(not_visited_n[0])
            curentpoint = not_visited_n[0]
            i = i+1
        elif len(not_visited_n)==2:
            neighbor0 = not_visited_n[0]
            neighbor1 = not_visited_n[1]
            if neighbor0[0]==curentpoint[0] or neighbor0[1]==curentpoint[1]:
                skel_list.append(neighbor0)
                skel_list.append(neighbor1)
                curentpoint = neighbor1
            else:
                skel_list.append(neighbor1)
                skel_list.append(neighbor0)
                curentpoint = neighbor0
            i = i + 2
        else:
            break
    if maxiter<=2:
        print('time out')

    RecImage = np.zeros(skeleton.shape, dtype=np.uint8)
    for i in range(int((len(skel_list)-5)/windowsize)):
        window_start = np.array(skel_list[i*windowsize+2])
        start_pre = skel_list[i * windowsize]
        start_pos = skel_list[i*windowsize+4]
        window_end = np.array(skel_list[(i+1) * windowsize+2])
        end_pre = skel_list[(i+1) * windowsize]
        end_pos = skel_list[(i+1) * windowsize+4]

        direction = np.array([start_pos[0]-start_pre[0], start_pos[1]-start_pre[1]])
        normal = np.array([-direction[1], direction[0]])
        normal = normal / np.linalg.norm(normal)
        point1 = window_start+ normal*recwidth
        point2 = window_start- normal*recwidth

        direction = np.array([end_pos[0] - end_pre[0], end_pos[1] - end_pre[1]])
        normal = np.array([-direction[1], direction[0]])
        normal = normal / np.linalg.norm(normal)
        point4 = window_end + normal * recwidth
        point3 = window_end - normal * recwidth
        points =[point1, point2, point3, point4]
        points = np.expand_dims(points, axis=0).astype(np.int32)
        points = points[:,:,::-1]
        cv2.fillPoly(RecImage, points, i+1)
        rect_list.append([point1, point2, point3, point4])
    segGBM = GBM_im_new * RecImage
    seg_width = []
    for i in range(int((len(skel_list)-2)/windowsize)):
        area = np.where(segGBM==i+1)
        area = len(area[0])
        seg_width.append(area/windowsize)

    return seg_width, rect_list

def GBM_measures_cal(GBM_ins, pixelsize):
    # GBM_id = 0
    im_shape = GBM_ins.shape[0:2]
    # GBM_ins = measure.label(GBM_ins, connectivity=2)
    GBM_n = np.max(GBM_ins)
    if GBM_n == 0:
        print("There is no GBM in this image!")
        return 1

    ## calculate the GBM width
    skeleton_img = np.zeros(im_shape,dtype=np.uint8)
    GBM_im_new = np.zeros(im_shape,dtype=np.uint8)
    GBM_seg_dic = {}
    GBM_seg_dic['skeleton_length'] = 0
    GBM_seg_dic['area'] = 0
    for i in range(GBM_n):
        ins = np.zeros(im_shape)
        ins[np.where(GBM_ins == i+1)] = True
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        # ins = cv2.morphologyEx(ins, cv2.MORPH_CLOSE, kernel, iterations=1)

        ins = ins.astype(np.uint8)
        if np.sum(ins) < (MINGBMWIDTH/pixelsize)**2*10:
            continue
        GBM_im_new = np.maximum(GBM_im_new, ins)
        skeleton = skeletonize(ins)
        skeleton_img = np.maximum(skeleton_img, skeleton.astype(np.uint8))
        if np.sum(skeleton) < MINGBMWIDTH/pixelsize*10:
            continue

        GBM_contours = cv2.findContours(ins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(GBM_contours[0]) == 2:
            ctLen1 = GBM_contours[0][0].shape[0]
            ctLen2 = GBM_contours[0][1].shape[0]
            if 0.5<ctLen1/ctLen2<2:
                skeleton, skel_len, endpoints = find_longest_path_ring(skeleton)
            else:
                skeleton, skel_len, endpoints = find_longest_path_band(skeleton)
        else:
            skeleton, skel_len, endpoints = find_longest_path_band(skeleton)

        # skeleton_img = np.maximum(skeleton_img, skeleton.astype(np.uint8))
        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length'] + skel_len
        GBM_seg_dic['area'] = GBM_seg_dic['area'] + np.sum(ins)


    if GBM_seg_dic['skeleton_length']:
        GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['area']/GBM_seg_dic['skeleton_length']
    else:
        GBM_seg_dic['GBM_mean_width'] = 0

    GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length'] * pixelsize
    GBM_seg_dic['area'] = GBM_seg_dic['area'] * pixelsize * pixelsize
    GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['GBM_mean_width'] * pixelsize

    return GBM_im_new, skeleton_img, GBM_seg_dic

def GBM_seg_measures_cal(GBM_ins):
    # GBM_id = 0
    im_shape = GBM_ins.shape[0:2]
    GBM_ins = measure.label(GBM_ins, connectivity=2)
    GBM_n = np.max(GBM_ins)
    if GBM_n == 0:
        print("There is no GBM in this image!")
        return 1

    ## calculate the GBM width
    skeleton_img = np.zeros(im_shape,dtype=np.uint8)
    GBM_im_new = np.zeros(im_shape,dtype=np.uint8)
    GBM_seg_dic = {}
    GBM_seg_dic['skeleton_length'] = 0
    GBM_seg_dic['area'] = 0
    GBM_seg_width = []
    GBM_rect_list = []
    for i in range(GBM_n):
        ins = np.zeros(im_shape)
        ins[np.where(GBM_ins == i+1)] = True
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        ins = cv2.morphologyEx(ins, cv2.MORPH_CLOSE, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (GBM_dilation_kernel_size, GBM_dilation_kernel_size))
        ins_dilated = cv2.dilate(ins, kernel, iterations=1)
        ins = ins.astype(np.uint8)
        if np.sum(ins) < 1000:
            continue
        GBM_im_new = np.maximum(GBM_im_new, ins)
        skeleton = skeletonize(ins)
        if np.sum(skeleton) < 500:
            continue

        GBM_contours = cv2.findContours(ins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(GBM_contours[0]) == 2:
            ctLen1 = GBM_contours[0][0].shape[0]
            ctLen2 = GBM_contours[0][1].shape[0]
            if 0.5<ctLen1/ctLen2<2:
                skeleton, skel_len, endpoints = find_longest_path_ring(skeleton)
            else:
                skeleton, skel_len, endpoints = find_longest_path_band(skeleton)
        else:
            skeleton, skel_len, endpoints = find_longest_path_band(skeleton)

        windowsize = 100
        recwidth = 100
        startpoint = endpoints[0]
        seg_width, rect_list = seg_width_cal(skeleton, GBM_im_new, startpoint, windowsize, recwidth)
        GBM_rect_list.append(rect_list)
        GBM_seg_width = GBM_seg_width+seg_width
        skeleton_img = np.maximum(skeleton_img, skeleton.astype(np.uint8))
        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length'] + skel_len
        GBM_seg_dic['area'] = GBM_seg_dic['area'] + np.sum(ins)
        # GBM_seg_dic['GBM_area'] = len(mask_pos[0])
        # GBM_seg_dic['GBM_mean_width'] = len(mask_pos[0])/len(pos[0])
        if GBM_seg_dic['skeleton_length']:
            GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['area']/GBM_seg_dic['skeleton_length']
        else:
            GBM_seg_dic['GBM_mean_width'] = 0

    return GBM_im_new, skeleton_img, GBM_seg_dic , GBM_seg_width, GBM_rect_list

def save_mask_image(outfile, GBM_img, FP_img, skeleton_img, FP_width_img):
    im_shape = GBM_img.shape[0:2]

    GBM_img = np.int64(GBM_img > 0)
    GBM_img_c = np.zeros((im_shape[0], im_shape[1], 3))
    for i in range(3):
        GBM_img_c[:, :, i] = GBM_img * cl2[i]
    FP_img = np.int64(FP_img > 0)
    FP_img_c = np.zeros((im_shape[0], im_shape[1], 3))
    for i in range(3):
        FP_img_c[:, :, i] = FP_img * cl1[i]

    show_im = GBM_img_c+FP_img_c

    GBM_ske_index = np.where(skeleton_img==1)
    show_im[GBM_ske_index[0], GBM_ske_index[1], 0] = cl4[0]
    show_im[GBM_ske_index[0], GBM_ske_index[1], 1] = cl4[1]
    show_im[GBM_ske_index[0], GBM_ske_index[1], 2] = cl4[2]

    FP_index = np.where(FP_width_img != 0)
    show_im[FP_index[0], FP_index[1], 0] = cl3[0]
    show_im[FP_index[0], FP_index[1], 1] = cl3[1]
    show_im[FP_index[0], FP_index[1], 2] = cl3[2]

    show_im = show_im.astype(np.int64)

    cv2.imwrite(outfile, show_im)
# GBM_show(outfile, GBM_im_new, FP_im, skeleton_img, FP_width_img)
def save_maskon_image(image, GBM_img, FP_img, skeleton_img, FP_width_img, outfile):
    im_shape = GBM_img.shape[0:2]

    GBM_img = np.int64(GBM_img > 0)
    GBM_img_c = np.zeros((im_shape[0], im_shape[1], 3))
    for i in range(3):
        GBM_img_c[:, :, i] = GBM_img * cl1[i]
    FP_img = np.int64(FP_img > 0)
    FP_img_c = np.zeros((im_shape[0], im_shape[1], 3))
    for i in range(3):
        FP_img_c[:, :, i] = FP_img * cl2[i]

    colored_mask = GBM_img_c + FP_img_c

    GBM_ske_index = np.where(skeleton_img == 1)
    colored_mask[GBM_ske_index[0], GBM_ske_index[1], 0] = cl4[0]
    colored_mask[GBM_ske_index[0], GBM_ske_index[1], 1] = cl4[1]
    colored_mask[GBM_ske_index[0], GBM_ske_index[1], 2] = cl4[2]

    FP_index = np.where(FP_width_img != 0)
    colored_mask[FP_index[0], FP_index[1], 0] = cl3[0]
    colored_mask[FP_index[0], FP_index[1], 1] = cl3[1]
    colored_mask[FP_index[0], FP_index[1], 2] = cl3[2]

    colored_mask = colored_mask.astype(np.int64)

    image_rgb = np.stack([image] * 3, axis=-1)
    alpha = 0.5
    overlayed_image = (image_rgb * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    Image.fromarray(overlayed_image).save(outfile)

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
def FP_measures_cal(GBM_ins, FP_ins, pixelsize):

    im_shape = FP_ins.shape[0:2]

    SDheight = round(min(int(MINGBMWIDTH / pixelsize / 1.5), min(im_shape) / 20))
    if SDheight <= 3:
        SDheight = 2

    GBM_in = np.zeros(im_shape)
    n_GBM = np.max(GBM_ins)
    if n_GBM == 0:
        print("There is no GBM in this image!")
        return 1
    FP_seg_width = []
    FP_seg_dic = {}
    FP_seg_dic['FP_num'] = 0
    FP_seg_dic['FP_mean_width'] = 0
    FP_width_img = np.zeros(im_shape)
    FP_widths = []
    for i in range(n_GBM):
        GBM_in = np.zeros(im_shape)
        GBM_in[np.where(GBM_ins == i + 1)] = True
        GBM_in = GBM_in.astype(np.uint8)
        if np.sum(GBM_in) < (MINGBMWIDTH/pixelsize)**2*10:
            continue
        skeleton = skeletonize(GBM_in)
        if np.sum(skeleton) < MINGBMWIDTH/pixelsize*10:
            continue

        kernel = np.ones((SDheight,SDheight),dtype=np.uint8)
        GBM_in_dilated = cv2.dilate(GBM_in, kernel,1) # dilate the GBM to calculate the FP
        GBM_in_dilated = GBM_in_dilated.astype(np.uint8)
        GBM_contour_dilated, h = cv2.findContours(GBM_in_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        GBM_contour_dilated_list = [np.squeeze(arr) for arr in GBM_contour_dilated]
        GBM_contour_dilated_array = np.concatenate(GBM_contour_dilated_list, axis=0)
        # GBM_contour_dilated = np.squeeze(GBM_contour_dilated[0]) # GBM contour after dilation
        GBM_contour_dilated_img = np.zeros(im_shape).astype(np.uint8)
        GBM_contour_dilated_img[GBM_contour_dilated_array[:,1], GBM_contour_dilated_array[:,0]] = 1 # image of the GBM contour after dilation
        FP_width_pos = GBM_contour_dilated_img*FP_ins # width image of each FP along the i th GBM
        FP_width_img = np.maximum(FP_width_img, FP_width_pos)

        FP_seg_ids = np.unique(FP_width_pos) # the ids of the FP along the i th GBM
        FP_seg_ids = FP_seg_ids.astype(np.int64)
        FP_seg_ids = FP_seg_ids[FP_seg_ids != 0]
        FP_seg_dic['FP_num'] = FP_seg_dic['FP_num'] + len(FP_seg_ids)
        for j in range(len(FP_seg_ids)):
            # FP_seg_id = FP_seg_ids[j]-1 #
            # FP_seg_dic = {}  #
            # FP_seg_dic['GBM_id'] = GBM_ins[i]['id']
            # FP_seg_dic['FP_id'] = FP_seg_id
            FP_seg_pos = np.where(FP_width_pos == FP_seg_ids[j])
            FP_seg_dic['FP_mean_width'] = FP_seg_dic['FP_mean_width'] + len(FP_seg_pos[0])
            FP_widths.append(pixelsize * len(FP_seg_pos[0]))

            # FP_seg_dic['FP_width_point'] = FP_seg_pos # width line points of the j th FP along the i th GBM
            # FP_seg_dic['FP_width'] = len(FP_seg_pos[0]) # width value of the j th FP along the i th GBM
            # FP_area = np.where(FP_ins == FP_seg_id+1)
            # FP_seg_dic['FP_area'] = len(FP_area[0])
            # FP_width_list.append(FP_seg_dic)
    if FP_seg_dic['FP_num'] == 0:
        FP_seg_dic['FP_mean_width'] = 0
    else:
        FP_seg_dic['FP_mean_width'] = pixelsize* FP_seg_dic['FP_mean_width']/FP_seg_dic['FP_num']

    return FP_width_img, FP_seg_dic, FP_widths
    # return show_im, FP_width_list



def main():



    # imagefilepath = '../data/mmseg_data/Rat/images_high/iter1/151G188_RT15-019_5k-10K_images_BOX_052716/151G188_RT15019_Group_001-006_9' ## input
    # segfilepath = '../data/mmseg_data/Rat/images_high/iter1/151G188_mask_GBMaFP_proc/151G188_RT15019_Group_001-006/pred_psp_9' ## output
    # outCSV = '../data/mmseg_data/Rat/images_high/iter1/151G188_mask_GBMaFP_proc/151G188_RT15019_Group_001-006/width_151G188_001-006_psp2.csv'
    # outSeg = '../data/mmseg_data/Rat/images_high/iter1/151G188_mask_GBMaFP_proc/151G188_RT15019_Group_001-006/figs_psp2'
    # outFPjson = '../data/mmseg_data/Rat/images_high/iter1/151G188_mask_GBMaFP_proc/151G188_RT15019_Group_001-006/FPwidths_psp2.json'

    # imagefilepath = '../data/mmseg_data/Rat/images_high/iter1/15IG048_RT-15-008_5k_high_mag_Rat_1-6_13-18_25-30_37-42/Rat_1-6'  ## input
    # segfilepath =  '../data/mmseg_data/Rat/images_high/iter1/15IG048_mask_GBMaFP_proc/Rat_1-6/pred_psp_9'  ## output
    # outCSV = '../data/mmseg_data/Rat/images_high/iter1/15IG048_mask_GBMaFP_proc/Rat_1-6/width_151G048_001-006_psp2.csv'
    # outSeg = '../data/mmseg_data/Rat/images_high/iter1/15IG048_mask_GBMaFP_proc/Rat_1-6/figs_psp2'
    # outFPjson = '../data/mmseg_data/Rat/images_high/iter1/15IG048_mask_GBMaFP_proc/Rat_1-6/FPwidths_psp2.json'

    # imagefilepath = '../data/mmseg_data/Rat/images_high/iter1/15IG048_RT-15-008_5k_high_mag_Rat_1-6_13-18_25-30_37-42/Rat_1-6'  ## input
    # segfilepath = '../data/mmseg_data/Rat/images_high/iter1/15IG048_mask_GBMaFP_proc/Rat_1-6/pred_hr'  ## output
    # outCSV = '../data/mmseg_data/Rat/images_high/iter1/15IG048_mask_GBMaFP_proc/Rat_1-6/width_151G048_001-006_hr.csv'
    # outSeg = '../data/mmseg_data/Rat/images_high/iter1/15IG048_mask_GBMaFP_proc/Rat_1-6/figs_hr'
    # outFPjson = '../data/mmseg_data/Rat/images_high/iter1/15IG048_mask_GBMaFP_proc/Rat_1-6/FPwidths_hr.json'

    # pixelsize = 3.891  ## nm
    #
    # normal_pixel = 200 / pixelsize
    #
    # GBMkernelsize = min(int(normal_pixel / 5), 30)
    # # GBMkernelsize = 10
    #
    # if GBMkernelsize <= 3:
    #     GBMkernelsize = 2
    #     print('The image resulution is too low, accuracy may be unaccuracy!')
    # FPkernelsize = round(2.5 * GBMkernelsize)
    # # FPkernelsize = 20

    imagefilepath = '../data/mmseg_data/evaluation/evaluationdata20240513/Wildtype'  ## input
    segfilepath = '../data/mmseg_data/evaluation/evaluationdata20240513/Wildtype_pred_psp'  ## output
    outCSV = '../data/mmseg_data/evaluation/evaluationdata20240513/width_Wildtype_psp_com.csv'
    outSeg = '../data/mmseg_data/evaluation/evaluationdata20240513/Wildtype_fig_psp_com'
    outFPjson = '../data/mmseg_data/evaluation/evaluationdata20240513/Wildtype_FPwidths_psp_com.json'

    pixelsize = 4.122  ## nm

    normal_pixel = 200 / pixelsize

    # for WT
    GBMkernelsize = min(int(normal_pixel / 15), 30)

    # for ILK
    # GBMkernelsize = min(int(normal_pixel / 10), 30)

    if GBMkernelsize <= 3:
        GBMkernelsize = 2
        print('The image resulution is too low, accuracy may be unaccuracy!')
    # for WT
    FPkernelsize = round(5 * GBMkernelsize)
    # for ILK
    # FPkernelsize = round(7 * GBMkernelsize)

    if not os.path.isdir(outSeg):
        os.mkdir(outSeg)

    filelist = [file for file in os.listdir(segfilepath) if file.endswith(('tif','png'))]
    width_info_list = []
    FP_widths_json = []
    GBM_widths_json = []
    for file in tqdm(filelist, desc='file'):
        image_id = file.split('.')[0]
        imagefile = os.path.join(imagefilepath,image_id+'.jpg')
        segfilename = os.path.join(segfilepath, file)
        segimage = cv2.imread(segfilename, cv2.IMREAD_COLOR)
        GBM_im = np.zeros(segimage.shape[0:2])
        GBM_im[(segimage[:, :, 2] > 0) & (segimage[:, :, 1] == 0) & (segimage[:, :, 0] == 0)] = True
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        GBM_im = cv2.morphologyEx(GBM_im, cv2.MORPH_CLOSE, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        GBM_im = cv2.morphologyEx(GBM_im, cv2.MORPH_OPEN, kernel, iterations=1)
        GBM_ins = measure.label(GBM_im, connectivity=2)

        FP_im = np.zeros(segimage.shape[0:2])
        FP_im[(segimage[:, :, 0] > 0) & (segimage[:, :, 1] == 0) & (segimage[:, :, 2] == 0)] = True
        # Erode the FP to separate some close FPs, should be dilated later
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (FPkernelsize, FPkernelsize))
        FP_im = cv2.morphologyEx(FP_im, cv2.MORPH_OPEN, kernel, iterations=1)
        FP_ins = measure.label(FP_im, connectivity=1)

        im_dict = {"image_id": file.split('.')[0]}

        # try:
        GBM_im_new, skeleton_img, GBM_info = GBM_measures_cal(GBM_ins, pixelsize)
        # GBM_im_new, skeleton_img, GBM_info, GBM_seg_width, GBM_rect_list = GBM_seg_measures_cal(GBM_ins)
        FP_width_img, FP_info, FP_widths = FP_measures_cal(GBM_ins, FP_ins, pixelsize)
        width_info_list.append({**im_dict, **GBM_info, **FP_info})
        # outRectfile = os.path.join(outRect, file.split('.')[0]+'.png')
        outSegfile = os.path.join(outSeg, file.split('.')[0] + '.png')

        FP_widths_info = {}
        FP_widths_info["image_id"] =image_id
        FP_widths_info["FP_widths"] = FP_widths
        FP_widths_json.append(FP_widths_info)

        GBM_widths_info = {}
        GBM_widths_info["image_id"] = image_id
        # GBM_widths_info["GBM_widths"] = GBM_seg_width
        GBM_widths_json.append(GBM_widths_info)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        FP_width_img = cv2.morphologyEx(FP_width_img, cv2.MORPH_DILATE, kernel, iterations=1)
        skeleton_img = cv2.morphologyEx(skeleton_img, cv2.MORPH_DILATE, kernel, iterations=1)

        image = Image.open(imagefile).convert('L')
        save_maskon_image(image, GBM_im_new, FP_im, skeleton_img, FP_width_img, outSegfile)

    with open(outFPjson, 'w') as file:
        json.dump(FP_widths_json, file, indent=4)

    df = pd.DataFrame(width_info_list)
    df.to_csv(outCSV, index=False)

if __name__ == '__main__':
    main()
