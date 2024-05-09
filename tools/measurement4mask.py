## calculate the GBM mean width from the DL model prediction mask image
## the mean with is calculated by area / skeleton length
## the skeleton length is pruned by graph algorith, dijkstra for example
## can process the skeleton extraction with rings
## calculate the FP mean width from the DL model prediction mask image

import json
import os
import pandas as pd
from skimage.morphology import skeletonize, medial_axis
from skimage import measure, morphology
#from tqdm import tqdm
import networkx as nx
from PIL import Image

GBM_closing_kernel_size = 50
FP_openning_kernel_size = 15

cl1 = (255, 0, 0) # color for FP
cl2 = (0, 0, 255) # color for GBM
cl3 = (255, 255, 255)     # color for FP width
cl4 = (255, 255, 255)     # color for GBM skeleton
cl5 = (0, 0, 255)     # color for GBM width

GBM_closing_kernel_size = 10
GBM_opening_kernel_size = 10
GBM_dilation_kernel_size = 20 ## decided the distance from GBM to calculate the FP width
# GBM = 1, FP = 2

import cv2
import numpy as np

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

    def get_neighbors(point, directions, im_height, im_width):
        x, y = point
        neighbors = []
        for dx, dy in directions:
            if x + dx < im_height and y + dy < im_width:
                if skel[x + dx, y + dy]:
                    neighbors.append((x + dx, y + dy))
        return neighbors

    def track_paths(vertices, start, second, neighbors1):

        path = []
        # neighbors = get_neighbors(start, directions1, im_height, im_width)
        visited = {start}
        for neighbor in neighbors1:
            visited.add(neighbor)

        path.append(start)
        path.append(second)
        endpoint = second
        while path:
            x, y = path[-1]
            visited.add((x, y))
            neighbors = get_neighbors((x, y), directions1, im_height, im_width)
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
        neighbors1 = get_neighbors(bif_point, directions1, im_height, im_width)
        # neighbors2 = get_neighbors(bif_point, directions2, im_height, im_width)
        for neighbor in neighbors1:
            path, endpoint = track_paths(vertices, bif_point, neighbor, neighbors1)

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

    newskel = []
    for i in range(len(pathV)-1):
        newskel = newskel + pathdict[(pathV[i], pathV[i+1])]
    newskelimage = np.zeros(skel.shape)
    rows, cols = zip(*newskel)
    newskelimage[rows, cols] = True

    return newskelimage, max_distance

def find_longest_path_ring(skel):
    directions1 = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
    im_height, im_width = skel.shape[0:2]

    #

    def get_neighbors(point, directions, im_height, im_width):
        x, y = point
        neighbors = []
        for dx, dy in directions:
            if x + dx < im_height and y + dy < im_width:
                if skel[x + dx, y + dy]:
                    neighbors.append((x + dx, y + dy))
        return neighbors

    def track_paths(bifurcation, start):
        # track_paths(skel, vertices, neighbor, neighbors1, neighbors2)

        path = []
        # neighbors = get_neighbors(start, directions1, im_height, im_width)
        visited = {start}
        path.append(start)
        while path:
            x, y = path[-1]
            visited.add((x, y))
            neighbors = get_neighbors((x, y), directions1, im_height, im_width)
            n_visited = 0
            for neighbor in neighbors:
                if neighbor in visited:
                    n_visited = n_visited+1
                if neighbor not in visited:
                    if neighbor in bifurcation:
                        return path
                    else:
                        path.append(neighbor)
            if n_visited == len(neighbors):
                return path

    endpoints = find_endpoints(skel)
    bifurcation_points = find_bifurcation_points(skel)

    skeleton_bool = skel.astype(bool)
    endpoint_coords = list(zip(*np.where(endpoints)))
    bifurcation_coords = list(zip(*np.where(bifurcation_points)))
    # skel_coords = list(zip(*np.where(skeleton_bool == 1)))

    for endpoint in endpoint_coords:
        branch = track_paths(bifurcation_coords, endpoint)
        row_indices, col_indices = zip(*branch)
        skeleton_bool[row_indices, col_indices] = False

    skel_len = np.sum(skeleton_bool)



    return skeleton_bool, skel_len
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
    elif len(bifurcation_coords) == 0:
        skeleton = skeleton_bool
        skel_len = np.sum(skeleton)
    else:
        skeleton, skel_len = find_longest_path(skeleton_bool, bifurcation_coords, endpoint_coords)
    return skeleton, skel_len

def GBM_measures_cal(GBM_ins, pixelsize, normal_pixel):
    # GBM_id = 0
    im_shape = GBM_ins.shape[0:2]

    GBM_ins = measure.label(GBM_ins, connectivity=2)
    GBM_n = np.max(GBM_ins)
    if GBM_n == 0:
        #print("There is no GBM in this image!")
        #return 1
        raise Exception("There is no GBM in this image!")

    ## calculate the GBM width
    skeleton_img = np.zeros(im_shape,dtype=np.uint8)
    GBM_im_new = np.zeros(im_shape,dtype=np.uint8)
    GBM_seg_dic = {}
    GBM_seg_dic['skeleton_length'] = 0
    GBM_seg_dic['area'] = 0

    print("GBM_n is: " + str(GBM_n))
    for i in range(GBM_n):
        ins = np.zeros(im_shape)
        ins[np.where(GBM_ins == i+1)] = True
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        ins = cv2.morphologyEx(ins, cv2.MORPH_CLOSE, kernel, iterations=1)

        ins = ins.astype(np.uint8)
        if np.sum(ins) < normal_pixel*12.5:
            # print(i)
            continue
        GBM_im_new = np.maximum(GBM_im_new, ins)
        skeleton = skeletonize(ins)
        if np.sum(skeleton) < normal_pixel*6:
            # print(i)
            continue

        GBM_contours = cv2.findContours(ins, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(GBM_contours[0]) == 2:
            ctLen1 = GBM_contours[0][0].shape[0]
            ctLen2 = GBM_contours[0][1].shape[0]
            if 0.5<ctLen1/ctLen2<2:
                skeleton, skel_len = find_longest_path_ring(skeleton)
            else:
                skeleton, skel_len = find_longest_path_band(skeleton)
        else:
            skeleton, skel_len = find_longest_path_band(skeleton)

        skeleton_img = np.maximum(skeleton_img, skeleton.astype(np.uint8))
        GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length'] + skel_len
        GBM_seg_dic['area'] = GBM_seg_dic['area'] + np.sum(ins)
    if GBM_seg_dic['skeleton_length']:
        GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['area']/GBM_seg_dic['skeleton_length']
    else:
        GBM_seg_dic['GBM_mean_width'] = 0
    GBM_seg_dic['skeleton_length'] = GBM_seg_dic['skeleton_length'] * pixelsize
    GBM_seg_dic['GBM_mean_width'] = GBM_seg_dic['GBM_mean_width'] * pixelsize
    GBM_seg_dic['area'] = GBM_seg_dic['area'] * pixelsize * pixelsize

    return GBM_im_new, skeleton_img, GBM_seg_dic

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
    #Image.fromarray(overlayed_image).save(outfile)
    return Image.fromarray(overlayed_image)

def FP_measures_cal(GBM_ins, FP_ins, pixelsize, normal_pixel):

    im_shape = FP_ins.shape[0:2]
    SDheight = round(min(int(normal_pixel / 2), min(im_shape)/20))
    if SDheight <= 3:
        SDheight = 2
    GBM_in = np.zeros(im_shape)
    n_GBM = np.max(GBM_ins)
    if n_GBM == 0:
        #print("There is no GBM in this image!")
        #return 1
        raise Exception("There is no GBM in this image!")
    FP_seg_dic = {}
    FP_seg_dic['FP_num'] = 0
    FP_seg_dic['FP_mean_width'] = 0
    FP_width_img = np.zeros(im_shape)
    FP_widths = []
    for i in range(n_GBM):
        GBM_in[np.where(GBM_ins == i + 1)] = True
        GBM_in[np.where(GBM_ins != i + 1)] = False
        if np.sum(GBM_in) < normal_pixel*12.5:
            continue
        skeleton = skeletonize(GBM_in)
        if np.sum(skeleton) < normal_pixel*6:
            continue

        kernel = np.ones((SDheight,SDheight),dtype=np.uint8)
        GBM_in_dilated = cv2.dilate(GBM_in, kernel,1) # dilate the GBM to calculate the FP
        GBM_in_dilated = GBM_in_dilated.astype(np.uint8)
        GBM_contour_dilated, h = cv2.findContours(GBM_in_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        GBM_contour_dilated = np.squeeze(GBM_contour_dilated[0]) # GBM contour after dilation
        GBM_contour_dilated_img = np.zeros(im_shape).astype(np.uint8)
        GBM_contour_dilated_img[GBM_contour_dilated[:,1], GBM_contour_dilated[:,0]] = 1 # image of the GBM contour after dilation
        FP_width_pos = GBM_contour_dilated_img*FP_ins # width image of each FP along the i th GBM
        FP_width_img = np.maximum(FP_width_img, FP_width_pos)

        FP_seg_ids = np.unique(FP_width_pos) # the ids of the FP along the i th GBM
        FP_seg_ids = FP_seg_ids.astype(np.int64)
        FP_seg_ids = FP_seg_ids[FP_seg_ids != 0]
        FP_seg_dic['FP_num'] = FP_seg_dic['FP_num'] + len(FP_seg_ids)
        for j in range(len(FP_seg_ids)):

            FP_seg_pos = np.where(FP_width_pos == FP_seg_ids[j])
            FP_seg_dic['FP_mean_width'] = FP_seg_dic['FP_mean_width'] + len(FP_seg_pos[0])
            FP_widths.append(pixelsize*len(FP_seg_pos[0]))


    if FP_seg_dic['FP_num'] == 0:
        FP_seg_dic['FP_mean_width'] = 0
    else:
        FP_seg_dic['FP_mean_width'] = pixelsize * FP_seg_dic['FP_mean_width']/FP_seg_dic['FP_num']

    return FP_width_img, FP_seg_dic, FP_widths
    # return show_im, FP_width_list



def main(inputimage, inputseg, inputfilepath, pixelsize):
    message = ""

    # The default pixel size should be 4.122 (nm); this is not enforced here; rather,
    # the web client's form's "pixel size" field is set to be a required field and to have this default value.
    # pixelsize = 4.122 ## nm
    # pixelsize = 4.122*3
    # pixelsize = 4.122/3  ## nm

    # TODO: Ask Anqi to confirm pixel size logic; some numbers trigger both the "too high" and "too low" warnings....

    normal_pixel = 320/pixelsize

    GBMkernelsize = min(int(normal_pixel/8.5), 20)

    if GBMkernelsize<=3:
        GBMkernelsize = 2
        message += "The image resolution is too low, and the measurement result may be inaccurate! "
    FPkernelsize = round(1.5* GBMkernelsize)

    width_info_list = []
    FP_widths_json = []

    # Original: Reads from file:
    #segimage = cv2.imread(segfilename, cv2.IMREAD_COLOR)
    # Modification to take PIL Image as input: This raises an Assertion error
    #segimage = cv2.imdecode(np.asarray(inputseg), cv2.IMREAD_COLOR)
    # Therefore instead of cv2, just read as np array:
    segimage = np.asarray(inputseg)
    if normal_pixel >= min(segimage.shape[0], segimage.shape[1])/3:
        message += "The image resolution is too high, and the measurement result may be inaccurate! "

    # Note: CV2 reads images to ABGR order (why?!) whereas np.asarray will read to RGB(A).
    # So, in the following code, I have flipped the minor axis indices in Anqi's conditionals.

    GBM_im = np.zeros(segimage.shape[0:2])
    #GBM_im[(segimage[:, :, 2] > 0) & (segimage[:, :, 1] == 0) & (segimage[:, :, 0] == 0)] = True # BGR
    GBM_im[(segimage[:, :, 0] > 0) & (segimage[:, :, 1] == 0) & (segimage[:, :, 2] == 0)] = True # RGB
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (GBMkernelsize, GBMkernelsize))
    GBM_im = cv2.morphologyEx(GBM_im, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (GBMkernelsize, GBMkernelsize))
    GBM_im = cv2.morphologyEx(GBM_im, cv2.MORPH_OPEN, kernel, iterations=1)
    GBM_ins = measure.label(GBM_im, connectivity=2)

    FP_im = np.zeros(segimage.shape[0:2])
    #FP_im[(segimage[:, :, 0] > 0) & (segimage[:, :, 1] == 0) & (segimage[:, :, 2] == 0)] = True # BGR
    FP_im[(segimage[:, :, 2] > 0) & (segimage[:, :, 1] == 0) & (segimage[:, :, 0] == 0)] = True # RGB
    # open the FP to separate some close FPs, should be dilated later
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (FPkernelsize, FPkernelsize))
    FP_im = cv2.morphologyEx(FP_im, cv2.MORPH_OPEN, kernel, iterations=1)
    FP_ins = measure.label(FP_im, connectivity=1)

    image_id = inputfilepath.split('/')[-1].split('.')[0]
    im_dict = {"image_id": image_id}

    try:
        GBM_im_new, skeleton_img, GBM_info = GBM_measures_cal(GBM_ins, pixelsize, normal_pixel)
        FP_width_img, FP_info, FP_widths = FP_measures_cal(GBM_ins, FP_ins, pixelsize, normal_pixel)
        success = True
        message = message or "OK."
    except Exception as e:
        # If x_measures_cal throws exception (e.g. no GBM in this image), forward that message through...
        success = False
        message += str(e)
        return success, message, None, None, None

    width_info_list.append({**im_dict, **GBM_info, **FP_info})
    #outfile = os.path.join(outfig, file.split('.')[0]+'.png')

    FP_widths_info = {}
    FP_widths_info["image_id"] = image_id
    FP_widths_info["FP_widths"] = FP_widths
    FP_widths_json.append(FP_widths_info)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    FP_width_img = cv2.morphologyEx(FP_width_img, cv2.MORPH_DILATE, kernel, iterations=1)
    skeleton_img = cv2.morphologyEx(skeleton_img, cv2.MORPH_DILATE, kernel, iterations=1)
    image = Image.open(inputimage).convert('L')
    measurement_mask_img = save_maskon_image(image, GBM_im_new, FP_im, skeleton_img, FP_width_img, "outfile-unused")

    #with open(outjson, 'w') as file:
    #    json.dump(FP_widths_json, file, indent=4)

    #df = pd.DataFrame(width_info_list)
    #df.to_csv(outCSV, index=False)

    return success, message, measurement_mask_img, FP_widths_json, width_info_list

