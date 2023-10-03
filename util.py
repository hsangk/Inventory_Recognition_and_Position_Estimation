import cv2 as cv
import numpy as np
import torch
import operator
import math
import os
import glob
from functools import reduce
from sympy import Symbol, solve, Matrix, diff
import natsort
from pyzbar.pyzbar import decode

from PIL import Image

# import segmentation_models_pytorch as smp

from GSA.GroundingDINO.groundingdino.util.inference import Model
from GSA.segment_anything.segment_anything import sam_model_registry, SamPredictor
from GSA.EfficientSAM.MobileSAM.setup_mobile_sam import setup_model

def vector_cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def vector_inner(a, b):
    return [a[0] * b[0], a[1] * b[1], a[2] * b[2]]


def vector_norm(a):
    return sum([aa ** 2 for aa in a])


def r_dist(a1, b1, a2, b2):
    cross_u1u2 = vector_cross(a2, b2)
    p1p2 = [bb - aa for aa, bb in zip(a1, b1)]
    inner_pu = vector_inner(p1p2, cross_u1u2)
    return vector_norm(inner_pu) / vector_norm(cross_u1u2)


def poly_transformation(point_list):
    left_top = min(point_list, key=lambda  p: p[0] - p[1])
    left_bottom = min(point_list, key=lambda p: p[0] + p[1])
    right_bottom = max(point_list, key=lambda p: p[0] - p[1])

    for p in point_list:
        if p not in [left_top, left_bottom, right_bottom]:
            right_top = p

    return [right_top, left_top, left_bottom, right_bottom]


def perspective_transformation(image, point_list, now_frame, idx):
    pts_from = np.float32(point_list)

    ###
    x_coords = pts_from[:, 0]
    y_coords = pts_from[:, 1]

    min_x = int(min(x_coords))
    max_x = int(max(x_coords))
    min_y = int(min(y_coords))
    max_y = int(max(y_coords))
    pts_from_img = image[min_y:max_y, min_x:max_x]

    decoded = decode(pts_from_img)

    # if decoded:
    #     warp_matrix = []
    #     result = pts_from_img
    #     transformed = False
    #     # print("not transformed : {}_{}_{}.png".format(now_frame, idx, LR) )
    # ###
    # else:
    width = 400
    height = 400

    pts_to = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

    warp_matrix = cv.getPerspectiveTransform(pts_from, pts_to)
    result = cv.warpPerspective(image, warp_matrix, (width, height))

    transformed = True


    return result, warp_matrix, transformed


def anti_clockwise(coords):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))

    return sorted(coords, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)


def canonical(coords):
    dist = 1000000000
    c = 0
    for i, v in enumerate(coords):
        d = v.x**2 + v.y**2
        if dist > d:
            dist = d
            c = i
    return coords[c:] + coords[:c]


def read_points(path):
    data = [line.rstrip() for line in open(path)]
    dict = {}
    for d in data:
        marker = d.split()[0]
        coord = list(map(float, d.split()[1:]))
        dict[marker] = coord
    return dict


def to_tensor(x):
    return x.transpose(2, 0, 1).astype('float32')


def solve_PNP(object_points, marker_points, dimension_points, poly_points, mtx, dist, results_dict, now_frame, marker_info):

    if len(object_points) and len(marker_points) and len(dimension_points):
        _, rvecs, tvecs = cv.solvePnP(dimension_points[0], poly_points[0].astype(float), mtx, dist)

        rotation_transform_mat = cv.Rodrigues(rvecs)
        camera_pose = np.dot(-np.linalg.inv(rotation_transform_mat[0]), tvecs)
        RT = np.concatenate((rotation_transform_mat[0], tvecs), axis=1)
        KRT = Matrix(np.dot(mtx, RT))

        x = Symbol('X')
        y = Symbol('Y')
        z = Symbol('Z')

        var = Matrix([x, y, z, 1.])
        KRTv = KRT * var

        for i in object_points:
            object_center = object_points[i].mean(axis=0)
            sxy = Matrix(np.array([object_center[0], object_center[1], 1.]))

            KRTv_sxy = KRTv - sxy * KRTv[-1]

            epi_line = solve([KRTv_sxy[0], KRTv_sxy[1]])

            if z not in epi_line.keys():
                coef_x, coef_y, coef_z, c_x, c_y, c_z = \
                    diff(epi_line[x], z), diff(epi_line[y], z), 1, epi_line[x].subs(z, 0), epi_line[y].subs(z, 0), 0

            if i not in results_dict.keys():
                results_dict[i] = np.array([
                    [camera_pose[0][0], camera_pose[1][0], camera_pose[2][0], object_center[0], object_center[1],
                     KRTv_sxy[0], KRTv_sxy[1], coef_x, coef_y, coef_z, c_x, c_y, c_z, now_frame, marker_info]])
            else:
                results_dict[i] = np.append(results_dict[i], np.array([
                    [camera_pose[0][0], camera_pose[1][0], camera_pose[2][0], object_center[0], object_center[1],
                     KRTv_sxy[0], KRTv_sxy[1], coef_x, coef_y, coef_z, c_x, c_y, c_z, now_frame, marker_info]]), axis=0)

    return results_dict

def save_image(decoded_path, pt_image,  now_frame, object_points, marker_points):
    for i in object_points:
        for j in object_points[i]:
            cv.circle(pt_image, tuple(j), 5, (0, 255, 0), -1)  # R

    for i in range(len(marker_points)):
        for j in marker_points[i]:
            cv.circle(pt_image, tuple(j), 5, (0, 255, 0), -1)

    cv.imwrite(decoded_path + '/' + str(now_frame) + '.jpg', pt_image)


def model_setup():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "./GSA/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./checkpoint/groundingdino_swint_ogc.pth"

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./checkpoint/sam_vit_h_4b8939.pth"

    MOBILE_SAM_CHECKPOINT_PATH = "./checkpoint/mobile_sam.pt"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    # Building mobile-SAM
    checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
    mobile_sam= setup_model()
    mobile_sam.load_state_dict(checkpoint, strict=True)
    mobile_sam.to(device=DEVICE)
    sam_predictor = SamPredictor(mobile_sam)

    print('=== Model Setup Completed ===')

    #######
    # return preprocessing_fn, seg_model
    return grounding_dino_model, sam_predictor


def camera_calibration(root_path):

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    directory = os.path.join(root_path, 'checkboard')

    images = glob.glob(os.path.join(directory, '*.*'))

    for fname in images:
        imgs = cv.imread(fname)
        gray = cv.cvtColor(imgs, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (6, 9), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            check = cv.drawChessboardCorners(imgs, (6, 9), corners2, ret)
            # cv.imwrite(os.path.join(root_path, 'check') + '/' + fname.split('/')[-1], check)

    _, mtx, dist, temp_rvecs, temp_tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('=== Camera Calibration Completed ===')

    return mtx, dist

def img2vid(img_path, save_path):
    img_arr = []
    img_file_path = os.path.join(img_path, '*.jpg')
    sorted_img_path = natsort.natsorted(glob.glob(img_file_path))

    for filename in sorted_img_path:
        img = cv.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_arr.append(img)

    out = cv.VideoWriter(save_path + '/result.mp4', cv.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_arr)):
        out.write(img_arr[i])
    out.release()