import cv2 as cv
from pyzbar.pyzbar import decode
import pandas as pd
import numpy as np
from sympy import Symbol, solve, diff

# from .util import anti_clockwise, perspective_transformation, canonical, vector_norm, vector_cross, r_dist, poly_transformation
from util import anti_clockwise, perspective_transformation, canonical, vector_norm, vector_cross, r_dist, poly_transformation



def coord_estimation(object_dict):
    # 연속된 2 frame만 계산
    result_std_dict = {}
    result_sin_dict = {}

    for idx, key in enumerate(object_dict):
        data = object_dict[key]

        real_point = [0, 0, 0]

        x = Symbol('X')
        y = Symbol('Y')
        z = Symbol('Z')
        m = Symbol('M')
        n = Symbol('N')

        mask = []
        est_s = []
        est_d = []


        if idx < 5:
            criteria = data[0][-1]
        else:
        ###
            marker = dict()

            for x in data:
                if x[-1] not in marker.keys():
                    marker[x[-1]] = 1
                else:
                    marker[x[-1]] += 1

            criteria = max(marker, key=marker.get)
        print(key, "'s criteria : ", criteria)

        num = 0

        for i in range(len(data) - 1):
            if data[i + 1][-1] != criteria or num > 3:
                continue
            else:
                num += 1
                p1 = [data[i][0], data[i][1], data[i][2]]
                u1 = [data[i][7], data[i][8], data[i][9]]
                u1_norm = vector_norm(u1)
                t1 = [m * u1[tt] + p1[tt] for tt in range(3)]

                p2 = [data[i + 1][0], data[i + 1][1], data[i + 1][2]]
                u2 = [data[i + 1][7], data[i + 1][8], data[i + 1][9]]
                u2_norm = vector_norm(u2)
                t2 = [n * u2[tt] + p2[tt] for tt in range(3)]

                u1u2 = vector_cross(u1, u2)
                u1u2_norm = vector_norm(u1u2)

                sin = u1u2_norm / (u1_norm * u2_norm)

                r = r_dist(p1, p2, u1, u2)

                p1t2 = [tt - pp for pp, tt in zip(p1, t2)]
                p2t1 = [tt - pp for pp, tt in zip(p2, t1)]

                cross_p1t2u1 = vector_cross(p1t2, u1)
                cross_p2t1u2 = vector_cross(p2t1, u2)

                form1 = vector_norm(cross_p1t2u1) - r * u1_norm
                form2 = vector_norm(cross_p2t1u2) - r * u2_norm

                m_x = solve(diff(form2, m))
                n_x = solve(diff(form1, n))

                n_t1 = [tt.subs(m, m_x[0]) for tt in t1]
                n_t2 = [tt.subs(n, n_x[0]) for tt in t2]
                n_t = [(tt1 + tt2) / 2 for tt1, tt2 in zip(n_t1, n_t2)]

                est_d.append(n_t)
                est_s.append(sin)
                mask.append(1)

        if len(est_d) != 0:
            est_d = np.array(est_d).astype(float)
            est_s = np.array(est_s)
            mask = np.array(mask)

            total_m = np.sum(mask)
            est_s_m = est_s * mask
            est_s_t = total_m * est_s_m / np.sum(est_s_m)
            est_x = est_d[:, 0] * mask
            est_y = est_d[:, 1] * mask
            est_z = est_d[:, 2] * mask

            # ------------------------------------------
            # mean
            est_std_x_result = np.sum(est_x) / total_m
            est_std_y_result = np.sum(est_y) / total_m
            est_std_z_result = abs(np.sum(est_z) / total_m)

            # ------------------------------------------
            # sin, mean
            est_sin_x = est_s_t * est_x
            est_sin_y = est_s_t * est_y
            est_sin_z = est_s_t * est_z

            est_sin_x_result = np.sum(est_sin_x) / total_m
            est_sin_y_result = np.sum(est_sin_y) / total_m
            est_sin_z_result = abs(np.sum(est_sin_z) / total_m)

            result_std_dict[key] = [est_std_x_result, est_std_y_result, est_std_z_result]
            result_sin_dict[key] = [est_sin_x_result, est_sin_y_result, est_sin_z_result]

    return result_std_dict


def alignment(ori_image, seg_image, now_frame, floor_dict):

    pt_image = ori_image

    object_points = {}
    marker_points = []
    dimension_points = []
    poly_points = []
    marker_info = ''

    ret, thresh = cv.threshold(seg_image, 127, 255, 0)

    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    QR_center = []
    QR_point = []
    QR_info = []

    for i, contour in enumerate(contours):
        contour_approx = cv.approxPolyDP(contour, cv.arcLength(contour, True) * 0.02, True)

        area = cv.contourArea(contour)
        center = np.array([0, 0])
        point = []

        for approx in contour_approx:
            center += approx[0]
            point.append(list(approx[0]))

        center = np.int0(center / 4)

        cents = np.append(center, area)

        if len(point) == 4:
            QR_center.append(list(center))
            QR_point.append(point)
            QR_info.append(cents)

    decoded_num = 0

    tmp = 0

    for idx, point in enumerate(QR_point):

        point_anti_clockwise = anti_clockwise(point)

        pt_result, warp_matrix, transformed = perspective_transformation(ori_image, point_anti_clockwise, now_frame, idx)
        tmp += 1

        decoded = decode(pt_result)

        coord_2d = point_anti_clockwise

        if decoded:

            qr_data = str(decoded[0].data).split("'")[1]

            decoded_num += 1

            ###
            poly_2d = []
            polygon = canonical(decoded[0].polygon)

            if transformed:
                inv_warp = np.linalg.inv(warp_matrix)

                for j in polygon:
                    perspective_marker = np.append(np.array(j), np.array([1.])) # 3, 1
                    homogeneous = inv_warp @ perspective_marker
                    homogeneous = [int(homogeneous[0] / homogeneous[2]), int(homogeneous[1] / homogeneous[2])]
                    poly_2d.append(homogeneous)

            else:
                for j in polygon:
                    poly_2d.append([j[0], j[1]])

            if qr_data in floor_dict.keys():

                if len(marker_points) == 0:
                    center_point = floor_dict[qr_data]
                    center_x = center_point[0]
                    center_y = center_point[1]
                    center_z = center_point[2]

                    coord_3d = [[center_x + 0.1, center_y + 0.1, center_z], [center_x - 0.1, center_y + 0.1, center_z],
                                [center_x - 0.1, center_y - 0.1, center_z], [center_x + 0.1, center_y - 0.1, center_z]]

                    dimension_point = coord_3d

                    coord_center = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
                    dimension_point = [[dimension_point[j][k] + coord_center[j][k] for k in range(3)] for j in range(4)]
                    dimension_points.append(dimension_point)
                    marker_points.append(coord_2d)
                    poly_points.append(poly_transformation(poly_2d))
                    marker_info = qr_data

                else:
                    continue

            # elif qr_data in object_list:
            else:
                object_points[str(decoded[0].data).split("'")[1]] = np.array(poly_2d)

    return object_points, np.array(marker_points), np.array(dimension_points), np.array(poly_points), pt_image, marker_info
