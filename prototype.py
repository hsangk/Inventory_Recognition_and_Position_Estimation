import argparse
import cv2 as cv
import os
import csv
import time
from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from .util import segmentation, read_points, solve_PNP, model_setup, camera_calibration, img2vid, save_image
# from . import QR

from util import read_points, solve_PNP, model_setup, camera_calibration, img2vid, save_image
import QR
from segmentation import segment

def get_args_parse():
    parser = argparse.ArgumentParser('Prototype')

    parser.add_argument('--skip', '-s', default=5, type=int)
    parser.add_argument('--save_video', '-v', action='store_true')
    directory = os.path.join(os.path.dirname(__file__), 'test')
    parser.add_argument('--root', '-r', default=directory)

    return parser


def main():
    args = get_args_parse().parse_args([])
    print(args.save_video)
    grounding_dino_model, sam_predictor = model_setup()
    mtx, dist = camera_calibration(args.root)

    video_folder = os.path.join(args.root, 'video')


    save_video = False

    if save_video:
        os.makedirs(os.path.join(args.root, 'decoded_images'), exist_ok=True)
        decoded_images_path = os.path.join(args.root, 'decoded_images')
        print("SAVE VIDEO")
    else:
        decoded_images_path = ''


    # Every file in folder ----------------------------
    video_paths = []
    for file in os.listdir(video_folder):
        if file == 'DJI_20230922163652_0001_W.MP4':
            video_paths.append(os.path.join(video_folder, file))

    for video_path in video_paths:
        print()
        video_name = video_path.split("/")[-1]
        print("***** ", video_name, " *****")
        print()

        os.makedirs(os.path.join(args.root, 'results'), exist_ok=True)
        csv_path = os.path.join(args.root, "results/{}.csv".format(video_name.split(".")[0]))
        print("=== csv_path : ", csv_path, ' ===')
        print()

        results_dict = {}
        coef_x, coef_y, coef_z, c_x, c_y, c_z = 0, 0, 0, 0, 0, 0

        floor_dict = read_points(os.path.join(args.root, "floor_points.txt"))

        cap = cv.VideoCapture(video_path)
        print(video_path)

        start_time = time.time()

        if cap.isOpened():
            total_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            print('total frame:', total_frame)

            with tqdm(total=total_frame, unit='frame') as pbar:
                while True:
                    ret, ori_image = cap.read()
                    now_frame = cap.get(cv.CAP_PROP_POS_FRAMES)

                    if ret:
                        if now_frame % args.skip == 0:
                            seg_image = segment(ori_image, grounding_dino_model, sam_predictor)
                            object_points, marker_points, dimension_points, poly_points, pt_image, marker_info = QR.alignment(
                                ori_image, seg_image, now_frame, floor_dict)
                            results_dict = solve_PNP(object_points, marker_points, dimension_points, poly_points, mtx, dist,
                                                     results_dict, now_frame, marker_info)
                            # if args.save_video:
                            if save_video:
                                save_image(decoded_images_path, pt_image, now_frame, object_points, marker_points)
                    else:
                        break
                    pbar.update(1)
        else:
            print("Can't open video.")
        cap.release()
        cv.destroyAllWindows()

        print("Time : ", time.time() - start_time)

        result_dict = QR.coord_estimation(results_dict)

        # if args.save_video:
        if save_video:
            save_video_path = args.root
            img2vid(img_path=decoded_images_path, save_path=save_video_path)

        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            for k, v in result_dict.items():
                writer.writerow([k] + v)

        print(result_dict)

    return csv_path

main()