import argparse
import os
import sys

import cv2 as cv
import numpy as np


def binarization(img):
    threshold = np.average(img)
    __, binary_img = cv.threshold(img, threshold, 0xff, cv.THRESH_BINARY)
    return binary_img


median_filter_ksize = 7


def median_filter(img):
    median_filtered_img = cv.medianBlur(img, median_filter_ksize)
    return median_filtered_img


def keep_largest_area(img):
    label_cnt, labels, stats, __ = cv.connectedComponentsWithStats(img)
    label_kept = max([(i, stats[i][cv.CC_STAT_AREA])
                      for i in range(1, label_cnt)], key=lambda t: t[1])[0]
    has_label = np.zeros(label_cnt, dtype=np.bool)
    has_label[label_kept] = True
    return img * has_label[labels]


morphological_opening_kernel = np.ones((20, 20), dtype=np.uint8)


def morphological_open(img):
    return cv.morphologyEx(img, cv.MORPH_OPEN, morphological_opening_kernel)


morphological_closing_kernel = np.ones((25, 25), dtype=np.uint8)


def morphological_close(img):
    return cv.morphologyEx(img, cv.MORPH_CLOSE, morphological_closing_kernel)


def fit(img):
    h, w = img.shape
    binary_img = img//0xff
    one_cnt = np.sum(binary_img, axis=0, dtype=np.float64)
    chosen_col = one_cnt != 0
    x = np.arange(w)[chosen_col]
    y = np.sum(
        np.tile(np.arange(h, dtype=np.float64).reshape((h, 1)), w) * binary_img, axis=0, dtype=np.float64)[chosen_col] / one_cnt[chosen_col]
    coeff = np.polyfit(x, y, 2)
    return np.poly1d(coeff)


def tailor(img, binary_img, f):
    h, w = img.shape
    curve = f(np.arange(w))
    max_val = np.max(curve)
    x = np.tile(np.arange(w), (h, 1))
    y = np.clip((
        np.tile(np.arange(h).reshape((h, 1)), w) + np.tile(curve, (h, 1)) - max_val).astype(np.intc), 0, h-1)
    return img[y, x], binary_img[y, x]


def crop(img, x, y, w, h):
    return img[y:(y+h), x:(x+w)]


def process_image(src_file_name, dst_file_name, src_path, dst_path):
    orig_img = cv.imread(os.path.join(
        src_path, src_file_name), cv.IMREAD_UNCHANGED)
    # denoised_img = cv.xphoto.bm3dDenoising(orig_img)
    binary_img = binarization(orig_img)
    median_filtered_img = median_filter(binary_img)
    largest_area_img = keep_largest_area(median_filtered_img)
    morphology_img = morphological_open(
        morphological_close(largest_area_img))
    f = fit(morphology_img)
    tailored_img, tailored_binary_img = tailor(orig_img, morphology_img, f)
    x, y, w, h = cv.boundingRect(tailored_binary_img)
    cropped_img = crop(tailored_img, x, y, w, h)
    cv.imwrite(os.path.join(dst_path, dst_file_name), cropped_img)


def traverse_dataset(src_root_path, dst_root_path):
    img_cnt = 0
    for src_sub_dir_name in os.listdir(src_root_path):
        src_sub_dir = os.path.join(src_root_path, src_sub_dir_name)
        for src_sub_sub_dir_name in os.listdir(src_sub_dir):
            src_sub_sub_dir = os.path.join(src_sub_dir, src_sub_sub_dir_name)
            for file_name in os.listdir(src_sub_sub_dir):
                dst_dir = os.path.join(
                    dst_root_path, src_sub_dir_name, src_sub_sub_dir_name)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                process_image(file_name, file_name, src_sub_sub_dir, dst_dir)
                img_cnt += 1
                sys.stdout.write(
                    "Processed " + str(img_cnt) + " image(s)...\r")
    print()
    print("Finished processing " + str(img_cnt) + " images.")


parser = argparse.ArgumentParser(
    description="Data preprocess for image classification")
parser.add_argument('--src_root_path', type=str,
                    help='the absolute path of the whole dataset')
parser.add_argument('--dst_root_path', type=str,
                    help='the path to store the preprocessed images')

if __name__ == "__main__":
    args = parser.parse_args()
    src_root_path = args.src_root_path
    dst_root_path = args.dst_root_path
    traverse_dataset(src_root_path, dst_root_path)
