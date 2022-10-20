#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 13:12:04 2022

convert instance labels to three class labels:
0: background
1: interior
2: boundary
@author: jma
"""

import os
from tkinter import image_names

join = os.path.join
import argparse

import numpy as np
import tifffile as tif
from skimage import exposure, io, morphology, segmentation
from tqdm import tqdm
from multiprocessing import Pool


def normalize_channel(img, lower=1, upper=99):
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(
            img, in_range=(percentiles[0], percentiles[1]), out_range="uint8"
        )
    else:
        img_norm = img
    return img_norm.astype(np.uint8)


def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.

    Returns
    -------
    interior : (H,W), np.uint8
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode="inner")
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior


def batch_works(k):
    sub_image_names = [
        image_names[i] for i in range(len(image_names)) if i % num_workers == k
    ]
    sub_image_paths = [join(img_path, image_name) for image_name in sub_image_names]
    sub_label_paths = [
        join(gt_path, image_name.split(".")[0] + "_label.tiff")
        for image_name in sub_image_names
    ]
    for image_path, label_path in zip(sub_image_paths, sub_label_paths):
        img_name = os.path.basename(image_path)
        gt_name = os.path.basename(label_path)
        if image_path.endswith(".tif") or image_path.endswith(".tiff"):
            img_data = tif.imread(image_path)
        else:
            img_data = io.imread(image_path)
        gt_data = tif.imread(label_path)

        # normalize image data
        if len(img_data.shape) == 2:
            img_data = np.repeat(np.expand_dims(img_data, axis=-1), 3, axis=-1)
        elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
            img_data = img_data[:, :, :3]
        else:
            pass
        pre_img_data = np.zeros(img_data.shape, dtype=np.uint8)
        for i in range(3):
            img_channel_i = img_data[:, :, i]
            if len(img_channel_i[np.nonzero(img_channel_i)]) > 0:
                pre_img_data[:, :, i] = normalize_channel(
                    img_channel_i, lower=1, upper=99
                )

        # conver instance bask to three-class mask: interior, boundary
        interior_map = create_interior_map(gt_data.astype(np.int16))

        io.imsave(
            join(target_path, "images", img_name.split(".")[0] + ".png"),
            pre_img_data.astype(np.uint8),
            check_contrast=False,
        )
        io.imsave(
            join(target_path, "labels", gt_name.split(".")[0] + ".png"),
            interior_map.astype(np.uint8),
            check_contrast=False,
        )


def main():
    parser = argparse.ArgumentParser(
        "Preprocessing for microscopy image segmentation", add_help=False
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default="./data/Train_Labeled",
        type=str,
        help="training data path; subfolders: images, labels",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="./data/Train_Pre_3class",
        type=str,
        help="preprocessing data path",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        default=8,
        type=int,
        help="number of workers for multiprocessing",
    )
    args = parser.parse_args()

    source_path = args.input_path
    target_path = args.output_path
    num_workers = args.num_workers

    img_path = join(source_path, "images")
    gt_path = join(source_path, "labels")

    img_names = sorted(os.listdir(img_path))
    gt_names = [img_name.split(".")[0] + "_label.tiff" for img_name in img_names]

    pre_img_path = join(target_path, "images")
    pre_gt_path = join(target_path, "labels")
    os.makedirs(pre_img_path, exist_ok=True)
    os.makedirs(pre_gt_path, exist_ok=True)

    pool = Pool(num_workers)
    pool.map(batch_works, range(num_workers))
    pool.close()


if __name__ == "__main__":
    main()
