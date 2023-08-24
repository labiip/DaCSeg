# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:42:13 2021

@author: LXYYYYYYYY
"""
import datetime
import json
import os
import re
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm

ROOT_DIR = "home/xinyu.fan/data"
IMAGE_DIR = "home/xinyu.fan/data/imgb"
ANNOTATION_DIR = os.path.join(ROOT_DIR, "annotations")

INFO = {
    "description": "chrom Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2017,
    "contributor": "xx",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'chromosome',
        'supercategory': 'chromosome',
    }
]


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    coco_output_val = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    coco_output_test = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    count_s = 0

    image_id = 1
    segmentation_id = 1
    image_id_val = 1
    segmentation_id_val = 1
    image_id_test = 1
    segmentation_id_test = 1

    image_files = os.listdir(IMAGE_DIR)

    annotations = np.load("D:/E/data/annotations_with_overlap_else.npy", allow_pickle=True).item()
    for image_filename in tqdm(image_files):
        # go through each image
        count_s += 1
        image_dir = os.path.join(IMAGE_DIR, image_filename)
        image = Image.open(image_dir)

        if count_s > 0 and count_s <= 197:
            image_info_test = pycococreatortools.create_image_info(
                image_id_test, os.path.basename(image_filename), image.size)
            coco_output_test["images"].append(image_info_test)
        elif count_s > 394 and count_s <= 591:
            image_info_val = pycococreatortools.create_image_info(
                image_id_val, os.path.basename(image_filename), image.size)
            coco_output_val["images"].append(image_info_val)
        else:
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

        # if count_s > 800:
        #     image_info_test = pycococreatortools.create_image_info(
        #         image_id_test, os.path.basename(image_filename), image.size)
        #     coco_output_test["images"].append(image_info_test)
        # else:
        #     image_info = pycococreatortools.create_image_info(
        #         image_id, os.path.basename(image_filename), image.size)
        #     coco_output["images"].append(image_info)

        Gband_img = annotations[image_filename.split(".")[0][:-1]]
        hbox = Gband_img['hbox']
        masks = Gband_img['masks']
        overlaps = Gband_img['overlap']
        overlap_elses = Gband_img['overlap_else']

        # filter for associated png annotations
        ww, hh = image.size

        image1 = np.array(image)
        for i in range(len(masks)):

            binary_mask = np.zeros((hh, ww), dtype=np.bool)
            binary_overlap = np.zeros((hh, ww), dtype=np.bool)
            binary_overlap_else = np.zeros((hh, ww), dtype=np.bool)

            category_info = {'id': 1, 'is_crowd': 0}
            temp = masks[i].astype(np.uint8)
            # whether int or numpy,
            # if int , it is no-overlap
            overlap = overlaps[i]
            overlap_e = overlap_elses[i]
            if isinstance(overlap, int):
                overlap = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.bool)
                overlap_e = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.bool)
            else:
                overlap = overlap.astype(np.bool)
                overlap_e = overlap_e.astype(np.bool)
            binary_mask[hbox[i][0]:hbox[i][2], hbox[i][1]:hbox[i][3]] = temp
            # plt.imshow(binary_mask)
            # plt.show()
            binary_overlap[hbox[i][0]:hbox[i][2], hbox[i][1]:hbox[i][3]] = overlap
            # plt.imshow(binary_overlap)
            # plt.show()
            binary_overlap_else[hbox[i][0]:hbox[i][2], hbox[i][1]:hbox[i][3]] = overlap_e
            # plt.imshow(binary_overlap_else)
            # plt.show()

            # if (hbox[i][3]- hbox[i][1])==temp.shape[0]:
            #     binary_mask[] = temp.T
            #     binary_overlap[hbox[i][0]:hbox[i][2], hbox[i][1]:hbox[i][3]] = overlap.T
            # else:
            #     binary_mask[hbox[i][0]:hbox[i][0]+temp.shape[0],hbox[i][1]:hbox[i][1]+temp.shape[1]] = temp
            #     binary_overlap[hbox[i][0]:hbox[i][0] + overlap.shape[0], hbox[i][1]:hbox[i][1] + overlap.shape[1]] = overlap

            if count_s > 0 and count_s <= 197:
                annotation_info_test = pycococreatortools.create_annotation_info(
                    segmentation_id_test, image_id_test, category_info, binary_mask,
                    image.size, tolerance=2)
                poly_overlap = pycococreatortools.binary_mask_to_polygon(binary_overlap, tolerance=2)
                poly_overlap_else = pycococreatortools.binary_mask_to_polygon(binary_overlap_else, tolerance=2)
                annotation_info_test.update(overlap=poly_overlap)
                annotation_info_test.update(overlap_else=poly_overlap_else)
            elif count_s > 394 and count_s <= 591:
                annotation_info_val = pycococreatortools.create_annotation_info(
                    segmentation_id_val, image_id_val, category_info, binary_mask,
                    image.size, tolerance=2)
                poly_overlap = pycococreatortools.binary_mask_to_polygon(binary_overlap, tolerance=2)
                poly_overlap_else = pycococreatortools.binary_mask_to_polygon(binary_overlap_else, tolerance=2)
                annotation_info_val.update(overlap=poly_overlap)
                annotation_info_val.update(overlap_else=poly_overlap_else)
            else:
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)
                poly_overlap = pycococreatortools.binary_mask_to_polygon(binary_overlap, tolerance=2)
                poly_overlap_else = pycococreatortools.binary_mask_to_polygon(binary_overlap_else, tolerance=2)
                annotation_info.update(overlap=poly_overlap)
                annotation_info.update(overlap_else=poly_overlap_else)

            if annotation_info_test is not None:
                if count_s > 0 and count_s <= 197:
                    coco_output_test["annotations"].append(annotation_info_test)
                elif count_s > 394 and count_s <= 591:
                    coco_output_val["annotations"].append(annotation_info_val)
                else:
                    coco_output["annotations"].append(annotation_info)


            # if annotation_info is not None:
            #     if count_s > 800:
            #         coco_output_test["annotations"].append(annotation_info_test)
            #     else:
            #         coco_output["annotations"].append(annotation_info)

            if count_s > 0 and count_s <= 197:
                segmentation_id_test = segmentation_id_test + 1
            elif count_s > 394 and count_s <= 591:
                segmentation_id_val = segmentation_id_val + 1
            else:
                segmentation_id = segmentation_id + 1
        if count_s > 0 and count_s <= 197:
            print(image_id_test)
            image_id_test = image_id_test + 1
        elif count_s > 394 and count_s <= 591:
            print(image_id_val)
            image_id_val = image_id_val + 1
        else:
            print(image_id)
            image_id = image_id + 1
    print(image_id, image_id_val, image_id_test)

        #     if count_s > 800:
        #         segmentation_id_test = segmentation_id_test + 1
        #     else:
        #         segmentation_id = segmentation_id + 1
        # if count_s > 800:
        #     image_id_test = image_id_test + 1
        #     # print(image_id_test)
        # else:
        #     # print(image_id)
        #     image_id = image_id + 1
    # print(image_id,image_id_test)
    with open('{}/instances_overlap5_else_train2017.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    with open('{}/instances_overlap5_else_val2017.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output_val, output_json_file)
    with open('{}/instances_overlap5_else_test2017.json'.format(ROOT_DIR), 'w') as output_json_file_test:
        json.dump(coco_output_test, output_json_file_test)
    # with open('{}/instances_train2017_overlap.json'.format(ROOT_DIR), 'w') as output_json_file:
    #     json.dump(coco_output, output_json_file)
    # with open('{}/instances_test2017_overlap.json'.format(ROOT_DIR), 'w') as output_json_file_test:
    #     json.dump(coco_output_test, output_json_file_test)


if __name__ == "__main__":
    main()