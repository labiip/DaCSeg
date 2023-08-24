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
    },
    {
        'id': 2,
        'name': 'overlap',
        'supercategory': 'chromosome',
    }
]


# def filter_for_jpeg(root, files):
#     file_types = ['*.jpeg', '*.jpg', '*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     files = [os.path.join(root, f) for f in files]
#     files = [f for f in files if re.match(file_types, f)]
#     return files
#
#
# def filter_for_annotations(root, files, image_filename):
#     file_types = ['*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0] #os.path.basename返回最后文件名 os.path.splitext分离文件名和路径
#     file_name_prefix = basename_no_extension + '.*'
#     files = [os.path.join(root, f) for f in files]
#     files = [f for f in files if re.match(file_types, f)]
#     files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
#     return files


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
    # filter for jpeg images
    # for root, _, files in os.walk(IMAGE_DIR):
    # image_files = filter_for_jpeg(root, files)
    image_files = os.listdir(IMAGE_DIR)

    annotations = np.load("D:/E/data/annotations_with_overlap.npy", allow_pickle=True).item()
    for image_filename in image_files:
        # go through each image
        count_s += 1
        image_dir = os.path.join(IMAGE_DIR, image_filename)
        image = Image.open(image_dir)  # 打开图片 RGB picture
        # image2 =cv2.imread(IMAGE_DIR+image_filename) BGR picture
        # print(IMAGE_DIR+image_filename)
        # cv2.imwrite("./images/"+image_filename.split(".")[0]+".jpg",image2)
        if count_s > 0 and count_s <= 197:
            image_info_test = pycococreatortools.create_image_info(
                image_id_test, os.path.basename(image_filename), image.size)
            coco_output_test["images"].append(image_info_test)
        elif count_s > 197 and count_s <= 394:
            image_info_val = pycococreatortools.create_image_info(
                image_id_val, os.path.basename(image_filename), image.size)
            coco_output_val["images"].append(image_info_val)
        else:
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
            coco_output["images"].append(image_info)

        Gband_img = annotations[image_filename.split(".")[0][:-1]]
        hbox = Gband_img['hbox']
        masks = Gband_img['masks']#box
        overlaps = Gband_img['overlap']#box
        # filter for associated png annotations
        ww, hh = image.size


        for i in range(len(masks)):
            try:
                if overlaps[i] == 0:
                    category_info = {'id': 1, 'is_crowd': 0}
                    binary_mask = np.zeros((hh, ww))
                    temp = masks[i].astype(np.uint8)
                    if (hbox[i][3] - hbox[i][1]) == temp.shape[0]:
                            binary_mask[hbox[i][0]:hbox[i][2], hbox[i][1]:hbox[i][3]] = temp.T
                #                image1[hbox[i][1]:hbox[i][3],hbox[i][0]:hbox[i][2],0] = masks[i].astype(np.uint8)*255
                #                image1[hbox[i][1]:hbox[i][3],hbox[i][0]:hbox[i][2],1] = masks[i].astype(np.uint8)*255
                #                image[hbox[i][1]:hbox[i][3],hbox[i][0]:hbox[i][2],2] = masks[i].astype(np.uint8)*255
                    else:
                    #                print(hbox[i][2]-hbox[i][0],hbox[i][3]- hbox[i][1],temp.shape)
                    #                if temp.shape[0]!=hbox[i][2]-hbox[i][0] or hbox[i][3]- hbox[i][1]!=temp.shape[1]:
                    #                    binary_mask[hbox[i][0]:hbox[i][0]+temp.shape[0],hbox[i][1]:hbox[i][1]+temp.shape[1]] = temp
                    #                else:
                    #                    binary_mask[hbox[i][0]:hbox[i][2],hbox[i][1]:hbox[i][3]] = temp
                        binary_mask[hbox[i][0]:hbox[i][0] + temp.shape[0], hbox[i][1]:hbox[i][1] + temp.shape[1]] = temp

                #                image1[hbox[i][1]:hbox[i][3],hbox[i][0]:hbox[i][2],0] = masks[i].astype(np.uint8).T*255
                #                image1[hbox[i][1]:hbox[i][3],hbox[i][0]:hbox[i][2],1] = masks[i].astype(np.uint8).T*255
                #                image1[hbox[i][1]:hbox[i][3],hbox[i][0]:hbox[i][2],2] = masks[i].astype(np.uint8).T*255

                    if count_s > 0 and count_s <= 197:
                        annotation_info_test = pycococreatortools.create_annotation_info(
                            segmentation_id_test, image_id_test, category_info, binary_mask,
                            image.size, tolerance=2)
                    elif count_s > 197 and count_s <= 394:
                        annotation_info_val = pycococreatortools.create_annotation_info(
                            segmentation_id_val, image_id_val, category_info, binary_mask,
                            image.size, tolerance=2)
                    else:
                        annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)

            except:
                category_info = {'id': 2, 'is_crowd': 0}
                binary_mask_overlap = np.zeros((hh, ww))
                binary_overlap = np.zeros((hh, ww))
                temp_mask = masks[i].astype(np.uint8)
                temp_overlap = overlaps[i].astype(np.uint8)
                # plt.imshow(temp_overlap)
                # plt.show()
                binary_overlap[hbox[i][0]:hbox[i][0] + temp_overlap.shape[0], hbox[i][1]:hbox[i][1] + temp_overlap.shape[1]] = temp_overlap
                # plt.imshow(binary_overlap)
                # plt.show()
                overlap = pycococreatortools.resize_binary_mask(binary_overlap,image.size)
                overlap = pycococreatortools.binary_mask_to_polygon(overlap)
                if (hbox[i][3] - hbox[i][1]) == temp_mask.shape[0]:
                    binary_mask_overlap[hbox[i][0]:hbox[i][2], hbox[i][1]:hbox[i][3]] = temp_mask.T
                else:
                    binary_mask_overlap[hbox[i][0]:hbox[i][0] + temp_mask.shape[0], hbox[i][1]:hbox[i][1] + temp_mask.shape[1]] = temp_mask
                # plt.imshow(binary_mask_overlap)
                # plt.show()
                if count_s > 0 and count_s <= 197:
                    annotation_info_test = pycococreatortools.create_annotation_info(
                        segmentation_id_test, image_id_test, category_info, binary_mask_overlap,
                        image.size, tolerance=2)
                    annotation_info_test['overlap'] = overlap
                elif count_s > 197 and count_s <= 394:
                    annotation_info_val = pycococreatortools.create_annotation_info(
                        segmentation_id_val, image_id_val, category_info, binary_mask_overlap,
                        image.size, tolerance=2)
                    annotation_info_val['overlap'] = overlap
                else:
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask_overlap,
                        image.size, tolerance=2)
                    annotation_info['overlap'] = overlap

            if annotation_info_test is not None:
                if count_s > 0 and count_s <= 197:
                    coco_output_test["annotations"].append(annotation_info_test)
                elif count_s > 197 and count_s <= 394:
                    coco_output_val["annotations"].append(annotation_info_val)
                else:
                    coco_output["annotations"].append(annotation_info)

            if count_s > 0 and count_s <= 197:
                segmentation_id_test = segmentation_id_test + 1
            elif count_s > 197 and count_s <= 394:
                segmentation_id_val = segmentation_id_val + 1
            else:
                segmentation_id = segmentation_id + 1
        if count_s > 0 and count_s <= 197:
            print(image_id_test)
            image_id_test = image_id_test + 1
        elif count_s > 197 and count_s <= 394:
            print(image_id_val)
            image_id_val = image_id_val + 1
        else:
            print(image_id)
            image_id = image_id + 1
    print(image_id, image_id_val, image_id_test)
    with open('{}/instances_overlap1_train2017.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    with open('{}/instances_overlap1_val2017.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output_val, output_json_file)
    with open('{}/instances_overlap1_test2017.json'.format(ROOT_DIR), 'w') as output_json_file_test:
        json.dump(coco_output_test, output_json_file_test)


#
#
if __name__ == "__main__":
    main()