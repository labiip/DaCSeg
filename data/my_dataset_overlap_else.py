import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO
from train_utils import coco_remove_images_without_annotations, convert_coco_poly_mask


class Chromdataset_else(data.Dataset):
    """`MS Coco Detection <https://cocodataset.org/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    # def __init__(self, root, dataset="train", fold="1", transforms=None, years="2017"):
    #     super(CocoDetection, self).__init__()
    #     assert dataset in ["train", "val","test"], 'dataset must be in ["train", "val"]'
    #     anno_file = f"instances_fold{fold}_{dataset}{years}.json"
    #     assert os.path.exists(root), "file '{}' does not exist.".format(root)
    #     # self.img_root = os.path.join(root, f"{dataset}{years}")
    #     self.img_root = os.path.join(root, "imgb")
    #     assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
    #     self.anno_path = os.path.join(root, "annotations", fold , anno_file)
    #     assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)
    #
    #     self.mode = dataset
    #     self.transforms = transforms
    #     self.coco = COCO(self.anno_path)

    def __init__(self, root, dataset="train", transforms=None, years="2017"):
        super(Chromdataset_else, self).__init__()
        assert dataset in ["train", "val", "test"], 'dataset must be in ["train", "val"]'
        anno_file = f"instances_overlap1_else_{dataset}{years}.json"
        assert os.path.exists(root), "file '{}' does not exist.".format(root)
        # self.img_root = os.path.join(root, f"{dataset}{years}")
        self.img_root = os.path.join(root, "imgb")
        assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
        self.anno_path = os.path.join(root, "annotations", anno_file)
        assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.mode = dataset
        self.transforms = transforms
        self.coco = COCO(self.anno_path)


        data_classes = dict([(v["id"], v["name"]) for k, v in self.coco.cats.items()])
        max_index = max(data_classes.keys())  # 90

        coco_classes = {}
        for k in range(0, max_index + 1):
            if k in data_classes:
                coco_classes[k] = data_classes[k]
            else:
                coco_classes[k] = "N/A"

        if dataset == "train":
            json_str = json.dumps(coco_classes, indent=4)
            with open("cocochrom_indices.json", "w") as f:
                f.write(json_str)

        self.coco_classes = coco_classes

        ids = list(sorted(self.coco.imgs.keys()))
        if dataset == "train":

            valid_ids = coco_remove_images_without_annotations(self.coco, ids)
            self.ids = valid_ids
        else:
            self.ids = ids

    def parse_targets(self,
                      img_id: int,
                      coco_targets: list,
                      w: int = None,
                      h: int = None):
        assert w > 0
        assert h > 0


        anno = [obj for obj in coco_targets if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # [xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_mask(segmentations, h, w)

        overlaps = [obj["overlap"] for obj in anno]
        overlapmasks = convert_coco_poly_mask(overlaps, h, w)

        overlap_elses = [obj["overlap_else"] for obj in anno]
        overlapelse_masks = convert_coco_poly_mask(overlap_elses, h, w)

        # for i in range(len(overlaps)):
        #     if len(overlaps[i]) !=0:
        #         overlapmasks = convert_coco_poly_mask(overlaps, h, w)
        #     else:
        #         overlapmasks = None


        # x_max>x_min且y_max>y_min
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        overlapmasks = overlapmasks[keep]
        overlapelse_masks = overlapelse_masks[keep]
        area = area[keep]
        iscrowd = iscrowd[keep]

        # _labels = torch.sum(overlapmasks, dim=[-1, -2], keepdim=False)
        # _labels_b = torch.ones_like(_labels)
        # overlap_labels = torch.where(_labels == 0, _labels, _labels_b)


        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["overlaps"] = overlapmasks
        target["overlap_elses"] = overlapelse_masks
        target["image_id"] = torch.tensor([img_id])
        # target["overlap_labels"] = overlap_labels
        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd
        # mask_all = np.zeros((1200, 1600), dtype=np.uint8)
        # a = len(masks)
        # for i in range(0,len(masks)):
        #     mask_all += masks[i,:,:].numpy()
        # plt.imshow(mask_all)
        # plt.show()

        return target




        # ax = plt.gca()
        # ax.set_autoscale_on(False)
        # polygons = []
        # color = []
        # c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
        # # polygon
        # for seg in ann['segmentation']:
        #     poly = np.array(seg).reshape(
        #                     (int(len(seg) / 2), 2))
        #     polygons.append(Polygon(poly))
        #     color.append(c)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # img_id = self.ids[index]
        # img_id = [img_id]
        # f = open(self.anno_path, 'rb')
        # dict_f = json.load(f)
        # imgAnns = dict_f['annotations']


        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        coco_target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_root, path)).convert('RGB')

        w, h = img.size
        target = self.parse_targets(img_id, coco_target, w, h)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]

        img_info = coco.loadImgs(img_id)[0]
        w = img_info["width"]
        h = img_info["height"]
        return h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# train = CocoDetection("/data/coco2017/", dataset="train")
# print(len(train))
# t = train[0]
# print(t)
