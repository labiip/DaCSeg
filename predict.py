import os
import time
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from network_files import OverlapMaskRCNN
from backbone import resnet50_fpn_backbone
from backbone import resnet101_fpn_backbone
from draw_box_utils import draw_objs
from draw_box_utils import draw_overlaps
from draw_box_utils import draw_masks_color


def create_model(num_classes, box_thresh=0.5):
    backbone = resnet101_fpn_backbone()
    model = OverlapMaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model

# def create_model(num_classes, box_thresh=0.5):
#     backbone = resnet50_fpn_backbone()
#     model = MaskRCNN(backbone,
#                      num_classes=num_classes,
#                      rpn_score_thresh=box_thresh,
#                      box_score_thresh=box_thresh)
#
#     return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 1  # Background not included
    box_thresh = 0.5
    weights_path = "/home/xinyu.fan/model.pth"

    img_path = "/home/xinyu.fan/data/imgb/bsq SZ1700297 G43-0 59B.tif"
    label_json_path = "./cocochrom_indices.json"

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"])
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # load image
    assert os.path.exists(img_path), f"{img_path} does not exits."
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # Validation Mode
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_mask = predictions["masks"].to("cpu").numpy()
        predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]




        # for i,(box,mask) in enumerate(zip(predict_boxes,predict_mask)):
        #     mask += mask
        #     # plt.imshow(mask)
        #     # plt.show()
        #     binary_mask = mask[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        #     # plt.imshow(binary_mask,cmap='gray')
        #     plt.imshow(binary_mask)
        #
        #     # plt.show()
        #     binary_mask = np.zeros((predict_mask.shape[1],predict_mask.shape[2]))


        # mask28 = predictions["masks28"].to("cpu").numpy()
        # mask28 = np.squeeze(mask28, axis=1)
        # binary_mask = np.zeros((mask28.shape[1],mask28.shape[2]))
        # binary_mask = binary_mask + mask28[0,:,:]
        # # binary_mask[binary_mask > 0] = 255
        # plt.imshow(binary_mask)
        # plt.show()
        # for i in range(mask28.shape[0]):
        #     binary_mask = binary_mask + mask28[i,:,:]
        #     plt.imshow(binary_mask)
        #     plt.show()
        # binary_mask[binary_mask > 0] = 255
        # plt.imshow(mask28[0,:,:])
        # plt.show()

        # predict_edge = predictions["mask_edges"].to("cpu").numpy()
        # predict_edge = np.squeeze(predict_edge, axis=1)
        # if "overlap_proposals" in predictions:
        #     predict_overlap_boxes = predictions["overlap_proposals"].to("cpu").numpy()
        #     predict_overlap = predictions["overlap_prob"].to("cpu").numpy()
        #     predict_overlap = np.squeeze(predict_overlap, axis=1)
        #     predict_overlap_boundary = predictions["overlap_boundary_prob"].to("cpu").numpy()
        #     predict_overlap_boundary = np.squeeze(predict_overlap_boundary, axis=1)
        #     predict_overlapelse = predictions["overlapelse_prob"].to("cpu").numpy()
        #     predict_overlapelse = np.squeeze(predict_overlapelse, axis=1)
        #     predict_overlapelse_boundary = predictions["overlapelse_boundary_prob"].to("cpu").numpy()
        #     predict_overlapelse_boundary = np.squeeze(predict_overlapelse_boundary, axis=1)
        #     # predict_else = predictions["elses"].to("cpu").numpy()
        #     # predict_else = np.squeeze(predict_else, axis=1)
        # else:
        #     print("No objects detected!")
        #     return
        #
        # if len(predict_boxes) == 0:
        #     print("No objects detected!")
        #     return

        # plot_img = draw_objs(original_img,
        #                      boxes=predict_boxes,
        #                      classes=predict_classes,
        #                      scores=predict_scores,
        #                      masks=predict_mask,
        #                      category_index=category_index,
        #                      line_thickness=3,
        #                      font='arial.ttf',
        #                      font_size=20)
        # plt.imshow(plot_img)
        # plt.show()
        # plot_img.save("figg2.jpg")
        plot_img = draw_masks_color(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # plot_img.save("paper1_result1.jpg")

        # plot_img.save('a.eps', dpi=600, format='eps')
        # plot_img.savefig('b.eps', dpi=1200, format='eps')
        #
        # plot_overlap = draw_overlaps(original_img,
        #                            boxes=predict_overlap_boxes,
        #                            masks=predict_overlap,
        #                            line_thickness=3,
        #                            font='arial.ttf',
        #                            font_size=20
        #                            )
        # plt.imshow(plot_overlap)
        # plt.show()

        # # # plot_img.save("test_overlap_result.jpg")
        # plot_overlapelse = draw_overlaps(original_img,
        #                            boxes=predict_overlap_boxes,
        #                            masks=predict_overlapelse,
        #                            line_thickness=3,
        #                            font='arial.ttf',
        #                            font_size=20
        #                            )
        # plt.imshow(plot_overlapelse)
        # plt.show()
        #
        # plot_overlap_boundary = draw_overlaps(original_img,
        #                            boxes=predict_overlap_boxes,
        #                            masks=predict_overlap_boundary,
        #                            line_thickness=3,
        #                            font='arial.ttf',
        #                            font_size=20
        #                            )
        # plt.imshow(plot_overlap_boundary)
        # plt.show()
        #
        # plot_overlapelse_boundary = draw_overlaps(original_img,
        #                            boxes=predict_overlap_boxes,
        #                            masks=predict_overlapelse_boundary,
        #                            line_thickness=3,
        #                            font='arial.ttf',
        #                            font_size=20
        #                            )
        # plt.imshow(plot_overlapelse_boundary)
        # plt.show()


if __name__ == '__main__':
    main()

