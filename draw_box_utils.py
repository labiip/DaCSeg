from PIL.Image import Image, fromarray
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from PIL import ImageColor
import matplotlib.pyplot as plt
import numpy as np

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_text(draw,
              box: list,
              cls: int,
              idc: int,
              score: float,
              category_index: dict,
              color: str,
              font: str = 'arial.ttf',
              font_size: int = 24):
    """
    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()


    left, top, right, bottom = box
    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str = f"{category_index[str(cls)]}{idc}: {int(100 * score)}%"
    display_str_heights = [font.getsize(ds)[1] for ds in display_str]
    # Each display_str has a top and bottom margin of 0.05x.
    display_str_height = (1 + 2 * 0.05) * max(display_str_heights)

    if top > display_str_height:
        text_top = top - display_str_height
        text_bottom = top
    else:
        text_top = bottom
        text_bottom = bottom + display_str_height

    for ds in display_str:
        text_width, text_height = font.getsize(ds)
        margin = np.ceil(0.05 * text_width)
        draw.rectangle([(left, text_top),
                        (left + text_width + 2 * margin, text_bottom)], fill=color)
        draw.text((left + margin, text_top),
                  ds,
                  fill='black',
                  font=font)
        left += text_width


def draw_masks(image, masks, colors, thresh: float = 0.7, alpha: float = 0.5):
    np_image = np.array(image)
    masks = np.where(masks > thresh, True, False)

    # colors = np.array(colors)
    img_to_draw = np.copy(np_image)
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors):
        img_to_draw[mask] = color

    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))
def draw_masks_color2(image, masks, thresh: float = 0.7, alpha: float = 0.8):
    np_image = np.array(image)
    # plt.imshow(np_image)
    # plt.show()
    masks = np.where(masks > thresh, True, False)

    # colors = np.array(colors)
    img_to_draw = np.copy(np_image)
    # TODO: There might be a way to vectorize this
    # for mask in masks:
    #     color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
    #     img_to_draw[mask] = img_to_draw[mask] * 0.2 + np.random.randint(0, 255, size=(1,1,3), dtype=np.uint8) * 0.8
    for mask, color in zip(masks, colors):
        color = np.array(color)
        if (color == (0,0,0)).all():
            pass
        else:
            img_to_draw[mask] = img_to_draw[mask] * 0.4 + color * 0.6
        # plt.imshow(img_to_draw)
        # plt.show()


    out = np_image * (1 - alpha) + img_to_draw * alpha
    return fromarray(out.astype(np.uint8))



def draw_objs(image: Image,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = True):
    """

    Args:

        draw_boxes_on_image:
        draw_masks_on_image:

    Returns:

    """

    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    if len(boxes) == 0:
        return image

    colors = [ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)]) for cls in classes]
    # for cls in range(1,len(classes)):
    #     color = ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)])
    #     colors.append(color)

    # for cls in range(1,2):
    #     color = ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)])
    #     colors.append(color)

    if draw_boxes_on_image:
        # Draw all boxes onto image.
        draw = ImageDraw.Draw(image)

        for box, cls, score, color in zip(boxes, classes, scores, colors):
            left, top, right, bottom = box
            draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=line_thickness, fill=color)
            draw_text(draw, box.tolist(), int(cls), float(score), category_index, color, font, font_size)

    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        image = draw_masks(image, masks, colors, mask_thresh)

    return image

def draw_masks_color(image: Image,
              boxes: np.ndarray = None,
              classes: np.ndarray = None,
              scores: np.ndarray = None,
              masks: np.ndarray = None,
              category_index: dict = None,
              box_thresh: float = 0.1,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = True):

    idxs = np.greater(scores, box_thresh)
    boxes = boxes[idxs]
    classes = classes[idxs]
    scores = scores[idxs]
    if masks is not None:
        masks = masks[idxs]
    if len(boxes) == 0:
        return image

    colors = []
    for cls in range(0,len(boxes)):
        color = ImageColor.getrgb(STANDARD_COLORS[cls % len(STANDARD_COLORS)])
        colors.append(color)

    if draw_boxes_on_image:
        # Draw all boxes onto image.
        draw = ImageDraw.Draw(image)
        idc = 1
        for box, cls, score, color in zip(boxes, classes, scores, colors):

        #obtain idx

            left, top, right, bottom = box

            draw.line([(left, top), (left, bottom), (right, bottom),
            (right, top), (left, top)], width=line_thickness, fill=color)

            # draw_text(draw, box.tolist(), int(cls), int(idc), float(score), category_index, color, font, font_size)
            idc += 1

    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        image = draw_masks_color2(image, masks, mask_thresh)

    return image

def draw_overlaps(image: Image,
              boxes: np.ndarray = None,
              masks: np.ndarray = None,
              mask_thresh: float = 0.5,
              line_thickness: int = 8,
              font: str = 'arial.ttf',
              font_size: int = 24,
              draw_boxes_on_image: bool = True,
              draw_masks_on_image: bool = True):
    """

    Args:
        mask_thresh:
        draw_boxes_on_image:
        draw_masks_on_image:

    Returns:

    """
    try:
        font = ImageFont.truetype(font, font_size)
    except IOError:
        font = ImageFont.load_default()
    # if draw_boxes_on_image:
    #     # Draw all boxes onto image.
    #     draw = ImageDraw.Draw(image)
    #     idc = 1
    #     for box, color in zip(boxes, colors):
    #
    #         left, top, right, bottom = box
    #         draw.line([(left, top), (left, bottom), (right, bottom),
    #                    (right, top), (left, top)], width=line_thickness, fill=color)
    #
    #
    #         display_str = f"{idc}"
    #         # display_str_heights = [font.getsize(ds)[0] for ds in display_str]
    #
    #         # Each display_str has a top and bottom margin of 0.05x.
    #         display_str_height = 12.0
    #
    #         if top > display_str_height:
    #             text_top = top - display_str_height
    #             text_bottom = top
    #         else:
    #             text_top = bottom
    #             text_bottom = bottom + display_str_height
    #
    #         text_width, text_height = 11,6
    #         margin = np.ceil(0.05 * text_width)
    #         draw.rectangle([(left, text_top),
    #                         (left + text_width + 2 * margin, text_bottom)], fill=color)
    #         draw.text((left + margin, text_top),
    #                   display_str,
    #                   fill='black',
    #                   font=font)
    #         left += text_width
    #         idc += 1

    if draw_masks_on_image and (masks is not None):
        # Draw all mask onto image.
        image = draw_masks_color2(image, masks, mask_thresh)

    return image


