import cv2
import numpy as np


def scale_and_pad(image, boxes, target_hw):
    target_y, target_x = target_hw
    if float(image.shape[0]) / float(image.shape[1]) < target_y / target_x:
        f = float(target_x) / image.shape[1]
        dsize = (target_x, int(image.shape[0] * f))
    else:
        f = float(target_y) / image.shape[0]
        dsize = (int(image.shape[1] * f), target_y)

    image = cv2.resize(image, dsize=dsize)

    scaled_boxes = boxes * np.atleast_2d(np.array([f, f, f, f]))

    resized_image = cv2.copyMakeBorder(
        image,
        top=0,
        left=0,
        right=target_x - image.shape[1],
        bottom=target_y - image.shape[0],
        borderType=cv2.BORDER_REPLICATE
    )

    return resized_image, scaled_boxes
