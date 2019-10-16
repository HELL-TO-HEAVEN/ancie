import numpy as np
import cv2
from typing import List, Tuple


def draw_boxes(img, boxes, box_texts: List=None, rect_color: Tuple=(255,0,255), font_color: Tuple=(0, 120, 0)):
    img = np.ascontiguousarray(img)
    for i, bbox in enumerate(boxes):
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), rect_color, thickness=1)
        if box_texts is not None:
            t = str(box_texts[i]) if isinstance(box_texts, list) else box_texts
            fontFace = 0
            fontScale = .2 * (img.shape[0] / 720.0)
            thickness = 1
            fg = font_color
            textSize, baseline = cv2.getTextSize(t, fontFace, fontScale, thickness)
            cv2.putText(img, t, (int(bbox[0]), int(bbox[1] + textSize[1] + baseline / 2)),
                        fontFace, fontScale, fg, thickness)
    return img

def show_img_boxes(debugimg, boxes, gt_boxes=None, titles=None, wait=None):

    if gt_boxes is not None:
        debugimg = draw_boxes(
            img=debugimg,
            boxes=gt_boxes,
            box_texts=titles,
            rect_color=(0, 255, 0)
        )

    debugimg = draw_boxes(
        img=debugimg,
        boxes=boxes,
        box_texts=titles,
        rect_color=(255,0,255)
    )

    cv2.imshow('debug', debugimg)
    cv2.waitKey(delay=wait)
    return debugimg