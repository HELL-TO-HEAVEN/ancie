from typing import Union, Callable
from logging import log
import numpy as np
import torch

from ancie.data_tools.bboxes import assign_boxes_to_heatmap_slots, extract_boxes_from_heatmap_slots
from ancie.data_tools.document_handlers import DocumentTensors
from ancie.data_tools.image import scale_and_pad


def crop_to_boxes(dt: DocumentTensors):
    """
    Crop everything but the bounding boxes
    """
    min_xy = np.maximum(0, np.min(dt.boxes[:, :2], axis=0) - 10).astype(np.int)
    max_xy = (np.max(dt.boxes[:, 2:], axis=0) + 10).astype(np.int)
    new_img = dt.image[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :]
    new_boxes = dt.boxes - np.array([min_xy[0], min_xy[1], min_xy[0],min_xy[1]])[None, :]
    return DocumentTensors(
        image=new_img,
        boxes=new_boxes,
        labels=dt.labels,
        batch_idx=dt.batch_idx
    )

def _to_t_if_exists(x: Union[torch.Tensor, np.ndarray], chans_lead=False):
    if not isinstance(x, torch.Tensor) and x is not None:
        if chans_lead and x.ndim == 3:
            x = x.transpose(2, 0, 1)
        x = torch.tensor(x)
    return x


def to_pytorch_tensors(dt: DocumentTensors):
    img = _to_t_if_exists(dt.image, chans_lead=True)
    bboxes = _to_t_if_exists(dt.boxes)
    labels = _to_t_if_exists(dt.labels)
    heamap = _to_t_if_exists(dt.heatmap)

    return DocumentTensors(
        image=img,
        boxes=bboxes,
        labels=labels,
        heatmap=heamap
    )


def get_resize_transform(target_hw, model_shrinkage_factor: int=None):

    if model_shrinkage_factor is not None:
        th = target_hw[0] - target_hw[0] % model_shrinkage_factor
        tw = target_hw[1] - target_hw[1] % model_shrinkage_factor
        log(
            level=30,
            msg='Changing target_hw to {th:},{tw:} to fit model shrinkager of {model_shrinkage_factor:}'.format(**locals())
        )
        target_hw = tuple((th, tw))

    def resizer(dt: DocumentTensors):
        img = dt.image
        if model_shrinkage_factor is not None:
            h = img.shape[0] - img.shape[0] % model_shrinkage_factor
            w = img.shape[1] - img.shape[1] % model_shrinkage_factor
            img = img[:h, :w, :]


        bboxes = dt.boxes
        resized_img, scaled_boxes = scale_and_pad(
            image=img,
            boxes=bboxes,
            target_hw=target_hw
        )

        return DocumentTensors(
            image=resized_img,
            boxes=scaled_boxes,
            labels=dt.labels,
            batch_idx=dt.batch_idx
        )

    return resizer


def get_heatmap_maker(boundry=0.1) -> Callable:
    """
    Produce heatmpas with 3 classes: 0 - background, 1 - boundry, 2 - foreground
    :param boundry: percentange of box periphery to define as boundry class
    :return:
    """

    def make_heatmaps(dt: DocumentTensors):
        img = dt.image
        boxes = dt.boxes
        heatmap = np.zeros(img.shape[:2], dtype=np.int32)
        for fbox in boxes:
            box = fbox.astype(np.int)
            bw = box[2] - box[0]
            dbw = int(boundry*bw)
            bh = box[3] - box[1]
            dbh = int(boundry*bh)

            heatmap[box[1]:box[3], box[0]:box[2]] = 1
            heatmap[box[1]+dbh:box[3]-dbh, box[0]+dbw:box[2]-dbw] = 2

        dt.heatmap = heatmap
        return dt

    return make_heatmaps

def get_box_assigner(model_shrinkae_facotr: float) -> Callable:

    def assign_boxes(dt: DocumentTensors):
        img_shape = dt.image.shape
        fmap_h = np.ceil(img_shape[0] / model_shrinkae_facotr).astype(np.int)
        fmap_w = np.ceil(img_shape[1] / model_shrinkae_facotr).astype(np.int)
        pix_per_h = img_shape[0] / fmap_h
        pix_per_w = img_shape[1] / fmap_w

        assignment_status, ass_cords = assign_boxes_to_heatmap_slots(
            fmap_wh=(fmap_w, fmap_h),
            pixels_per_fmap_wh=(pix_per_w, pix_per_h),
            bboxes=dt.boxes
        )

        labels = np.concatenate([ass_cords, assignment_status[:, None]], axis=1)
        dt.labels = labels
        return dt

    return assign_boxes