import numpy as np
from typing import Tuple

BOX_DTYPE = np.float32

def get_empty_bbox():
    return np.zeros(4, dtype=BOX_DTYPE)

def xywh_to_xyxy(box):
    """BBox transform"""
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return np.array([box[0], box[1], x2, y2])

def _point_in_box(point, box):
    """
    :param point: np.array([x,y])
    :param box: np.array([x1,y1, x2, y2])
    :return: Bool
    """
    truth_list = [(point[i] <= box[i::2][1]) & (point[i] >= box[i::2][0]) for i in range(2)]
    return int(all(truth_list))

# def _build_relative_center_points(fmap_w, fmap_h):
#     sh_x, sh_y = np.meshgrid(np.arange(fmap_w), np.arange(fmap_h))
#     pts = np.vstack((sh_x.ravel(), sh_y.ravel())).transpose()
#     cntr_pts = pts + np.array([0.5] * 2, np.float32)[None, :]
#     relative_pts = cntr_pts / np.array([fmap_w, fmap_h], np.float32)[None, :]
#     return relative_pts

def _build_relative_center_points(fmap_w, fmap_h):
    hs = np.tile((0.5 + np.arange(fmap_h)) / fmap_h, fmap_w)
    ws = np.repeat((0.5 + np.arange(fmap_w)) / fmap_w, fmap_h)
    relative_pts = np.stack([ws, hs], axis=1)
    return relative_pts


def assign_boxes_to_heatmap_slots(
        fmap_wh: Tuple[int, int],
        pixels_per_fmap_wh:Tuple[float, float],
        bboxes: np.ndarray
):
    fmap_w, fmap_h = fmap_wh
    ppf_w, ppf_h = pixels_per_fmap_wh
    t_w = round(fmap_w*ppf_w)
    t_h = round(fmap_h*ppf_h)

    relative_pts = _build_relative_center_points(fmap_h=fmap_h, fmap_w=fmap_w)
    relative_boxes = bboxes / np.array([t_w, t_h, t_w, t_h])
    box_centers = (relative_boxes[:, :2] + relative_boxes[:, 2:]) / 2
    box_wh = relative_boxes[:, 2:] - relative_boxes[:, :2]
    cntr_shift_vec = (box_centers[None, :, :] - relative_pts[:, None, :])
    dists = np.argsort(np.sum(cntr_shift_vec**2, axis=2).flatten())
    gt_map = np.repeat(-1, box_centers.shape[0])
    slot_map = np.repeat(-1, relative_pts.shape[0])
    for idx in dists:
        gt = idx % box_centers.shape[0]
        slot = idx // box_centers.shape[0]
        if gt_map[gt] < 0 and slot_map[slot] < 0:
            gt_map[gt] = slot
            slot_map[slot] = gt
            continue
        if np.all(gt_map >= 0):
            break
    target_cords = np.zeros((relative_pts.shape[0], 4), np.float32)
    target_cords[gt_map, :2] = cntr_shift_vec[gt_map, np.arange(gt_map.shape[0]), :]
    target_cords[gt_map, 2:] = box_wh

    assignment_status = (slot_map >= 0).astype(np.int)
    return assignment_status, target_cords

def extract_boxes_from_heatmap_slots(fmap_wh, cords, status):
    fmap_w, fmap_h = fmap_wh
    relative_pts = _build_relative_center_points(fmap_h=fmap_h, fmap_w=fmap_w)
    pos_boxes = status > 0
    box_centers = cords[pos_boxes, :2] + relative_pts[pos_boxes, :]
    dx = (cords[pos_boxes, None, 2:] / 2) * np.array([-1, 1]).reshape(1,2,1)
    out_boxes = (box_centers[:, None, :] + dx).reshape(-1, 4)

    return out_boxes