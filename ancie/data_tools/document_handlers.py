from typing import Callable, List, Union

import attr
import torch

from ancie.data_tools import Document
import numpy as np
import cv2

TorchOrNumpy = Union[torch.Tensor, np.ndarray]

@attr.s
class DocumentTensors:
    image = attr.ib(type=TorchOrNumpy, repr=False)
    boxes = attr.ib(type=TorchOrNumpy, repr=False)
    heatmap = attr.ib(default=None, type=TorchOrNumpy, repr=False)
    labels = attr.ib(default=None, type=TorchOrNumpy, repr=False)
    batch_idx = attr.ib(default=None, type=TorchOrNumpy, repr=False)
    shape = attr.ib(init=False, type=List)

    def __attrs_post_init__(self):
        self.shape = list(self.image.shape)

def extract_image_from_document(doc: Document) -> np.ndarray:
    img_path = doc.file_path
    # this is BGR
    img = cv2.imread(str(img_path))
    return img

def extract_boxes_from_document(doc: Document) -> np.ndarray:
    bboxes = np.zeros((doc.word_count, 4), dtype=np.float32)
    for idx, word in enumerate(doc.words):
        bboxes[idx, :] = word.xyxy_box
    return bboxes


def tensor_handler(doc: Document):
    img = extract_image_from_document(doc)
    bboxes = extract_boxes_from_document(doc)

    return DocumentTensors(
        image=img,
        boxes=bboxes,
    )

def tensor_collate_fn(batch: List[DocumentTensors]):
    max_shape = list(np.max([x.shape for x in batch], axis=0))
    b_shape = [len(batch)] + max_shape

    # Assuming same handler generates all DocumentTensors
    batch_boxes = torch.cat([b.boxes for b in batch])
    box_bidx = torch.ones(batch_boxes.shape[0], dtype=torch.int)

    bhmap = None
    if batch[0].heatmap is not None:
        b_shape_hmap = [len(batch)] + max_shape[1:]
        bhmap = torch.zeros(b_shape_hmap, dtype=batch[0].heatmap.dtype)

    blabels = None
    if batch[0].labels is not None:
        lab_max_shape = list(np.max([x.labels.shape for x in batch], axis=0))
        blabels = torch.zeros([len(batch)] + lab_max_shape, dtype=batch[0].labels.dtype)
        pass

    bimg = torch.zeros(b_shape, dtype=batch[0].image.dtype)

    box_bidx_count = 0
    for idx, dt in enumerate(batch):
        bimg[idx, :dt.shape[0], :dt.shape[1], :dt.shape[2]] = dt.image
        if bhmap is not None:
            bhmap[idx, :dt.shape[1], :dt.shape[2]] = dt.heatmap
        if blabels is not None:
            blabels[idx, :dt.labels.shape[0], :dt.labels.shape[1]] = dt.labels
        bc = dt.boxes.shape[0]
        box_bidx[box_bidx_count:box_bidx_count+bc] *= idx
        box_bidx_count += bc

    return DocumentTensors(
        image=bimg,
        boxes=batch_boxes,
        heatmap=bhmap,
        labels=blabels,
        batch_idx=box_bidx
    )


@attr.s
class DocHandler:

    @classmethod
    def tensor_handler(cls, transforms=None):
        if transforms is None:
            transforms = []
        return cls(
            handler_func=tensor_handler,
            collate_func=tensor_collate_fn,
            transforms=transforms
        )

    handler_func = attr.ib(type=Callable)
    transforms = attr.ib(type=List)
    collate_func = attr.ib(default=None, type=Callable)


