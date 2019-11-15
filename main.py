from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from ancie.data_tools.bboxes import extract_boxes_from_heatmap_slots
from ancie.data_tools import iamdb_converter

from ancie.data_tools.transformations import (
    get_box_assigner,
    crop_to_boxes,
    to_pytorch_tensors,
    get_resize_transform,
    get_heatmap_maker
)

from ancie.data_tools import (
    show_img_boxes,
    DocumentCollectionDataset,
    DocHandler
)

from ancie.models.feature_extractor import (
    bottleneck_resnet,
    HeatMapGenerator
)

def build_networks(hmap_channels):
    fmap_encoder, model_shrinkage_facotr_hw, fmap_channels = bottleneck_resnet()
    assert model_shrinkage_facotr_hw[0] == model_shrinkage_facotr_hw[1], "Feature Map shrinkage not same for H and W"
    model_shrinkage_facotr = int(model_shrinkage_facotr_hw[0])

    heatmap_generator = HeatMapGenerator(in_channels=fmap_channels, out_channels=hmap_channels)

    return fmap_encoder, heatmap_generator, model_shrinkage_facotr


if __name__ == '__main__':
    import numpy as np

    fname = 'D:\\research\\ancie_project\\words.txt'
    word_file = Path(fname)
    image_path = Path('D:\\research\\www.fki.inf.unibe.ch\\DBs\\iamDB\\data\\forms')

    iam_dataset = iamdb_converter.build_dataset(
        image_folder=image_path,
        word_file=word_file
    )

    fmap_encoder, heatmap_generator, model_shrinkage_facotr = build_networks(
        hmap_channels=3
    )

    doc_handler = DocHandler.tensor_handler(
        transforms=[
            crop_to_boxes,
            get_resize_transform(
                target_hw=(1118, 1024),
                model_shrinkage_factor=model_shrinkage_facotr
            ),
            get_heatmap_maker(),
            get_box_assigner(
                model_shrinkae_facotr=model_shrinkage_facotr
            ),
            to_pytorch_tensors
        ]
    )

    dataset = DocumentCollectionDataset(
        doc_collection=iam_dataset,
        doc_handler=doc_handler
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=dataset.get_collate_fn(),
        sampler=RandomSampler(dataset)
    )


    hmap_xent_loss = torch.nn.CrossEntropyLoss(reduction='mean', weight=torch.tensor([1., 3., 3.]))
    softmax = torch.nn.Softmax(dim=1)


    for batch in loader:

        # (1) Features
        # in (bsize, 3, h, w) out (bsize, ch, h // 32, w // 32)
        feature_map = fmap_encoder(batch.image.float())

        # (2) Heatmap
        # in (bsize, ch, h // 32, w // 32) out (bsize, num_hmap_cls, h, w)
        hmap_logits = heatmap_generator(feature_map)
        # outs a scalar
        hmap_loss = hmap_xent_loss(hmap_logits, batch.heatmap.long())

        # (4) Regression
        hmap = softmax(hmap_logits)
        # A net that takes hmap as input and predicts labels
        # (bsize, fmap_h*fmap_w, 5) - dist from feature center & width, height of box. last cord flag if feature
        # assigned to box
        batch.labels




        for idx in range(batch.image.shape[0]):
            bidx = batch.batch_idx == idx
            bboxes = batch.boxes[bidx, :].numpy()

            # Check proper slot assignment
            fmap_h = np.ceil(batch.image.shape[2] / model_shrinkage_facotr).astype(np.int)
            fmap_w = np.ceil(batch.image.shape[3] / model_shrinkage_facotr).astype(np.int)
            boxes = extract_boxes_from_heatmap_slots(
                fmap_wh=(fmap_w, fmap_h),
                status=batch.labels.numpy()[idx, :, -1],
                cords=batch.labels.numpy()[idx, :, :-1],
            ) * np.array([batch.image.shape[3], batch.image.shape[2]] * 2)

            show_img_boxes(
                debugimg=batch.image[idx].numpy().transpose(1, 2, 0),
                boxes=boxes,
                wait=900
            )
        pass
