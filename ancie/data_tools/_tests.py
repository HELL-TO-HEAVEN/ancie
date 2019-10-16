from pathlib import Path

from torch.utils.data import DataLoader, RandomSampler

from ancie.data_tools import crop_to_boxes, to_pytorch_tensors, get_resize_transform, get_heatmap_maker
from ancie.data_tools.bboxes import extract_boxes_from_heatmap_slots
from ancie.data_tools.document_handlers import DocHandler
from ancie.data_tools.iamdb_converter import build_dataset
from ancie.data_tools.pytorch_datasets import DocumentCollectionDataset
from ancie.data_tools.transformations import get_box_assigner
from ancie.data_tools.visualization import show_img_boxes

if __name__ == '__main__':
    import numpy as np

    fname = 'D:\\research\\ancie_project\\words.txt'
    word_file = Path(fname)
    image_path = Path('D:\\research\\www.fki.inf.unibe.ch\\DBs\\iamDB\\data\\forms')
    model_shrinkae_facotr = 32

    iam_dataset = build_dataset(
        image_folder=image_path,
        word_file=word_file
    )

    dataset = DocumentCollectionDataset(
        doc_collection=iam_dataset,
        doc_handler=DocHandler.tensor_handler(
            transforms=[
                crop_to_boxes,
                get_resize_transform(
                    target_hw=(1118, 1024),
                    model_shrinkage_factor=model_shrinkae_facotr
                ),
                get_heatmap_maker(),
                get_box_assigner(
                    model_shrinkae_facotr=model_shrinkae_facotr
                ),
                to_pytorch_tensors
            ]
        )
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=dataset.get_collate_fn(),
        sampler=RandomSampler(dataset)
    )

    for batch in loader:
        for idx in range(batch.image.shape[0]):
            bidx = batch.batch_idx == idx
            bboxes = batch.boxes[bidx, :].numpy()

            # Check proper slot assignment
            fmap_h = np.ceil(batch.image.shape[2] / model_shrinkae_facotr).astype(np.int)
            fmap_w = np.ceil(batch.image.shape[3] / model_shrinkae_facotr).astype(np.int)
            boxes = extract_boxes_from_heatmap_slots(
                fmap_wh=(fmap_w, fmap_h),
                status=batch.labels.numpy()[idx, :, -1],
                cords=batch.labels.numpy()[idx, :, :-1],
            )*np.array([batch.image.shape[3], batch.image.shape[2]]*2)

            show_img_boxes(
                debugimg=batch.image[idx].numpy().transpose(1,2,0),
                boxes=boxes,
                wait=900
            )
        pass


