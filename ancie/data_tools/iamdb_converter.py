import attr
import numpy as np
import yaml
from pathlib import Path
from ancie.data_tools import Word, Document, DocumentCollection, CollectionStatistics
from ancie.data_tools.bboxes import BOX_DTYPE, get_empty_bbox, xywh_to_xyxy

def _proc_word_id_to_form_id(word_id):
    tmp = word_id.split('-')[:2]
    form_id = '{:s}-{:s}'.format(*tmp)
    return form_id

def _proc_iamdb_metadata_line(line):
    if line.startswith('#'):
        return None
    struct = line.split(' ')
    id = struct[0]
    correct_segment = struct[1]
    xywh_box = get_empty_bbox()
    for idx, orig_idx in enumerate(range(3, 7)):
        try:
            xywh_box[idx] = float(struct[orig_idx])
        except ValueError:
            return None
    xyxy_box = xywh_to_xyxy(xywh_box)
    if np.any(xyxy_box < 0):
        return None

    word = struct[8].strip()

    return Word(
        id=id,
        xyxy_box=xyxy_box,
        text=word,
        box_ok= correct_segment == 'ok'
    )

def extract_doc_to_words_for_iamdb(word_file: Path) -> dict:
    doc_base = {}
    for l in word_file.open('r').readlines():
        maybe_word = _proc_iamdb_metadata_line(l)
        if maybe_word is not None:
            doc_id = _proc_word_id_to_form_id(maybe_word.id)
            doc = doc_base.get(doc_id, [])
            doc.append(maybe_word)
            doc_base[doc_id] = doc
    return doc_base

def build_dataset(image_folder: Path, word_file: Path) -> DocumentCollection:
    doc_to_words = extract_doc_to_words_for_iamdb(word_file)
    iamdb_dataset = DocumentCollection(
        name='IAM dataset'
    )
    for doc_id in doc_to_words:
        doc_img_path = image_folder / Path(doc_id).with_suffix('.png')
        if doc_img_path.exists():
            doc = Document(
                file_path=str(doc_img_path),
                words=doc_to_words[doc_id]
            )
            iamdb_dataset.add_document(doc)

    return iamdb_dataset

if __name__ == '__main__':
    fname = 'D:\\research\\ancie_project\\words.txt'
    word_file = Path(fname)
    image_path = Path('D:\\research\\www.fki.inf.unibe.ch\\DBs\\iamDB\\data\\forms')
    iam_dataset = build_dataset(
        image_folder=image_path,
        word_file=word_file
    )

    iam_stats = CollectionStatistics(
        collection=iam_dataset
    )
    print(iam_dataset)
    print(iam_stats.word_count)
    print(len(iam_stats.get_dictionary()))
    print(iam_stats.get_dictionary())
    iam_dataset.to_yaml('D:\\research\\ancie_project\\iam_dataset.yaml')
