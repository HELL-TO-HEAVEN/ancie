from pathlib import Path

import attr
import numpy as np
from typing import List

import yaml


@attr.s
class Word:
    _FIELDS = tuple(('text', 'xyxy_box'))
    _OPTIONAL_FIELDS = tuple(('id', 'box_ok'))

    text = attr.ib(type=str)
    xyxy_box = attr.ib(type=np.ndarray, repr=False)
    id = attr.ib(default=None, type=str, repr=False)
    box_ok = attr.ib(default=True, type=bool)

    @classmethod
    def from_dict(cls, word_dict: dict):
        for f in cls._FIELDS:
            if f not in word_dict:
                raise AttributeError('word_dict missing required field {}'.format(f))

        return cls(
            text=word_dict['text'],
            xyxy_box=np.array(word_dict['xyxy_box']),
            id = word_dict.get('id'),
            box_ok=word_dict.get('box_ok', False)
        )

    def to_dict(self):
        return {
            'text': self.text,
            'xyxy_box': [float(x) for x in self.xyxy_box],
            'id': self.id,
            'box_ok': self.box_ok
        }

@attr.s
class Document:

    _FIELDS = tuple(('file_path'))

    file_path = attr.ib(type=str, repr=False)
    words = attr.ib(default=None, type=List[dict], repr=False)
    word_count = attr.ib(init=False, repr=True, type=int)

    def add_word(self, word: Word):
        self.words.append(word)
        self.word_count = len(self.words)

    def __attrs_post_init__(self):
        if self.words is None:
            self.words = []
        self.word_count = len(self.words)

    def to_dict(self):
        return {
            'file_path': self.file_path,
            'words': [doc.to_dict() for doc in self.words]
        }

@attr.s
class DocumentCollection:

    name = attr.ib(type=str)
    documents = attr.ib(default=None, type=List[Document], repr=False)
    document_count = attr.ib(init=False, repr=True)
    meta_data = attr.ib(default=None, type=dict, repr=False)

    def __attrs_post_init__(self):
        if self.documents is None:
            self.documents = []
        self.document_count = len(self.documents)

    def __getitem__(self, item):
        return self.documents.__getitem__(item)

    def add_document(self, doc: Document):
        self.documents.append(doc)
        self.document_count = len(self.documents)

    def doc_iterator(self):
        for doc in self.documents:
            yield doc

    def word_iterator(self):
        for doc in self.doc_iterator():
            for word in doc.words:
                yield word

    def to_dict(self):
        return {
            'name': self.name,
            'documents': [doc.to_dict() for doc in self.documents]
        }

    def to_yaml(self, file_path=None):
        f = Path(file_path)
        yaml.dump(self.to_dict(), f.open('w'))


@attr.s
class CollectionStatistics(object):

    collection = attr.ib(type=DocumentCollection, repr=False)
    _word_dict = attr.ib(init=False, type=dict, repr=False)

    def __attrs_post_init__(self):
        self._word_dict = {}


    @property
    def word_count(self):
        wc = 0
        for doc in self.collection.doc_iterator():
            wc += doc.word_count
        return wc

    def get_dictionary(self):
        if not len(self._word_dict):
            for word in self.collection.word_iterator():
                wc = self._word_dict.get(word.text.lower(), 0)
                self._word_dict[word.text.lower()] = wc + 1
        return self._word_dict

