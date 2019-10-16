from typing import Callable

from torch.utils.data import Dataset

from ancie.data_tools import DocumentCollection
from ancie.data_tools.document_handlers import DocHandler


class DocumentCollectionDataset(Dataset):

    def __init__(self, doc_collection: DocumentCollection, doc_handler: DocHandler):
        super(DocumentCollectionDataset, self).__init__()
        self.doc_collection = doc_collection
        self.doc_handler = doc_handler

    def __len__(self):
        return self.doc_collection.document_count

    def __getitem__(self, idx):
        # doc = self.doc_collection.documents[idx]
        doc = self.doc_collection[idx]
        handler_out = self.doc_handler.handler_func(doc)

        for f in self.doc_handler.transforms:
            handler_out = f(handler_out)

        return handler_out

    def get_collate_fn(self):
        return self.doc_handler.collate_func


