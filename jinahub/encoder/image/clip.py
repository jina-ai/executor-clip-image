from typing import Optional

import numpy as np
import torch
from jina import Executor, DocumentArray, requests

import clip


def _batch_generator(sequence, batch_size):
    for i in range(0, len(sequence), batch_size):
        yield sequence[i:i + batch_size]


class CLIPImageEncoder(Executor):
    """Encode image into embeddings."""

    def __init__(self, model_name: str = 'ViT-B/32', device: str = None, default_batch_size: int = 32,
                 default_traversal_path: str = 'r', *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not device:
            device = "cuda" if not device and torch.cuda.is_available() else "cpu"
        self.default_batch_size = default_batch_size
        self.default_traversal_path = default_traversal_path
        model, _ = clip.load(model_name, device)
        self.model = model

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters, **kwargs):
        """
        Encode all docs with images and store the encodings in the embedding attribute of the docs.
        :param docs: documents sent to the encoder
        :param parameters: dictionary to define the traversal_
        :param kwargs:
        """
        if docs:
            document_batches_generator = self._get_batches(docs, parameters)
            self._create_embeddings(document_batches_generator)

    def _get_batches(self, docs, parameters):
        traversal_path = parameters.get('traversal_path', self.default_traversal_path)
        batch_size = parameters.get('batch_size', self.default_batch_size)

        # traverse thought all documents which have to be processed
        flat_docs = docs.traverse_flat(traversal_path)

        # filter out documents without images
        filtered_docs = DocumentArray(doc for doc in flat_docs if doc.blob is not None)

        document_batches_generator = _batch_generator(filtered_docs, batch_size)
        return document_batches_generator

    def _create_embeddings(self, document_batches_generator):
        with torch.no_grad():
            for document_batch in document_batches_generator:
                blob_batch = document_batch.get_attributes('blob')
                tensor = torch.from_numpy(np.array(blob_batch))
                embedding_batch = self.model.encode_image(tensor)
                numpy_embedding_batch = embedding_batch.cpu().numpy()
                for document, numpy_embedding in zip(document_batch, numpy_embedding_batch):
                    document.embedding = numpy_embedding
