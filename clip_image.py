from typing import Optional, Iterable, Any, List, Union

import clip
import numpy as np
import torch
from PIL import Image
from jina import Executor, DocumentArray, requests


def _batch_generator(data: List[Any], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]


class CLIPImageEncoder(Executor):
    """
    Encode image into embeddings.

    :param model_name: use clip.available_models() to see all available models: ['RN50', 'RN101', 'RN50x4', 'ViT-B/32']
        - 'ViT-B/32': CLIP model based on the Vision Transformer architecture
        - 'RN50': CLIP model based on ResNet-50
        - 'RN50x4': CLIP model based on ResNet-50 which is scaled up 4x according to EfficientNet scaling rule
        - 'RN101': CLIP model based on ResNet-101
    :param use_default_preprocessing: if True, the same preprocessing is used which got used during training
    - prevents training-serving gap.
    :param device: device to use for encoding ['cuda', 'cpu] - if not set, the device is detected automatically
    :param default_batch_size: fallback batch size in case there is not batch size sent in the request
    :param default_traversal_paths: fallback traversal path in case there is not traversal path sent in the request
    :param jit: Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    """

    def __init__(
            self,
            model_name: str = 'ViT-B/32',
            use_default_preprocessing: bool = True,
            device: Optional[str] = None,
            default_batch_size: int = 32,
            default_traversal_paths: Union[List[str]] = ['r'],
            jit: bool = True,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.model, self.preprocess = clip.load(model_name, device, jit)
        self.use_default_preprocessing = use_default_preprocessing

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: dict, **kwargs):
        """
        Encode all docs with images and store the encodings in the embedding attribute of the docs.
        :param docs: documents sent to the encoder. The docs must have `blob` of the shape `Height x Width x 3`. By
            default, the input `blob` must be an `ndarray` with `dtype=uint8`. The `Height` and `Width` can have
            arbitrary values. When setting `use_default_preprocessing=False`, the input `blob` must have the size of
            `224x224x3` with `dtype=float32`.
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`. For example,
            `parameters={'traversal_paths': 'r', 'batch_size': 10}` will override the `self.default_traversal_paths` and
            `self.default_batch_size`.
        """
        if docs:
            document_batches_generator = self._get_input_data(docs, parameters)
            self._create_embeddings(document_batches_generator)

    def _get_input_data(self, docs: DocumentArray, parameters: dict):
        trav_paths = parameters.get('traversal_paths', self.default_traversal_paths)
        batch_size = parameters.get('batch_size', self.default_batch_size)

        # traverse thought all documents which have to be processed
        flat_docs = docs.traverse_flat(trav_paths)

        # filter out documents without images
        filtered_docs = [doc for doc in flat_docs if doc.blob is not None]

        return _batch_generator(filtered_docs, batch_size)

    def _create_embeddings(self, document_batches_generator: Iterable):
        with torch.no_grad():
            for document_batch in document_batches_generator:
                blob_batch = [d.blob for d in document_batch]
                if self.use_default_preprocessing:
                    images = [Image.fromarray(blob) for blob in blob_batch]
                    tensors = [self.preprocess(img) for img in images]
                    tensor = torch.stack(tensors)
                else:
                    tensor = torch.from_numpy(np.array([np.moveaxis(b, -1, 0) for b in blob_batch]))
                tensor = tensor.to(self.device)
                embedding_batch = self.model.encode_image(tensor)
                numpy_embedding_batch = embedding_batch.cpu().numpy()
                for document, numpy_embedding in zip(document_batch, numpy_embedding_batch):
                    document.embedding = numpy_embedding
