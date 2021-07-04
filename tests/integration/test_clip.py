import operator
import os
from glob import glob

import clip
import numpy as np
import torch
from PIL import Image
from jina import Flow, Document, DocumentArray, requests
from jina.executors import BaseExecutor
from jinahub.encoder.clip_image import CLIPImageEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_clip_any_image_shape():
    f = Flow().add(uses=CLIPImageEncoder)
    with f:
        results = f.post(
            on='/test', inputs=[Document(blob=np.ones((224, 224, 3), dtype=np.uint8))], return_results=True)
        assert len(results[0].docs.get_attributes('embedding')) == 1

        results = f.post(
            on='/test', inputs=[Document(blob=np.ones((100, 100, 3), dtype=np.uint8))], return_results=True)
        assert len(results[0].docs.get_attributes('embedding')) == 1


def test_clip_batch():
    f = Flow().add(uses={
        'jtype': CLIPImageEncoder.__name__,
        'with': {
            'default_batch_size': 10,
            'model_name': 'ViT-B/32',
            'device': 'cpu'
        }
    })
    with f:
        results = f.post(
            on='/test',
            inputs=(Document(blob=np.ones((224, 224, 3), dtype=np.uint8)) for _ in range(25)),
            return_results=True)
        assert len(results[0].docs.get_attributes('embedding')) == 25


def test_traversal_paths():

    blob = np.ones((224, 224, 3), dtype=np.uint8)
    docs = [Document(
        id='root1',
        blob=blob,
        chunks=[
            Document(
                id='chunk11',
                blob=blob,
                chunks=[
                    Document(id='chunk111', blob=blob),
                    Document(id='chunk112', blob=blob),
                ]
            ),
            Document(id='chunk12', blob=blob),
            Document(id='chunk13', blob=blob),
        ]
    )]
    f = Flow().add(uses={
        'jtype': CLIPImageEncoder.__name__,
        'with': {
            'default_traversal_paths': ['c'],
            'model_name': 'ViT-B/32',
            'device': 'cpu'
        }
    })
    with f:
        results = f.post(on='/test', inputs=docs, return_results=True)
        for path, count in [['r', 0], ['c', 3], ['cc', 0]]:
            assert len(DocumentArray(results[0].docs).traverse_flat([path]).get_attributes('embedding')) == count

        results = f.post(on='/test', inputs=docs, parameters={'traversal_paths': ['cc']}, return_results=True)
        for path, count in [['r', 0], ['c', 0], ['cc', 2]]:
            assert len(DocumentArray(results[0].docs).traverse_flat([path]).get_attributes('embedding')) == count


def test_custom_processing():
    f = Flow().add(uses=CLIPImageEncoder)
    with f:
        result1 = f.post(on='/test', inputs=[Document(blob=np.ones((224, 224, 3), dtype=np.uint8))],
                         return_results=True)

    f = Flow().add(uses={
        'jtype': CLIPImageEncoder.__name__,
        'with': {
            'use_default_preprocessing': False,
        }
    })

    with f:
        result2 = f.post(on='/test', inputs=[Document(blob=np.ones((224, 224, 3), dtype=np.float32))],
                         return_results=True)

    assert result1[0].docs[0].embedding is not None
    assert result2[0].docs[0].embedding is not None
    np.testing.assert_array_compare(operator.__ne__, result1[0].docs[0].embedding, result2[0].docs[0].embedding)


def test_no_documents():
    with Flow().add(uses=CLIPImageEncoder) as f:
        results = f.post(on='/test', inputs=[], return_results=True)
        assert results[0].status.code == 0  # SUCCESS


def test_clip_data():
    docs = []
    for file in glob(os.path.join(cur_dir, 'data', '*')):
        pil_image = Image.open(file)
        nd_image = np.array(pil_image)
        docs.append(Document(id=file, blob=nd_image))

    with Flow().add(uses=CLIPImageEncoder) as f:
        results = f.post(on='/test', inputs=docs, return_results=True)
        os.path.join(cur_dir, 'data', 'banana2.png')
        image_name_to_ndarray = {}
        for d in results[0].docs:
            image_name_to_ndarray[d.id] = d.embedding

    def dist(a, b):
        nonlocal image_name_to_ndarray
        a_embedding = image_name_to_ndarray[os.path.join(cur_dir, 'data', f'{a}.png')]
        b_embedding = image_name_to_ndarray[os.path.join(cur_dir, 'data', f'{b}.png')]
        return np.linalg.norm(a_embedding - b_embedding)

    # assert semantic meaning is captured in the encoding
    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satellite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
    assert small_distance < dist('banana2', 'satellite')
    assert small_distance < dist('banana2', 'studio')
    assert small_distance < dist('airplane', 'studio')
    assert small_distance < dist('airplane', 'satellite')
    assert small_distance < dist('studio', 'satellite')

    # assert same results like calculating it manually
    model, preprocess = clip.load('ViT-B/32', device='cpu')
    assert len(image_name_to_ndarray) == 5
    for file, actual_embedding in image_name_to_ndarray.items():
        image = preprocess(Image.open(file)).unsqueeze(0).to('cpu')

        with torch.no_grad():
            expected_embedding = model.encode_image(image).numpy()[0]

        np.testing.assert_almost_equal(actual_embedding, expected_embedding, 5)
