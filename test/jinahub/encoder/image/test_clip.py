from jina import Flow, Document, DocumentArray

from jinahub.encoder.image.clip import CLIPImageEncoder
import numpy as np


def test_clip_no_batch():
    def validate_callback(resp):
        assert 1 == len(DocumentArray(resp.data.docs).get_attributes('embedding'))

    f = Flow().add(uses=CLIPImageEncoder)
    with f:
        f.post(on='/test', inputs=[Document(blob=np.ones((3, 224, 224)))], on_done=validate_callback)


def test_clip_batch():
    def validate_callback(resp):
        assert 25 == len(DocumentArray(resp.data.docs).get_attributes('embedding'))

    f = Flow().add(uses={
        'jtype': CLIPImageEncoder.__name__,
        'with': {
            'default_batch_size': 10,
            'model_name': 'ViT-B/32',
            'device': 'cpu'
        }
    })
    with f:
        f.post(on='/test', inputs=(Document(blob=np.ones((3, 224, 224))) for _ in range(25)), on_done=validate_callback)


def test_traversal_path():
    def validate_default_traversal(resp):
        for path, count in [['r', 0], ['c', 3], ['cc', 0]]:
            assert len(DocumentArray(resp.data.docs).traverse_flat([path]).get_attributes('embedding')) == count

    def validate_request_traversal(resp):
        for path, count in [['r', 0], ['c', 0], ['cc', 2]]:
            assert len(DocumentArray(resp.data.docs).traverse_flat([path]).get_attributes('embedding')) == count

    blob = np.ones((3, 224, 224))
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
            'default_traversal_path': ['c'],
            'model_name': 'ViT-B/32',
            'device': 'cpu'
        }
    })
    with f:
        f.post(on='/test', inputs=docs, on_done=validate_default_traversal)
        f.post(on='/test', inputs=docs, parameters={'traversal_path': ['cc']}, on_done=validate_request_traversal)
