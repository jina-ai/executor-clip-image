from jina import Flow, Document, DocumentArray

from jinahub.encoder.image.clip import CLIPImageEncoder
import numpy as np


def test_clip_no_batch():
    def on_done(resp):
        assert 1 == len(DocumentArray(resp.data.docs).get_attributes('embedding'))

    f = Flow().add(uses=CLIPImageEncoder)
    with f:
        f.post(on='test', inputs=[Document(blob=np.ones((3, 224, 224)))], on_done=on_done)

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
        f.post(on='test', inputs=(Document(blob=np.ones((3, 224, 224))) for _ in range(25)), on_done=validate_callback)
