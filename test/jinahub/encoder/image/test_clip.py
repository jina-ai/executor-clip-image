from jina import Flow, Document

from jinahub.encoder.image.clip import CLIPImageEncoder
import numpy as np

def test_clip():
    def on_done(resp):
        print('assert', resp)
    f = Flow().add(uses=CLIPImageEncoder)
    with f:
        f.post(on='test', inputs=[Document(blob=np.ones((3, 224, 224)))], on_done=on_done)