# ClipImageEncoder

 **ClipImageEncoder** is a class that wraps the image embedding functionality from the **CLIP** model.

The **CLIP** model originally was proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf).

`ClipImageEncoder` encode images stored in the blob attribute of the **Document** and saves the encoding in the embedding attribute.

- Input shape: `Color Channel x Height x Width)`

- Output shape: `EmbeddingDimension`

      

## Example:

Here is an example usage of the clip encoder.

```python
    def process_response(resp):
        ...

    f = Flow().add(uses={
        'jtype': CLIPImageEncoder.__name__,
        'with': {
            'default_batch_size': 32,
            'model_name': 'ViT-B/32',
            'device': 'cpu'
        }
    })
    with f:
        f.post(on='/test', inputs=(Document(blob=np.ones((3, 224, 224))) for _ in range(25)), on_done=process_response)
```
