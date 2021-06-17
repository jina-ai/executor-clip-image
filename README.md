# ClipImageEncoder

 **ClipImageEncoder** is a class that wraps the image embedding functionality from the **CLIP** model.

The **CLIP** model originally was proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf).

`ClipImageEncoder` encode images stored in the blob attribute of the [**Document**](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) and saves the encoding in the embedding attribute.



## Prerequisite

Document, Executor, and Flow are the three fundamental concepts in Jina.

- [**Document**](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Document.md) is the basic data type in Jina;
- [**Executor**](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Executor.md) is how Jina processes Documents;
- [**Flow**](https://github.com/jina-ai/jina/blob/master/.github/2.0/cookbooks/Flow.md) is how Jina streamlines and scales Executors.

*Learn them all, nothing more, you are good to go.*


## Usage

### Via Jinahub

1. Clone the repo and build the Jinahub image

	```shell
	git clone https://github.com/jina-ai/executor-clip-image.git
	cd executor-clip-image
	jina hub push .
	```

1. Use the image in your codes

	```python
	from jinahub.encoder.clip_image import ClipImageEncoder
	from jina import Flow, Document
	import numpy as np
	
	f = Flow().add(
	        uses=jinahub+docker://xgdfljc:v1,
	        volumes='/your_home_folder/.cache/clip:/root/.cache/clip')
	
	def check_emb(resp):
	    for doc in resp.data.docs:
	        if doc.emb:
	            assert doc.emb.shape == (512,)
	
	with f:
		f.post(
		    on='/foo', 
		    inputs=Document(np.random.random((224, 224, 3))), 
		    on_done=check_emb)
	```


### Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-clip-image.git
	cd executor-clip-image
	docker build -t jinahub-clip-image .
	```

1. Use `jinahub-clip-image` in your codes

	```python
	from jinahub.encoder.clip_image import ClipImageEncoder
	from jina import Flow, Document
	import numpy as np
	
	f = Flow().add(
	        uses=docker://jinahub-clip-image:latest,
	        volumes='/your_home_folder/.cache/clip:/root/.cache/clip')
	
	def check_emb(resp):
	    for doc in resp.data.docs:
	        if doc.emb:
	            assert doc.emb.shape == (512,)
	
	with f:
		f.post(
		    on='/foo', 
		    inputs=Document(np.random.random((224, 224, 3))), 
		    on_done=check_emb)
	```

### Via Pypi

1. Install the `jinahub-clip-image`

	```bash
	pip install git+https://github.com/jina-ai/executor-clip-image.git
	```
2. Use `jinahub-clip-image` in your code

	```python
	from jinahub.encoder.clip_image import ClipImageEncoder
	from jina import Flow, Document
	import numpy as np
	
	f = Flow().add(uses=ClipImageEncoder)
	
	def check_emb(resp):
	    for doc in resp.data.docs:
	        if doc.emb:
	            assert doc.emb.shape == (512,)
	
	with f:
		f.post(
		    on='/foo', 
		    inputs=Document(np.random.random((224, 224, 3))), 
		    on_done=check_emb)
	```



## Parameters

### Inputs 

An ndarray of the shape `Height x Width x Color Channel`, min=0, max=1. If `use_default_preprocessing` is `true`, input images can have any height and width. Otherwise, the input format has to be 224x224x3.

### Outputs

Write the output ndarray of the shape `EmbeddingDimension` into `Document.embedding` field.


__*We might need to generate docs automatically?*__


## Reference
- https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf
- https://github.com/openai/CLIP

