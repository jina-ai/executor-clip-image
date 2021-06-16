FROM jinaai/jina:2.0.0rc6 as base

# install git
RUN apt-get -y update && apt-get install -y git

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# load default model during image creation TODO mount volume in cache directory
RUN python -c "import clip;clip.load('ViT-B/32', 'cpu', True)"

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# for testing the image
FROM base
RUN pip install -r tests/requirements.txt
RUN pip install .
RUN pytest -s -vv

FROM base
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
