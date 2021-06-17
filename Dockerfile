FROM jinaai/jina:2.0.0rc6 as base

# install git
RUN apt-get -y update && apt-get install -y git

# install requirements before copying the workspace
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# for testing the image
FROM base AS test
RUN pip install -r tests/requirements.txt
RUN pip install .
RUN pytest -s -vv

FROM base
RUN rm -rf tests/
ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
