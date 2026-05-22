FROM python:3.9-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .

RUN pip install numpy==1.26.4

RUN pip install -r requirements.txt

RUN sed -i 's/from pkg_resources import resource_stream/from importlib.resources import open_binary as resource_stream/' \
        /usr/local/lib/python3.9/site-packages/pronouncing/__init__.py

COPY oww_models/ /usr/local/lib/python3.9/site-packages/openwakeword/resources/models/

RUN ln -sf /usr/local/bin/onnx2tf /usr/bin/onnx2tf

COPY train_wake_word.py dataset_helpers.py negative_loader.py ./

ENTRYPOINT ["python3", "train_wake_word.py"]
