ARG CONTEXT="runtime"
# FROM registry.k.avito.ru/avito/service-python/cuda/${CONTEXT}:3.8.12-cuda11.5.1-cudnn8.3.1.22
FROM registry.k.avito.ru/avito/service-python/cuda/${CONTEXT}:3.8.6-cuda10.2.89-cudnn7.6.5.32
ARG CONTEXT="runtime"

COPY ./requirements.txt $PROJECT_ROOT/

RUN	python3 -m pip install --upgrade pip==19.3.1
RUN	python3 -m pip install -r requirements.txt
# отдельно скачиваем torch с cu111 версией cuda, чтобы работало на тачках с A100
RUN pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html

COPY . $PROJECT_ROOT
