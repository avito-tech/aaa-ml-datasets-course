FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade poetry==2.1.3 poetry-plugin-export==1.9.0 poetry-core==2.1.3

COPY pyproject.toml .

RUN poetry config virtualenvs.create false

RUN poetry install --no-root

COPY . .

RUN mkdir -p /data/images

CMD ["bash"] 
