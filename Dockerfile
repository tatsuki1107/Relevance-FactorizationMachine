FROM python:3.9-slim-buster

WORKDIR /app
RUN apt-get update \
  && apt-get install -y gcc python3-dev\
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade poetry && poetry config virtualenvs.create false

# python modules
COPY src ./src
COPY test ./test
COPY utils ./utils

COPY poetry.lock pyproject.toml ./

RUN poetry install 
